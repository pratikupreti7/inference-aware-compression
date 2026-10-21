/* Closed-loop Lorenzo + uniform quantization + per-window canonical Huffman. Single precision (Cortex-M4F). */
#include "codec.h"
#include <string.h>
#include <math.h>

static int cmp_i32(const void *a, const void *b) { int32_t x = *(const int32_t *)a, y = *(const int32_t *)b; return (x > y) - (x < y); }
#include <stdlib.h>

/* ---- bit writer ---- */
typedef struct { uint8_t *buf; uint32_t cap; uint32_t bitpos; } bw_t;
static void bw_put(bw_t *w, uint32_t v, int nbits) {
    for (int i = nbits - 1; i >= 0; --i) {
        uint32_t byte = w->bitpos >> 3;
        if (byte < w->cap) { if ((v >> i) & 1u) w->buf[byte] |= (uint8_t)(0x80u >> (w->bitpos & 7)); }
        w->bitpos++;
    }
}

/* Huffman code lengths from counts sorted ascending (two-queue method, O(m)). lens out. */
static void huff_lengths(const uint32_t *cnt_sorted, int m, uint8_t *lens, uint32_t *q2, int32_t *parent) {
    if (m == 1) { lens[0] = 1; return; }
    /* nodes 0..m-1 leaves (counts sorted), internal nodes m..2m-2. parent[] links. q2 holds internal node weights. */
    int i = 0, j = 0, nint = 0;
    for (int t = 0; t < 2 * m - 1; ++t) parent[t] = -1;
    while (i + (nint - j) + (m - i) > 1 && (i < m || j < nint)) {
        int a, b; /* pick two smallest among leaves[i..] and internal[j..nint) */
        if (j < nint && (i >= m || q2[j] <= cnt_sorted[i])) a = m + j++; else a = i++;
        if (j < nint && (i >= m || q2[j] <= cnt_sorted[i])) b = m + j++; else b = i++;
        uint32_t wa = a < m ? cnt_sorted[a] : q2[a - m], wb = b < m ? cnt_sorted[b] : q2[b - m];
        q2[nint] = wa + wb; parent[a] = m + nint; parent[b] = m + nint; nint++;
        if (i >= m && j >= nint - 1 && nint == m - 1) break;
    }
    for (int l = 0; l < m; ++l) { int d = 0; int32_t p = parent[l]; while (p >= 0) { d++; p = parent[p]; } lens[l] = (uint8_t)(d > 31 ? 31 : d); }
}

uint32_t codec_encode(const float *x, int n, float eps, uint8_t *out, uint32_t out_cap, codec_ws_t *ws, float *xhat_opt) {
    const float q = 2.0f * eps, invq = 1.0f / q;
    float xh1 = 0.f, xh2 = 0.f;
    bw_t w = { out, out_cap, 0 };
    if (out) memset(out, 0, out_cap);
    bw_put(&w, (uint32_t)n, 16);
    /* prediction + quantization */
    for (int i = 0; i < n; ++i) {
        float xhat;
        if (i < 2) {
            int32_t kk = (int32_t)floorf(x[i] * invq + 0.5f);
            xhat = (float)kk * q; ws->k[i] = CODEC_ESC; ws->esc_val[i] = kk;
            bw_put(&w, (uint32_t)kk, 32);
        } else {
            float pred = 2.0f * xh1 - xh2;
            float r = x[i] - pred;
            int32_t kk = (int32_t)floorf(r * invq + 0.5f);
            if (kk > CODEC_KMAX || kk < -CODEC_KMAX) {
                int32_t ka = (int32_t)floorf(x[i] * invq + 0.5f);
                xhat = (float)ka * q; ws->k[i] = CODEC_ESC; ws->esc_val[i] = ka;
            } else { xhat = pred + (float)kk * q; ws->k[i] = kk; }
        }
        xh2 = xh1; xh1 = xhat;
        if (xhat_opt) xhat_opt[i] = xhat;
    }
    int m0 = n - 2; if (m0 <= 0) return w.bitpos;
    /* histogram of symbols k[2..n) (ESC included as a symbol) */
    memcpy(ws->sym, ws->k + 2, (size_t)m0 * sizeof(int32_t));
    qsort(ws->sym, (size_t)m0, sizeof(int32_t), cmp_i32);
    int m = 0;
    for (int i = 0; i < m0; ++i) { if (m == 0 || ws->sym[i] != ws->sym[m - 1]) { ws->sym[m] = ws->sym[i]; ws->cnt[m] = 1; m++; } else ws->cnt[m - 1]++; }
    /* sort (count, symbol) ascending by count for the two-queue algorithm: simple insertion on index array */
    static int32_t order[CODEC_MAX_N]; static uint32_t csorted[CODEC_MAX_N]; static uint32_t q2[CODEC_MAX_N]; static int32_t parent[2 * CODEC_MAX_N];
    for (int i = 0; i < m; ++i) order[i] = i;
    for (int i = 1; i < m; ++i) { int32_t v = order[i]; int j = i - 1; while (j >= 0 && ws->cnt[order[j]] > ws->cnt[v]) { order[j + 1] = order[j]; j--; } order[j + 1] = v; }
    for (int i = 0; i < m; ++i) csorted[i] = ws->cnt[order[i]];
    static uint8_t lens_sorted[CODEC_MAX_N];
    huff_lengths(csorted, m, lens_sorted, q2, parent);
    for (int i = 0; i < m; ++i) ws->len[order[i]] = lens_sorted[i];
    /* table: 16-bit count, then (16-bit symbol, 6-bit length) per entry */
    bw_put(&w, (uint32_t)m, 16);
    for (int i = 0; i < m; ++i) { bw_put(&w, (uint32_t)(uint16_t)ws->sym[i], 16); bw_put(&w, ws->len[i], 6); }
    /* canonical code assignment: codes by (length, symbol order) */
    static uint32_t code[CODEC_MAX_N];
    { uint32_t next = 0; for (int L = 1; L <= 31; ++L) { for (int i = 0; i < m; ++i) if (ws->len[i] == L) { code[i] = next++; } next <<= 1; } }
    /* data: binary search symbol -> index (ws->sym sorted ascending) */
    for (int i = 2; i < n; ++i) {
        int32_t s = ws->k[i]; int lo = 0, hi = m - 1, idx = 0;
        while (lo <= hi) { int mid = (lo + hi) >> 1; if (ws->sym[mid] == s) { idx = mid; break; } if (ws->sym[mid] < s) lo = mid + 1; else hi = mid - 1; }
        bw_put(&w, code[idx], ws->len[idx]);
        if (s == CODEC_ESC) bw_put(&w, (uint32_t)ws->esc_val[i], 32);
    }
    return w.bitpos;
}

void codec_reconstruct(const int32_t *k, const int32_t *esc_val, int n, float eps, float *xhat) {
    const float q = 2.0f * eps;
    for (int i = 0; i < n; ++i) {
        if (i < 2 || k[i] == CODEC_ESC) xhat[i] = (float)esc_val[i] * q;
        else xhat[i] = 2.0f * xhat[i - 1] - xhat[i - 2] + (float)k[i] * q;
    }
}
