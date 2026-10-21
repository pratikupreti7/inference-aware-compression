/* Native test: read vectors (n, eps, x[n], expected py_bits, py_k[n]) and compare. */
#include "codec.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb"); if (!f) return 2;
    static codec_ws_t ws; static float x[CODEC_MAX_N], xh[CODEC_MAX_N], xh2[CODEC_MAX_N]; static int32_t pk[CODEC_MAX_N]; static uint8_t out[32768];
    long tot = 0, same = 0; double bits_c = 0, bits_py = 0, worst = 0; int nwin = 0, bound_viol = 0;
    while (1) {
        int32_t n; float eps; uint32_t pybits;
        if (fread(&n, 4, 1, f) != 1) break;
        fread(&eps, 4, 1, f); fread(x, 4, n, f); fread(&pybits, 4, 1, f); fread(pk, 4, n, f);
        uint32_t b = codec_encode(x, n, eps, out, sizeof out, &ws, xh);
        codec_reconstruct(ws.k, ws.esc_val, n, eps, xh2);
        for (int i = 0; i < n; ++i) {
            double e = fabs((double)xh2[i] - (double)x[i]) / eps; if (e > worst) worst = e; if (e > 1.0 + 1e-5) bound_viol++;
            if (xh2[i] != xh[i]) { printf("decoder mismatch\n"); return 3; }
            int32_t kc = ws.k[i] == CODEC_ESC ? ws.esc_val[i] : ws.k[i]; tot++; same += (kc == pk[i]);
        }
        bits_c += b; bits_py += pybits; nwin++;
    }
    printf("windows=%d samples=%ld identical_indices=%.6f bits_c=%.0f bits_py=%.0f rel_diff=%.5f max_err_over_eps=%.7f bound_violations=%d\n",
           nwin, tot, (double)same / tot, bits_c, bits_py, (bits_c - bits_py) / bits_py, worst, bound_viol);
    return 0;
}
