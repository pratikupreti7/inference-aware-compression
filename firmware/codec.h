#ifndef CODEC_H
#define CODEC_H
#include <stdint.h>
#define CODEC_MAX_N 3840           /* 60 s at 64 Hz */
#define CODEC_KMAX 32767
#define CODEC_ESC  (-CODEC_KMAX - 1)
typedef struct {
    int32_t k[CODEC_MAX_N];        /* quantization indices (ESC marker for escaped samples) */
    int32_t esc_val[CODEC_MAX_N];  /* absolute bin index for escaped samples */
    int32_t sym[CODEC_MAX_N];      /* scratch: sorted symbols */
    uint32_t cnt[CODEC_MAX_N];     /* scratch: counts of unique symbols */
    uint8_t  len[CODEC_MAX_N];     /* scratch: code lengths */
} codec_ws_t;
/* Encode one channel window. Returns number of bits written to out (capacity out_cap bytes). */
uint32_t codec_encode(const float *x, int n, float eps, uint8_t *out, uint32_t out_cap, codec_ws_t *ws, float *xhat_opt);
/* Reconstruct from indices (decoder side of the prediction loop). */
void codec_reconstruct(const int32_t *k, const int32_t *esc_val, int n, float eps, float *xhat);
#endif
