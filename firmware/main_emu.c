/* Bare-metal Cortex-M4 harness: encodes one 60 s window of all six channels, reports bits via semihosting, exits. */
#include "codec.h"
#include "window_data.h"
#include <stdint.h>
#include <stdio.h>
static inline int sh_call(int op, void *arg) { register int r0 __asm__("r0") = op; register void *r1 __asm__("r1") = arg; __asm__ volatile("bkpt 0xAB" : "+r"(r0) : "r"(r1) : "memory"); return r0; }
static void sh_puts(const char *s) { sh_call(0x04, (void *)s); }
static void sh_exit(void) { sh_call(0x18, (void *)0x20026); }
static codec_ws_t ws; static uint8_t out[16384];
volatile uint32_t total_bits = 0, total_samples = 0;
int main(void) {
    char buf[96];
    for (int c = 0; c < NCHANS; ++c) {
        uint32_t b = codec_encode(chans[c].x, chans[c].n, chans[c].eps, out, sizeof out, &ws, 0);
        total_bits += b; total_samples += (uint32_t)chans[c].n;
        snprintf(buf, sizeof buf, "%s n=%d bits=%lu\n", chans[c].name, chans[c].n, (unsigned long)b); sh_puts(buf);
    }
    snprintf(buf, sizeof buf, "TOTAL samples=%lu bits=%lu\n", (unsigned long)total_samples, (unsigned long)total_bits); sh_puts(buf);
    sh_exit(); for (;;) {}
}
/* ---- startup ---- */
extern uint32_t _sidata, _sdata, _edata, _sbss, _ebss, _estack;
void Reset_Handler(void) {
    uint32_t *s = &_sidata, *d = &_sdata; while (d < &_edata) *d++ = *s++;
    for (d = &_sbss; d < &_ebss; ++d) *d = 0;
    *(volatile uint32_t *)0xE000ED88 |= (0xFu << 20);   /* CPACR: enable CP10/CP11 (FPU) */
    __asm__ volatile("dsb; isb");
    main(); for (;;) {}
}
void Default_Handler(void) { sh_puts("FAULT\n"); sh_exit(); for (;;) {} }
__attribute__((section(".isr_vector"))) const void *vectors[] = { &_estack, Reset_Handler, Default_Handler, Default_Handler, Default_Handler, Default_Handler, Default_Handler, 0, 0, 0, 0, Default_Handler, Default_Handler, 0, Default_Handler, Default_Handler };
