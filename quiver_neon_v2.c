/*
 * QUIVER PoC v2 — True Scalar vs NEON
 *
 * Key insight from v1: GCC auto-vectorizes __builtin_popcountll into
 * NEON `cnt` on ARMv8, so the "scalar" baseline was already using NEON.
 * 
 * This version uses:
 *   - TRUE scalar: bit-manipulation popcount (no NEON allowed)
 *   - COMPILER scalar: __builtin_popcountll (compiler chooses instructions)
 *   - NEON explicit: hand-written NEON with 128-bit wide processing
 *
 * The real value of explicit NEON:
 *   - 128-bit loads (2 words at once) vs 64-bit loads
 *   - Batched widening accumulation (no overflow checks)
 *   - Algorithms the compiler CAN'T auto-vectorize (select, complex rank)
 */

#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

/* ================================================================
 * TRUE SCALAR — pure bit manipulation, no NEON possible
 * ================================================================ */

/* Textbook SWAR popcount — cannot be auto-vectorized into NEON cnt */
static inline uint32_t true_scalar_popcnt64(uint64_t x) {
    x = x - ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    return (uint32_t)((x * 0x0101010101010101ULL) >> 56);
}

/* Attribute to prevent auto-vectorization of this function */
__attribute__((optimize("no-tree-vectorize")))
uint64_t true_scalar_popcount_block(const uint64_t *data, size_t n_words) {
    uint64_t count = 0;
    for (size_t i = 0; i < n_words; i++) {
        count += true_scalar_popcnt64(data[i]);
    }
    return count;
}

__attribute__((optimize("no-tree-vectorize")))
uint64_t true_scalar_rank(const uint64_t *data, size_t n_words, size_t pos) {
    size_t word_idx = pos / 64;
    size_t bit_idx  = pos % 64;
    uint64_t count = 0;

    for (size_t i = 0; i < word_idx && i < n_words; i++) {
        count += true_scalar_popcnt64(data[i]);
    }

    if (word_idx < n_words && bit_idx > 0) {
        uint64_t mask = (1ULL << bit_idx) - 1;
        count += true_scalar_popcnt64(data[word_idx] & mask);
    }
    return count;
}

__attribute__((optimize("no-tree-vectorize")))
int64_t true_scalar_select(const uint64_t *data, size_t n_words, uint64_t n) {
    for (size_t i = 0; i < n_words; i++) {
        uint32_t popcnt = true_scalar_popcnt64(data[i]);
        if (n < (uint64_t)popcnt) {
            uint64_t word = data[i];
            for (int bit = 0; bit < 64; bit++) {
                if (word & 1) {
                    if (n == 0) return (int64_t)(i * 64 + bit);
                    n--;
                }
                word >>= 1;
            }
        }
        n -= popcnt;
    }
    return -1;
}

__attribute__((optimize("no-tree-vectorize")))
void true_scalar_and_block(const uint64_t *a, const uint64_t *b, uint64_t *out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = a[i] & b[i];
}

__attribute__((optimize("no-tree-vectorize")))
void true_scalar_or_block(const uint64_t *a, const uint64_t *b, uint64_t *out, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = a[i] | b[i];
}

/* ================================================================
 * COMPILER-CHOSEN — let GCC decide (will use NEON cnt on ARMv8)
 * ================================================================ */

uint64_t compiler_popcount_block(const uint64_t *data, size_t n_words) {
    uint64_t count = 0;
    for (size_t i = 0; i < n_words; i++) {
        count += __builtin_popcountll(data[i]);
    }
    return count;
}

/* ================================================================
 * EXPLICIT NEON — hand-written 128-bit wide processing
 * ================================================================ */

uint64_t neon_popcount_block(const uint64_t *data, size_t n_words) {
    const uint8_t *ptr = (const uint8_t *)data;
    size_t byte_len = n_words * 8;
    size_t i = 0;
    uint64x2_t acc = vdupq_n_u64(0);

    while (i + 16 <= byte_len) {
        uint16x8_t batch_acc = vdupq_n_u16(0);
        size_t batch_end = i + 16 * 31;
        if (batch_end > byte_len) batch_end = byte_len;

        while (i + 16 <= batch_end) {
            uint8x16_t v   = vld1q_u8(ptr + i);
            uint8x16_t cnt = vcntq_u8(v);
            uint16x8_t w   = vpaddlq_u8(cnt);
            batch_acc = vaddq_u16(batch_acc, w);
            i += 16;
        }

        uint32x4_t wide32 = vpaddlq_u16(batch_acc);
        uint64x2_t wide64 = vpaddlq_u32(wide32);
        acc = vaddq_u64(acc, wide64);
    }

    uint64_t total = vgetq_lane_u64(acc, 0) + vgetq_lane_u64(acc, 1);

    while (i + 8 <= byte_len) {
        total += __builtin_popcountll(*(const uint64_t *)(ptr + i));
        i += 8;
    }
    return total;
}

uint64_t neon_rank(const uint64_t *data, size_t n_words, size_t pos) {
    size_t word_idx = pos / 64;
    size_t bit_idx  = pos % 64;
    size_t full = word_idx < n_words ? word_idx : n_words;
    uint64_t count = neon_popcount_block(data, full);
    if (word_idx < n_words && bit_idx > 0) {
        uint64_t mask = (1ULL << bit_idx) - 1;
        count += __builtin_popcountll(data[word_idx] & mask);
    }
    return count;
}

int64_t neon_select(const uint64_t *data, size_t n_words, uint64_t n) {
    const uint8_t *ptr = (const uint8_t *)data;
    size_t word_idx = 0;

    while (word_idx + 2 <= n_words) {
        uint8x16_t v    = vld1q_u8(ptr + word_idx * 8);
        uint8x16_t cnt  = vcntq_u8(v);
        uint16x8_t w16  = vpaddlq_u8(cnt);
        uint32x4_t w32  = vpaddlq_u16(w16);
        uint64x2_t w64  = vpaddlq_u32(w32);
        uint64_t chunk_pop = vgetq_lane_u64(w64, 0) + vgetq_lane_u64(w64, 1);
        if (n < chunk_pop) break;
        n -= chunk_pop;
        word_idx += 2;
    }

    for (size_t i = word_idx; i < n_words; i++) {
        uint32_t popcnt = __builtin_popcountll(data[i]);
        if (n < (uint64_t)popcnt) {
            uint64_t word = data[i];
            for (int bit = 0; bit < 64; bit++) {
                if (word & 1) {
                    if (n == 0) return (int64_t)(i * 64 + bit);
                    n--;
                }
                word >>= 1;
            }
        }
        n -= popcnt;
    }
    return -1;
}

void neon_and_block(const uint64_t *a, const uint64_t *b, uint64_t *out, size_t n_words) {
    const uint8_t *pa = (const uint8_t *)a;
    const uint8_t *pb = (const uint8_t *)b;
    uint8_t *po = (uint8_t *)out;
    size_t bytes = n_words * 8;
    size_t i = 0;
    for (; i + 16 <= bytes; i += 16) {
        vst1q_u8(po + i, vandq_u8(vld1q_u8(pa + i), vld1q_u8(pb + i)));
    }
    for (; i < bytes; i++) po[i] = pa[i] & pb[i];
}

void neon_or_block(const uint64_t *a, const uint64_t *b, uint64_t *out, size_t n_words) {
    const uint8_t *pa = (const uint8_t *)a;
    const uint8_t *pb = (const uint8_t *)b;
    uint8_t *po = (uint8_t *)out;
    size_t bytes = n_words * 8;
    size_t i = 0;
    for (; i + 16 <= bytes; i += 16) {
        vst1q_u8(po + i, vorrq_u8(vld1q_u8(pa + i), vld1q_u8(pb + i)));
    }
    for (; i < bytes; i++) po[i] = pa[i] | pb[i];
}

/* ================================================================
 * TIMING
 * ================================================================ */

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

void fill_random(uint64_t *data, size_t n) {
    for (size_t i = 0; i < n; i++)
        data[i] = i * 6364136223846793005ULL + 1442695040888963407ULL;
}

volatile uint64_t sink;

/* ================================================================
 * MAIN
 * ================================================================ */

int main(void) {
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║         QUIVER PoC v2 — True Scalar vs Compiler vs NEON         ║\n");
    printf("║              Running under QEMU aarch64 emulation               ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════╣\n");
    printf("║  Three contenders:                                              ║\n");
    printf("║    TRUE SCALAR : SWAR bit-manipulation, no NEON possible        ║\n");
    printf("║    COMPILER    : __builtin_popcountll (GCC auto-vectorizes)     ║\n");
    printf("║    NEON        : hand-written 128-bit wide SIMD                 ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");

    const size_t sizes[] = {64, 256, 1024, 4096};
    const int n_sizes = 4;

    uint64_t *data  = (uint64_t *)malloc(4096 * 8);
    uint64_t *data2 = (uint64_t *)malloc(4096 * 8);
    uint64_t *out   = (uint64_t *)malloc(4096 * 8);
    fill_random(data, 4096);
    fill_random(data2, 4096);
    for (size_t i = 0; i < 4096; i++) data2[i] ^= 0xCAFEBABEDEADBEEFULL;

    /* ──────── POPCOUNT ──────── */
    printf("━━━ POPCOUNT ━━━\n\n");
    printf("  %-12s  %12s  %12s  %12s   scalar→NEON\n",
           "size", "true scalar", "compiler", "NEON");
    printf("  %-12s  %12s  %12s  %12s   ──────────\n",
           "────", "───────────", "────────", "────");

    for (int si = 0; si < n_sizes; si++) {
        size_t sz = sizes[si];
        size_t iters = sz <= 256 ? 50000 : (sz <= 1024 ? 20000 : 5000);

        uint64_t t0, t1;

        t0 = now_ns();
        for (size_t it = 0; it < iters; it++) sink = true_scalar_popcount_block(data, sz);
        t1 = now_ns();
        double ts_ns = (double)(t1 - t0) / iters;

        t0 = now_ns();
        for (size_t it = 0; it < iters; it++) sink = compiler_popcount_block(data, sz);
        t1 = now_ns();
        double comp_ns = (double)(t1 - t0) / iters;

        t0 = now_ns();
        for (size_t it = 0; it < iters; it++) sink = neon_popcount_block(data, sz);
        t1 = now_ns();
        double neon_ns = (double)(t1 - t0) / iters;

        /* Verify */
        uint64_t a = true_scalar_popcount_block(data, sz);
        uint64_t b = compiler_popcount_block(data, sz);
        uint64_t c = neon_popcount_block(data, sz);
        char ok = (a == b && b == c) ? ' ' : '!';

        printf("  %4zu w %3zuKB  %9.0f ns  %9.0f ns  %9.0f ns   %.2fx %c\n",
               sz, sz*8/1024, ts_ns, comp_ns, neon_ns, ts_ns / neon_ns, ok);
    }

    /* ──────── RANK ──────── */
    printf("\n━━━ RANK ━━━\n\n");
    printf("  %-14s  %12s  %12s   speedup\n", "size", "true scalar", "NEON");
    printf("  %-14s  %12s  %12s   ───────\n", "────", "───────────", "────");

    for (int si = 1; si < n_sizes; si++) {
        size_t sz = sizes[si];
        size_t pos = sz * 64 / 2;
        size_t iters = sz <= 256 ? 50000 : (sz <= 1024 ? 20000 : 5000);

        uint64_t t0, t1;

        t0 = now_ns();
        for (size_t it = 0; it < iters; it++) sink = true_scalar_rank(data, sz, pos);
        t1 = now_ns();
        double ts_ns = (double)(t1 - t0) / iters;

        t0 = now_ns();
        for (size_t it = 0; it < iters; it++) sink = neon_rank(data, sz, pos);
        t1 = now_ns();
        double neon_ns = (double)(t1 - t0) / iters;

        uint64_t a = true_scalar_rank(data, sz, pos);
        uint64_t b = neon_rank(data, sz, pos);
        char ok = (a == b) ? ' ' : '!';

        printf("  %4zu w p=%5zu  %9.0f ns  %9.0f ns   %.2fx %c\n",
               sz, pos, ts_ns, neon_ns, ts_ns / neon_ns, ok);
    }

    /* ──────── SELECT ──────── */
    printf("\n━━━ SELECT ━━━\n\n");
    printf("  %-14s  %12s  %12s   speedup\n", "size", "true scalar", "NEON");
    printf("  %-14s  %12s  %12s   ───────\n", "────", "───────────", "────");

    for (int si = 1; si < n_sizes; si++) {
        size_t sz = sizes[si];
        uint64_t total = true_scalar_popcount_block(data, sz);
        uint64_t target = total / 2;
        size_t iters = sz <= 256 ? 50000 : (sz <= 1024 ? 20000 : 5000);

        uint64_t t0, t1;

        t0 = now_ns();
        for (size_t it = 0; it < iters; it++) sink = (uint64_t)true_scalar_select(data, sz, target);
        t1 = now_ns();
        double ts_ns = (double)(t1 - t0) / iters;

        t0 = now_ns();
        for (size_t it = 0; it < iters; it++) sink = (uint64_t)neon_select(data, sz, target);
        t1 = now_ns();
        double neon_ns = (double)(t1 - t0) / iters;

        int64_t a = true_scalar_select(data, sz, target);
        int64_t b = neon_select(data, sz, target);
        char ok = (a == b) ? ' ' : '!';

        printf("  %4zu w n=%5lu  %9.0f ns  %9.0f ns   %.2fx %c\n",
               sz, (unsigned long)target, ts_ns, neon_ns, ts_ns / neon_ns, ok);
    }

    /* ──────── BULK BITWISE ──────── */
    printf("\n━━━ BULK BITWISE (AND / OR) ━━━\n\n");
    printf("  %-12s  %-10s  %12s  %12s   speedup\n", "size", "op", "true scalar", "NEON");
    printf("  %-12s  %-10s  %12s  %12s   ───────\n", "────", "──", "───────────", "────");

    for (int si = 1; si < n_sizes; si++) {
        size_t sz = sizes[si];
        size_t iters = sz <= 256 ? 50000 : (sz <= 1024 ? 20000 : 5000);
        uint64_t t0, t1;

        /* AND */
        t0 = now_ns();
        for (size_t it = 0; it < iters; it++) { true_scalar_and_block(data, data2, out, sz); sink = out[0]; }
        t1 = now_ns();
        double ts_and = (double)(t1 - t0) / iters;

        t0 = now_ns();
        for (size_t it = 0; it < iters; it++) { neon_and_block(data, data2, out, sz); sink = out[0]; }
        t1 = now_ns();
        double neon_and = (double)(t1 - t0) / iters;

        printf("  %4zu w %3zuKB  %-10s  %9.0f ns  %9.0f ns   %.2fx\n",
               sz, sz*8/1024, "AND", ts_and, neon_and, ts_and / neon_and);

        /* OR */
        t0 = now_ns();
        for (size_t it = 0; it < iters; it++) { true_scalar_or_block(data, data2, out, sz); sink = out[0]; }
        t1 = now_ns();
        double ts_or = (double)(t1 - t0) / iters;

        t0 = now_ns();
        for (size_t it = 0; it < iters; it++) { neon_or_block(data, data2, out, sz); sink = out[0]; }
        t1 = now_ns();
        double neon_or = (double)(t1 - t0) / iters;

        printf("  %4zu w %3zuKB  %-10s  %9.0f ns  %9.0f ns   %.2fx\n",
               sz, sz*8/1024, "OR", ts_or, neon_or, ts_or / neon_or);
    }

    printf("\n═══════════════════════════════════════════════════════════════════\n");
    printf("  INTERPRETATION GUIDE:\n");
    printf("  • speedup > 1.0 = NEON wins over true scalar\n");
    printf("  • QEMU inflates NEON overhead vs real ARM silicon\n");
    printf("  • On real ARM: NEON advantage should be LARGER because\n");
    printf("    vcntq_u8 is single-cycle on Cortex-A, not emulated\n");
    printf("  • '!' after speedup = correctness mismatch (bug!)\n");
    printf("═══════════════════════════════════════════════════════════════════\n");

    free(data);
    free(data2);
    free(out);
    return 0;
}
