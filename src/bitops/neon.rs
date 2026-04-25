/// ARM NEON-accelerated bit operations.
///
/// These implementations use NEON SIMD intrinsics to process 128 bits
/// at a time. The key advantage over scalar:
/// - `vcntq_u8` counts bits per byte in a single instruction (16 bytes at once)
/// - Horizontal adds (`vpaddlq`) accumulate across vector lanes efficiently
/// - ARM's pipeline handles NEON ops alongside scalar ops
///
/// On x86, this module is replaced by the scalar fallback.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// NEON popcount across a slice of u64 words.
///
/// Strategy: interpret the data as u8x16 vectors, use `vcntq_u8` to get
/// per-byte popcounts, then widen (u8→u16→u32→u64) via `vpaddlq` to
/// accumulate without overflow.
///
/// Processing: 16 bytes (2 x u64) per iteration.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn popcount_block(data: &[u64]) -> u64 {
    unsafe { popcount_block_neon(data) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn popcount_block_neon(data: &[u64]) -> u64 {
    let ptr = data.as_ptr() as *const u8;
    let byte_len = data.len() * 8;
    let mut total: u64 = 0;

    // Accumulator: keeps running sum as u64x2 to avoid overflow
    let mut acc = vdupq_n_u64(0);
    let mut i = 0;

    // Process 16 bytes at a time (= 2 u64 words)
    // Batch: accumulate up to 31 iterations before widening to u64
    // (vcntq_u8 returns 0-8 per byte, vpaddlq_u8 gives 0-16 per u16,
    //  so u16 can hold 31 * 16 = 496 without overflow in u16::MAX=65535)
    while i + 16 <= byte_len {
        let mut batch_acc = vdupq_n_u16(0);
        let batch_end = core::cmp::min(i + 16 * 31, byte_len);

        while i + 16 <= batch_end {
            let v = vld1q_u8(ptr.add(i));           // load 16 bytes
            let cnt = vcntq_u8(v);                    // popcount per byte
            let cnt16 = vpaddlq_u8(cnt);              // pairwise widen u8→u16
            batch_acc = vaddq_u16(batch_acc, cnt16);   // accumulate in u16
            i += 16;
        }

        // Widen batch_acc: u16→u32→u64 and add to accumulator
        let wide32 = vpaddlq_u16(batch_acc);
        let wide64 = vpaddlq_u32(wide32);
        acc = vaddq_u64(acc, wide64);
    }

    // Extract from vector accumulator
    total += vgetq_lane_u64(acc, 0) as u64 + vgetq_lane_u64(acc, 1) as u64;

    // Handle remaining bytes (< 16) with scalar
    let remaining_words = (byte_len - i) / 8;
    let word_start = i / 8;
    for j in 0..remaining_words {
        total += data[word_start + j].count_ones() as u64;
    }

    total
}

/// NEON-accelerated rank: count set bits in positions [0, pos).
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn rank(data: &[u64], pos: usize) -> u64 {
    let word_idx = pos / 64;
    let bit_idx = pos % 64;

    // Count full words using NEON popcount
    let full_words = &data[..core::cmp::min(word_idx, data.len())];
    let mut count = popcount_block(full_words);

    // Handle partial word
    if word_idx < data.len() && bit_idx > 0 {
        let mask = (1u64 << bit_idx) - 1;
        count += (data[word_idx] & mask).count_ones() as u64;
    }

    count
}

/// NEON-accelerated select: find position of n-th set bit.
///
/// Strategy: use NEON popcount to skip full 128-bit chunks quickly,
/// then fall back to scalar for the final word.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn select(data: &[u64], mut n: u64) -> Option<usize> {
    unsafe { select_neon(data, n) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn select_neon(data: &[u64], mut n: u64) -> Option<usize> {
    let ptr = data.as_ptr() as *const u8;
    let mut word_idx = 0;

    // Skip pairs of u64 words (128 bits) using NEON popcount
    while word_idx + 2 <= data.len() {
        let v = vld1q_u8((ptr as *const u8).add(word_idx * 8));
        let cnt = vcntq_u8(v);
        let cnt16 = vpaddlq_u8(cnt);
        let cnt32 = vpaddlq_u16(cnt16);
        let cnt64 = vpaddlq_u32(cnt32);

        let chunk_pop = vgetq_lane_u64(cnt64, 0) as u64
                      + vgetq_lane_u64(cnt64, 1) as u64;

        if n < chunk_pop {
            // Target is in this chunk — check each word
            break;
        }
        n -= chunk_pop;
        word_idx += 2;
    }

    // Scalar search within remaining words
    for i in word_idx..data.len() {
        let popcnt = data[i].count_ones() as u64;
        if n < popcnt {
            return Some(i * 64 + select_in_word_scalar(data[i], n as u32) as usize);
        }
        n -= popcnt;
    }

    None
}

/// Scalar select within a single word (used after NEON narrows to the right word)
#[inline]
fn select_in_word_scalar(mut word: u64, mut n: u32) -> u32 {
    for i in 0..64 {
        if word & 1 == 1 {
            if n == 0 {
                return i;
            }
            n -= 1;
        }
        word >>= 1;
    }
    64
}

// =====================================================================
// Fallback: on non-ARM, delegate to scalar implementations
// =====================================================================

#[cfg(not(target_arch = "aarch64"))]
pub fn popcount_block(data: &[u64]) -> u64 {
    super::scalar::popcount_block(data)
}

#[cfg(not(target_arch = "aarch64"))]
pub fn rank(data: &[u64], pos: usize) -> u64 {
    super::scalar::rank(data, pos)
}

#[cfg(not(target_arch = "aarch64"))]
pub fn select(data: &[u64], n: u64) -> Option<usize> {
    super::scalar::select(data, n)
}
