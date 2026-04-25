/// Scalar (non-SIMD) implementations of core bit operations.
/// These serve as the baseline for benchmarking and as fallbacks on non-ARM targets.

/// Count set bits in a single u64
#[inline(always)]
pub fn popcount_u64(x: u64) -> u32 {
    x.count_ones()
}

/// Count set bits across a slice of u64 words.
/// This is the hot path — on ARM, NEON should crush this.
#[inline]
pub fn popcount_block(data: &[u64]) -> u64 {
    let mut count: u64 = 0;
    for &word in data {
        count += word.count_ones() as u64;
    }
    count
}

/// Rank: count set bits in positions [0, pos) across a block of u64 words.
/// pos is a bit index (0-based).
#[inline]
pub fn rank(data: &[u64], pos: usize) -> u64 {
    let word_idx = pos / 64;
    let bit_idx = pos % 64;
    let mut count: u64 = 0;

    // Count full words before the target word
    for i in 0..word_idx {
        if i < data.len() {
            count += data[i].count_ones() as u64;
        }
    }

    // Count bits within the target word up to (but not including) bit_idx
    if word_idx < data.len() && bit_idx > 0 {
        let mask = (1u64 << bit_idx) - 1;
        count += (data[word_idx] & mask).count_ones() as u64;
    }

    count
}

/// Select: find the position of the n-th set bit (0-indexed) in a block.
/// Returns None if there aren't enough set bits.
#[inline]
pub fn select(data: &[u64], mut n: u64) -> Option<usize> {
    for (word_idx, &word) in data.iter().enumerate() {
        let popcnt = word.count_ones() as u64;
        if n < popcnt {
            // The n-th bit is in this word — find it
            return Some(word_idx * 64 + select_in_word(word, n as u32) as usize);
        }
        n -= popcnt;
    }
    None
}

/// Find the position of the n-th set bit within a single u64 word.
/// Uses a broadword selection algorithm.
#[inline]
fn select_in_word(mut word: u64, mut n: u32) -> u32 {
    for i in 0..64 {
        if word & 1 == 1 {
            if n == 0 {
                return i;
            }
            n -= 1;
        }
        word >>= 1;
    }
    64 // shouldn't reach here if n < popcount
}

/// Broadword select — faster select_in_word using de Bruijn sequence.
/// Classic algorithm from Vigna's "Broadword Implementation of Rank/Select Queries"
#[inline]
pub fn select_in_word_broadword(x: u64, n: u32) -> u32 {
    // Parallel prefix sum to find byte containing the n-th bit
    let mut word = x;
    let mut rank = n;

    // Step 1: byte-level popcount via SWAR (SIMD Within A Register)
    let byte_counts = byte_popcount_swar(word);

    // Step 2: find which byte contains the n-th set bit
    let mut byte_idx = 0u32;
    let mut prefix_sum = 0u32;

    for b in 0..8 {
        let byte_count = ((byte_counts >> (b * 8)) & 0xFF) as u32;
        if prefix_sum + byte_count > rank {
            byte_idx = b;
            rank -= prefix_sum;
            break;
        }
        prefix_sum += byte_count;
    }

    // Step 3: linear scan within the target byte
    word >>= byte_idx * 8;
    for bit in 0..8 {
        if word & 1 == 1 {
            if rank == 0 {
                return byte_idx * 8 + bit;
            }
            rank -= 1;
        }
        word >>= 1;
    }

    64
}

/// SWAR byte-level popcount: returns a u64 where each byte contains the
/// popcount of the corresponding byte in the input.
#[inline(always)]
fn byte_popcount_swar(x: u64) -> u64 {
    let k1: u64 = 0x5555555555555555;
    let k2: u64 = 0x3333333333333333;
    let k4: u64 = 0x0f0f0f0f0f0f0f0f;

    let mut v = x;
    v = v - ((v >> 1) & k1);
    v = (v & k2) + ((v >> 2) & k2);
    v = (v + (v >> 4)) & k4;
    v // each byte now holds its popcount (0-8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_popcount_u64() {
        assert_eq!(popcount_u64(0), 0);
        assert_eq!(popcount_u64(u64::MAX), 64);
        assert_eq!(popcount_u64(0b1010_1010), 4);
        assert_eq!(popcount_u64(1), 1);
    }

    #[test]
    fn test_popcount_block() {
        let data = [0u64; 16];
        assert_eq!(popcount_block(&data), 0);

        let data = [u64::MAX; 16];
        assert_eq!(popcount_block(&data), 16 * 64);

        let data = [0b1111u64, 0b1111, 0b0000, 0b11];
        assert_eq!(popcount_block(&data), 10);
    }

    #[test]
    fn test_rank() {
        // Single word: 0b1011 = bits set at positions 0, 1, 3
        let data = [0b1011u64];
        assert_eq!(rank(&data, 0), 0); // no bits before pos 0
        assert_eq!(rank(&data, 1), 1); // bit 0 is set
        assert_eq!(rank(&data, 2), 2); // bits 0,1 are set
        assert_eq!(rank(&data, 3), 2); // bit 2 is NOT set
        assert_eq!(rank(&data, 4), 3); // bits 0,1,3 are set

        // Multi-word
        let data = [u64::MAX, 0b111u64];
        assert_eq!(rank(&data, 64), 64);
        assert_eq!(rank(&data, 66), 66);
        assert_eq!(rank(&data, 67), 67);
        assert_eq!(rank(&data, 68), 67);
    }

    #[test]
    fn test_select() {
        let data = [0b1011u64]; // bits at 0, 1, 3
        assert_eq!(select(&data, 0), Some(0));
        assert_eq!(select(&data, 1), Some(1));
        assert_eq!(select(&data, 2), Some(3));
        assert_eq!(select(&data, 3), None);

        // Multi-word
        let data = [u64::MAX, 0b101u64]; // 64 bits + bits at 64, 66
        assert_eq!(select(&data, 63), Some(63));
        assert_eq!(select(&data, 64), Some(64));
        assert_eq!(select(&data, 65), Some(66));
    }

    #[test]
    fn test_broadword_select() {
        assert_eq!(select_in_word_broadword(0b1011, 0), 0);
        assert_eq!(select_in_word_broadword(0b1011, 1), 1);
        assert_eq!(select_in_word_broadword(0b1011, 2), 3);
        assert_eq!(select_in_word_broadword(u64::MAX, 63), 63);
        assert_eq!(select_in_word_broadword(0x8000_0000_0000_0000, 0), 63);
    }
}
