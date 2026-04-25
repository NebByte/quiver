/// QuiverBitmap: A compressed bitmap designed for ARM.
///
/// This is a simplified Roaring-style bitmap with two container types:
/// - Array: sorted list of u16 values (sparse containers)
/// - Bitset: 1024-byte bitmap (dense containers)  
///
/// Key difference from Roaring: container size and transition thresholds
/// are designed for ARM Cortex-A cache characteristics:
/// - Bitset container = 1024 bytes = fits in half of a 32KB L1D on Cortex-A55
/// - Array→Bitset transition tuned for ARM memory access patterns
///
/// The bitmap partitions the u32 space into chunks of 8192 values (13 bits),
/// each stored in its own container. The high 19 bits select the chunk.

use crate::bitops;

/// Container size in bits (8192 = 1024 bytes as bitset)
/// Chosen to fit comfortably in ARM Cortex-A L1D cache.
const CONTAINER_BITS: usize = 8192;
const CONTAINER_WORDS: usize = CONTAINER_BITS / 64; // 128 u64 words
const CONTAINER_BYTES: usize = CONTAINER_BITS / 8;  // 1024 bytes

/// Threshold: if an array container has more than this many values,
/// convert to bitset. On ARM, the crossover point where bitset operations
/// beat sorted array scans is lower due to branch prediction differences.
const ARRAY_TO_BITSET_THRESHOLD: usize = 512;

#[derive(Clone, Debug)]
enum Container {
    /// Sorted array of u16 offsets within the chunk
    Array(Vec<u16>),
    /// Dense bitset: 128 u64 words = 8192 bits
    Bitset(Box<[u64; CONTAINER_WORDS]>),
}

#[derive(Clone, Debug)]
struct Chunk {
    key: u32,         // high 19 bits (chunk index)
    container: Container,
}

/// A compressed bitmap optimized for ARM cache hierarchy.
#[derive(Clone, Debug)]
pub struct QuiverBitmap {
    chunks: Vec<Chunk>,
}

impl QuiverBitmap {
    pub fn new() -> Self {
        QuiverBitmap { chunks: Vec::new() }
    }

    /// Insert a value into the bitmap.
    pub fn insert(&mut self, value: u32) {
        let chunk_key = value >> 13;       // high 19 bits
        let offset = (value & 0x1FFF) as u16; // low 13 bits

        // Find or create the chunk
        let chunk_idx = match self.chunks.binary_search_by_key(&chunk_key, |c| c.key) {
            Ok(idx) => idx,
            Err(idx) => {
                self.chunks.insert(idx, Chunk {
                    key: chunk_key,
                    container: Container::Array(Vec::new()),
                });
                idx
            }
        };

        let chunk = &mut self.chunks[chunk_idx];
        match &mut chunk.container {
            Container::Array(arr) => {
                if let Err(pos) = arr.binary_search(&offset) {
                    arr.insert(pos, offset);
                    // Check if we should convert to bitset
                    if arr.len() > ARRAY_TO_BITSET_THRESHOLD {
                        let mut bitset = Box::new([0u64; CONTAINER_WORDS]);
                        for &val in arr.iter() {
                            let word_idx = val as usize / 64;
                            let bit_idx = val as usize % 64;
                            bitset[word_idx] |= 1u64 << bit_idx;
                        }
                        chunk.container = Container::Bitset(bitset);
                    }
                }
            }
            Container::Bitset(bits) => {
                let word_idx = offset as usize / 64;
                let bit_idx = offset as usize % 64;
                bits[word_idx] |= 1u64 << bit_idx;
            }
        }
    }

    /// Check if a value is in the bitmap.
    pub fn contains(&self, value: u32) -> bool {
        let chunk_key = value >> 13;
        let offset = (value & 0x1FFF) as u16;

        match self.chunks.binary_search_by_key(&chunk_key, |c| c.key) {
            Ok(idx) => match &self.chunks[idx].container {
                Container::Array(arr) => arr.binary_search(&offset).is_ok(),
                Container::Bitset(bits) => {
                    let word_idx = offset as usize / 64;
                    let bit_idx = offset as usize % 64;
                    bits[word_idx] & (1u64 << bit_idx) != 0
                }
            },
            Err(_) => false,
        }
    }

    /// Total number of set bits — uses NEON-accelerated popcount on ARM.
    pub fn cardinality(&self) -> u64 {
        let mut total: u64 = 0;
        for chunk in &self.chunks {
            total += match &chunk.container {
                Container::Array(arr) => arr.len() as u64,
                Container::Bitset(bits) => bitops::popcount_block(bits.as_slice()),
            };
        }
        total
    }

    /// Rank: count set bits < value — uses NEON-accelerated rank on ARM.
    pub fn rank(&self, value: u32) -> u64 {
        let chunk_key = value >> 13;
        let offset = (value & 0x1FFF) as usize;
        let mut count: u64 = 0;

        for chunk in &self.chunks {
            if chunk.key < chunk_key {
                // Full chunk contributes entirely
                count += match &chunk.container {
                    Container::Array(arr) => arr.len() as u64,
                    Container::Bitset(bits) => bitops::popcount_block(bits.as_slice()),
                };
            } else if chunk.key == chunk_key {
                // Partial chunk — use rank
                count += match &chunk.container {
                    Container::Array(arr) => {
                        arr.partition_point(|&x| (x as usize) < offset) as u64
                    }
                    Container::Bitset(bits) => bitops::rank(bits.as_slice(), offset),
                };
                break;
            } else {
                break;
            }
        }
        count
    }

    /// Select: find the n-th set bit (0-indexed).
    pub fn select(&self, mut n: u64) -> Option<u32> {
        for chunk in &self.chunks {
            let chunk_card = match &chunk.container {
                Container::Array(arr) => arr.len() as u64,
                Container::Bitset(bits) => bitops::popcount_block(bits.as_slice()),
            };

            if n < chunk_card {
                // Target is in this chunk
                let offset = match &chunk.container {
                    Container::Array(arr) => arr[n as usize] as usize,
                    Container::Bitset(bits) => {
                        bitops::select(bits.as_slice(), n)?
                    }
                };
                return Some((chunk.key << 13) | offset as u32);
            }
            n -= chunk_card;
        }
        None
    }

    /// Intersection (AND) of two bitmaps — NEON-accelerated for bitset containers.
    pub fn and(&self, other: &QuiverBitmap) -> QuiverBitmap {
        let mut result = QuiverBitmap::new();
        let mut i = 0;
        let mut j = 0;

        while i < self.chunks.len() && j < other.chunks.len() {
            let a = &self.chunks[i];
            let b = &other.chunks[j];

            if a.key < b.key {
                i += 1;
            } else if a.key > b.key {
                j += 1;
            } else {
                // Same chunk key — intersect containers
                let container = match (&a.container, &b.container) {
                    (Container::Bitset(ba), Container::Bitset(bb)) => {
                        let mut out = Box::new([0u64; CONTAINER_WORDS]);
                        for k in 0..CONTAINER_WORDS {
                            out[k] = ba[k] & bb[k];
                        }
                        let card = bitops::popcount_block(out.as_slice());
                        if card == 0 {
                            i += 1;
                            j += 1;
                            continue;
                        }
                        Container::Bitset(out)
                    }
                    (Container::Array(aa), Container::Array(ab)) => {
                        // Sorted intersection
                        let mut out = Vec::new();
                        let mut ai = 0;
                        let mut bi = 0;
                        while ai < aa.len() && bi < ab.len() {
                            if aa[ai] < ab[bi] {
                                ai += 1;
                            } else if aa[ai] > ab[bi] {
                                bi += 1;
                            } else {
                                out.push(aa[ai]);
                                ai += 1;
                                bi += 1;
                            }
                        }
                        if out.is_empty() {
                            i += 1;
                            j += 1;
                            continue;
                        }
                        Container::Array(out)
                    }
                    // Mixed: iterate array, check bitset
                    (Container::Array(arr), Container::Bitset(bits))
                    | (Container::Bitset(bits), Container::Array(arr)) => {
                        let out: Vec<u16> = arr.iter()
                            .filter(|&&val| {
                                let wi = val as usize / 64;
                                let bi = val as usize % 64;
                                bits[wi] & (1u64 << bi) != 0
                            })
                            .copied()
                            .collect();
                        if out.is_empty() {
                            i += 1;
                            j += 1;
                            continue;
                        }
                        Container::Array(out)
                    }
                };

                result.chunks.push(Chunk { key: a.key, container });
                i += 1;
                j += 1;
            }
        }

        result
    }

    /// Union (OR) of two bitmaps.
    pub fn or(&self, other: &QuiverBitmap) -> QuiverBitmap {
        let mut result = QuiverBitmap::new();
        let mut i = 0;
        let mut j = 0;

        while i < self.chunks.len() || j < other.chunks.len() {
            let chunk = if i >= self.chunks.len() {
                j += 1;
                other.chunks[j - 1].clone()
            } else if j >= other.chunks.len() {
                i += 1;
                self.chunks[i - 1].clone()
            } else if self.chunks[i].key < other.chunks[j].key {
                i += 1;
                self.chunks[i - 1].clone()
            } else if self.chunks[i].key > other.chunks[j].key {
                j += 1;
                other.chunks[j - 1].clone()
            } else {
                // Same key — union containers
                let a = &self.chunks[i];
                let b = &other.chunks[j];
                i += 1;
                j += 1;

                let container = match (&a.container, &b.container) {
                    (Container::Bitset(ba), Container::Bitset(bb)) => {
                        let mut out = Box::new([0u64; CONTAINER_WORDS]);
                        for k in 0..CONTAINER_WORDS {
                            out[k] = ba[k] | bb[k];
                        }
                        Container::Bitset(out)
                    }
                    _ => {
                        // For simplicity in PoC: convert both to bitset and OR
                        let mut out = Box::new([0u64; CONTAINER_WORDS]);
                        Self::fill_bitset(&a.container, &mut out);
                        Self::fill_bitset(&b.container, &mut out);
                        Container::Bitset(out)
                    }
                };

                Chunk { key: a.key, container }
            };

            result.chunks.push(chunk);
        }

        result
    }

    fn fill_bitset(container: &Container, bitset: &mut [u64; CONTAINER_WORDS]) {
        match container {
            Container::Array(arr) => {
                for &val in arr {
                    let wi = val as usize / 64;
                    let bi = val as usize % 64;
                    bitset[wi] |= 1u64 << bi;
                }
            }
            Container::Bitset(bits) => {
                for k in 0..CONTAINER_WORDS {
                    bitset[k] |= bits[k];
                }
            }
        }
    }

    /// Memory usage in bytes.
    pub fn size_in_bytes(&self) -> usize {
        let mut size = std::mem::size_of::<Self>();
        for chunk in &self.chunks {
            size += std::mem::size_of::<Chunk>();
            size += match &chunk.container {
                Container::Array(arr) => arr.len() * 2, // u16 per element
                Container::Bitset(_) => CONTAINER_BYTES,
            };
        }
        size
    }

    /// Number of chunks (for diagnostics).
    pub fn num_chunks(&self) -> usize {
        self.chunks.len()
    }

    /// Iterate over all set bit positions.
    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.chunks.iter().flat_map(|chunk| {
            let base = chunk.key << 13;
            let values: Vec<u32> = match &chunk.container {
                Container::Array(arr) => {
                    arr.iter().map(|&v| base | v as u32).collect()
                }
                Container::Bitset(bits) => {
                    let mut out = Vec::new();
                    for (wi, &word) in bits.iter().enumerate() {
                        let mut w = word;
                        while w != 0 {
                            let bit = w.trailing_zeros();
                            out.push(base | (wi * 64 + bit as usize) as u32);
                            w &= w - 1; // clear lowest set bit
                        }
                    }
                    out
                }
            };
            values.into_iter()
        })
    }
}

impl Default for QuiverBitmap {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_contains() {
        let mut bm = QuiverBitmap::new();
        bm.insert(42);
        bm.insert(1000);
        bm.insert(100_000);

        assert!(bm.contains(42));
        assert!(bm.contains(1000));
        assert!(bm.contains(100_000));
        assert!(!bm.contains(43));
        assert!(!bm.contains(0));
    }

    #[test]
    fn test_cardinality() {
        let mut bm = QuiverBitmap::new();
        for i in 0..1000 {
            bm.insert(i * 7);
        }
        assert_eq!(bm.cardinality(), 1000);
    }

    #[test]
    fn test_rank() {
        let mut bm = QuiverBitmap::new();
        bm.insert(10);
        bm.insert(20);
        bm.insert(30);

        assert_eq!(bm.rank(0), 0);
        assert_eq!(bm.rank(10), 0);
        assert_eq!(bm.rank(11), 1);
        assert_eq!(bm.rank(21), 2);
        assert_eq!(bm.rank(31), 3);
    }

    #[test]
    fn test_select() {
        let mut bm = QuiverBitmap::new();
        bm.insert(10);
        bm.insert(20);
        bm.insert(30);

        assert_eq!(bm.select(0), Some(10));
        assert_eq!(bm.select(1), Some(20));
        assert_eq!(bm.select(2), Some(30));
        assert_eq!(bm.select(3), None);
    }

    #[test]
    fn test_intersection() {
        let mut a = QuiverBitmap::new();
        let mut b = QuiverBitmap::new();
        for i in 0..100 { a.insert(i); }
        for i in 50..150 { b.insert(i); }

        let c = a.and(&b);
        assert_eq!(c.cardinality(), 50); // 50..99
        assert!(c.contains(50));
        assert!(c.contains(99));
        assert!(!c.contains(49));
        assert!(!c.contains(100));
    }

    #[test]
    fn test_union() {
        let mut a = QuiverBitmap::new();
        let mut b = QuiverBitmap::new();
        for i in 0..100 { a.insert(i); }
        for i in 50..150 { b.insert(i); }

        let c = a.or(&b);
        assert_eq!(c.cardinality(), 150); // 0..149
    }

    #[test]
    fn test_dense_triggers_bitset() {
        let mut bm = QuiverBitmap::new();
        // Insert enough values in one chunk to trigger array→bitset conversion
        for i in 0..600 {
            bm.insert(i);
        }
        assert_eq!(bm.cardinality(), 600);
        // Verify all values present
        for i in 0..600 {
            assert!(bm.contains(i), "missing {}", i);
        }
    }

    #[test]
    fn test_iter() {
        let mut bm = QuiverBitmap::new();
        let values = vec![5, 100, 8000, 50000];
        for &v in &values {
            bm.insert(v);
        }
        let collected: Vec<u32> = bm.iter().collect();
        assert_eq!(collected, values);
    }
}
