/// FM-Index: full-text substring search over a Burrows-Wheeler-transformed
/// text, built on Quiver's rank/select primitives.
///
/// Why this matters for Quiver:
///   - Genomics: aligning DNA reads against a reference genome is FM-index
///     backward search. BWA, Bowtie2, BWT-based aligners are all FM-index
///     under the hood.
///   - Cybersecurity: signature scanning, log search, malware string
///     detection all reduce to "is pattern P in text T?" with T potentially
///     gigabytes. FM-index occupies space close to the entropy of T.
///   - On-device search (mobile/edge): contacts, messages, documents — the
///     same primitive, but the index has to fit in RAM. ARM-native rank
///     queries are the bottleneck; Quiver already optimizes them.
///
/// Algorithm:
///   1. Append a sentinel `\0` (smaller than any other byte).
///   2. Build the suffix array (SA) of the sentinel-terminated text.
///   3. BWT[i] = T[SA[i] - 1]    (or T[n-1] when SA[i] == 0).
///   4. C[c] = number of bytes in T strictly less than `c`.
///   5. occ(c, i) = number of `c` occurrences in BWT[0..i]   ← rank query.
///   6. Backward search collapses pattern `P` of length m in m steps:
///        sp = 0; ep = n - 1
///        for c in P reversed:
///            sp = C[c] + occ(c, sp)
///            ep = C[c] + occ(c, ep + 1) - 1
///        if sp > ep → 0 occurrences, else ep - sp + 1.
///
/// `occ` is implemented as one bitmap per symbol plus `bitops::rank` —
/// reusing the NEON-accelerated rank from `quiver/src/bitops`.

use crate::bitops;

const ALPHABET: usize = 256;

/// FM-Index over an arbitrary byte text.
pub struct FmIndex {
    /// Length including the sentinel.
    n: usize,
    /// Suffix array (full, not sampled — keep PoC simple).
    sa: Vec<u32>,
    /// Burrows-Wheeler-transformed text.
    bwt: Vec<u8>,
    /// Cumulative count: C[c] = number of bytes in `text` strictly < c.
    c_array: [u32; ALPHABET + 1],
    /// One bitmap per symbol, packed as u64 words. `rank_bitmaps[c]` is
    /// `Some` iff symbol `c` appears in BWT.
    rank_bitmaps: Vec<Option<Vec<u64>>>,
}

impl FmIndex {
    /// Build the FM-Index over `text`. `text` must not contain `\0`.
    pub fn build(text: &[u8]) -> Self {
        assert!(!text.contains(&0u8), "FmIndex requires \\0-free input (sentinel)");

        // Step 1: append sentinel.
        let mut t = Vec::with_capacity(text.len() + 1);
        t.extend_from_slice(text);
        t.push(0);
        let n = t.len();

        // Step 2: suffix array via slice comparison sort.
        // O(n^2 log n) worst case — fine for the PoC; production would use
        // SA-IS or DC3.
        let mut sa: Vec<u32> = (0..n as u32).collect();
        sa.sort_by(|&a, &b| t[a as usize..].cmp(&t[b as usize..]));

        // Step 3: BWT.
        let mut bwt = Vec::with_capacity(n);
        for &i in &sa {
            let prev = if i == 0 { t[n - 1] } else { t[i as usize - 1] };
            bwt.push(prev);
        }

        // Step 4: C array. C[c] = total occurrences of bytes < c in text.
        let mut c_array = [0u32; ALPHABET + 1];
        for &b in &t {
            c_array[b as usize + 1] += 1;
        }
        for i in 1..=ALPHABET {
            c_array[i] += c_array[i - 1];
        }

        // Step 5: per-symbol rank bitmaps over BWT.
        let n_words = (n + 63) / 64;
        let mut rank_bitmaps: Vec<Option<Vec<u64>>> = (0..ALPHABET).map(|_| None).collect();
        for (i, &c) in bwt.iter().enumerate() {
            let bm = rank_bitmaps[c as usize].get_or_insert_with(|| vec![0u64; n_words]);
            bm[i / 64] |= 1u64 << (i % 64);
        }

        FmIndex {
            n,
            sa,
            bwt,
            c_array,
            rank_bitmaps,
        }
    }

    /// Number of occurrences of `pattern` in the original text.
    pub fn count(&self, pattern: &[u8]) -> u64 {
        match self.backward_search(pattern) {
            Some((sp, ep)) => (ep - sp + 1) as u64,
            None => 0,
        }
    }

    /// All starting positions of `pattern` in the original text.
    pub fn locate(&self, pattern: &[u8]) -> Vec<u32> {
        match self.backward_search(pattern) {
            Some((sp, ep)) => (sp..=ep).map(|i| self.sa[i]).collect(),
            None => Vec::new(),
        }
    }

    /// True if `pattern` appears anywhere in the text.
    pub fn contains(&self, pattern: &[u8]) -> bool {
        self.backward_search(pattern).is_some()
    }

    /// Length of the indexed text (excluding the sentinel).
    pub fn text_len(&self) -> usize {
        self.n - 1
    }

    /// Approximate memory footprint of the index (heap + sentinel).
    pub fn size_in_bytes(&self) -> usize {
        let bm_bytes: usize = self
            .rank_bitmaps
            .iter()
            .filter_map(|o| o.as_ref().map(|v| v.len() * 8))
            .sum();
        std::mem::size_of::<Self>()
            + self.sa.len() * 4
            + self.bwt.len()
            + bm_bytes
    }

    /// Run backward search for `pattern`; returns the [sp, ep] range over
    /// the suffix array, or `None` if there are no matches.
    fn backward_search(&self, pattern: &[u8]) -> Option<(usize, usize)> {
        if pattern.is_empty() {
            return None;
        }
        let mut sp: i64 = 0;
        let mut ep: i64 = (self.n - 1) as i64;
        for &c in pattern.iter().rev() {
            let cc = c as usize;
            let occ_sp = self.occ(cc, sp as usize);          // BWT[0..sp]
            let occ_ep1 = self.occ(cc, (ep + 1) as usize);   // BWT[0..ep+1]
            sp = self.c_array[cc] as i64 + occ_sp as i64;
            ep = self.c_array[cc] as i64 + occ_ep1 as i64 - 1;
            if sp > ep {
                return None;
            }
        }
        Some((sp as usize, ep as usize))
    }

    /// occ(c, i) = number of occurrences of byte `c` in BWT[0..i].
    /// Implemented via the per-symbol bitmap + NEON-accelerated rank.
    #[inline]
    fn occ(&self, c: usize, i: usize) -> u64 {
        if i == 0 {
            return 0;
        }
        match &self.rank_bitmaps[c] {
            Some(bm) => bitops::rank(bm, i),
            None => 0,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contains_simple() {
        let idx = FmIndex::build(b"banana");
        assert!(idx.contains(b"ana"));
        assert!(idx.contains(b"ban"));
        assert!(idx.contains(b"a"));
        assert!(!idx.contains(b"x"));
        assert!(!idx.contains(b"banx"));
    }

    #[test]
    fn count_matches_naive_scan() {
        let text = b"abracadabra abracadabra abracadabra";
        let idx = FmIndex::build(text);

        for pat in [&b"abra"[..], b"a", b"ra", b"d", b"zzz", b"adab"].iter() {
            let expected = naive_count(text, pat);
            assert_eq!(idx.count(pat), expected as u64, "pattern={:?}", pat);
        }
    }

    #[test]
    fn locate_matches_naive_scan() {
        let text = b"the quick brown fox jumps over the lazy dog the the the";
        let idx = FmIndex::build(text);

        for pat in [&b"the"[..], b"fox", b"lazy", b" "].iter() {
            let mut got = idx.locate(pat);
            got.sort();
            let expected = naive_positions(text, pat);
            assert_eq!(got, expected, "pattern={:?}", pat);
        }
    }

    #[test]
    fn locate_empty_pattern() {
        let idx = FmIndex::build(b"hello");
        assert!(idx.locate(b"").is_empty());
        assert_eq!(idx.count(b""), 0);
    }

    #[test]
    fn handles_dna_alphabet() {
        // FM-Index's headline use case: 4-letter alphabet, lots of repeats.
        let text = b"ACGTACGTACGTACGTACGTACGTACGT";
        let idx = FmIndex::build(text);
        let pat = b"CGTAC";
        assert_eq!(idx.count(pat), naive_count(text, pat) as u64);
        let mut found = idx.locate(pat);
        found.sort();
        assert_eq!(found, naive_positions(text, pat));
    }

    fn naive_count(text: &[u8], pat: &[u8]) -> usize {
        if pat.is_empty() || pat.len() > text.len() {
            return 0;
        }
        (0..=text.len() - pat.len())
            .filter(|&i| &text[i..i + pat.len()] == pat)
            .count()
    }

    fn naive_positions(text: &[u8], pat: &[u8]) -> Vec<u32> {
        if pat.is_empty() || pat.len() > text.len() {
            return vec![];
        }
        (0..=text.len() - pat.len())
            .filter(|&i| &text[i..i + pat.len()] == pat)
            .map(|i| i as u32)
            .collect()
    }
}
