/// Cache-Adaptive B-Tree
///
/// A B-Tree whose node fanout is chosen at construction time based on the
/// detected ARM cache geometry. This is the "next layer up" from bitmaps:
/// where bitmaps handle filtering, B-Trees handle range lookups and sorted
/// access — the core indexing structure used by every database engine.
///
/// Why ARM-adaptive matters:
///   - Cortex-A55 (small core):  32KB L1D, 64-byte lines  → smaller nodes
///   - Cortex-A78 (big  core):  64KB L1D, 64-byte lines  → larger nodes
///   - Neoverse V1   (server):  64KB L1D, 64-byte lines  → larger nodes
/// Picking node size = N * cache_line_size keeps every key compare in L1
/// and lets ARM's prefetcher pipeline child loads.

/// Detected (or assumed) cache geometry of the running CPU.
///
/// Values are read from `/sys/devices/system/cpu/cpu0/cache/` on Linux
/// when available, and otherwise default to a modern ARM Cortex-A profile.
#[derive(Clone, Debug)]
pub struct CacheGeometry {
    pub line_size: usize,
    pub l1d_size: usize,
    pub l2_size: usize,
    pub source: &'static str,
}

impl CacheGeometry {
    /// Detect cache geometry of the current CPU.
    pub fn detect() -> Self {
        #[cfg(target_os = "linux")]
        {
            if let Some(g) = Self::from_sysfs() {
                return g;
            }
        }
        Self::default_modern_arm()
    }

    /// Default profile: modern ARM Cortex-A (A76/A78/Neoverse class).
    pub fn default_modern_arm() -> Self {
        CacheGeometry {
            line_size: 64,
            l1d_size: 64 * 1024,
            l2_size: 512 * 1024,
            source: "default-cortex-a78",
        }
    }

    /// Smaller-core profile, e.g. Cortex-A55 LITTLE cores in big.LITTLE.
    pub fn default_small_arm() -> Self {
        CacheGeometry {
            line_size: 64,
            l1d_size: 32 * 1024,
            l2_size: 256 * 1024,
            source: "default-cortex-a55",
        }
    }

    #[cfg(target_os = "linux")]
    fn from_sysfs() -> Option<Self> {
        let line_size = std::fs::read_to_string(
            "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size",
        )
        .ok()?
        .trim()
        .parse::<usize>()
        .ok()?;

        let l1d_size = parse_cache_size(
            &std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index0/size").ok()?,
        )?;

        // L2 may live at index1 or index2 depending on whether L1I is enumerated.
        let l2_size = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index2/size")
            .ok()
            .and_then(|s| parse_cache_size(&s))
            .or_else(|| {
                std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index1/size")
                    .ok()
                    .and_then(|s| parse_cache_size(&s))
            })
            .unwrap_or(512 * 1024);

        Some(CacheGeometry {
            line_size,
            l1d_size,
            l2_size,
            source: "sysfs",
        })
    }
}

#[cfg(target_os = "linux")]
fn parse_cache_size(s: &str) -> Option<usize> {
    let s = s.trim();
    if let Some(num) = s.strip_suffix('K') {
        num.parse::<usize>().ok().map(|v| v * 1024)
    } else if let Some(num) = s.strip_suffix('M') {
        num.parse::<usize>().ok().map(|v| v * 1024 * 1024)
    } else {
        s.parse::<usize>().ok()
    }
}

// ── Node ──────────────────────────────────────────────────────────

/// Bytes per (key, value) entry. We use i64 keys + u64 values.
const ENTRY_BYTES: usize = 16;
/// Minimum minimum-degree (CLRS `t`); any t >= 2 is valid.
const T_MIN: usize = 2;

struct Node {
    keys: Vec<i64>,
    values: Vec<u64>,
    children: Vec<Box<Node>>,
}

impl Node {
    fn new() -> Box<Self> {
        Box::new(Node {
            keys: Vec::new(),
            values: Vec::new(),
            children: Vec::new(),
        })
    }
    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
    fn n(&self) -> usize {
        self.keys.len()
    }
}

// ── Tree ──────────────────────────────────────────────────────────

/// A cache-adaptive B-Tree mapping `i64` keys to `u64` values.
///
/// Node fanout is chosen so that one node's key array fits in a small
/// number of cache lines on the detected ARM core.
pub struct CacheBTree {
    root: Box<Node>,
    /// CLRS minimum degree: every non-root node holds [t-1, 2t-1] keys.
    t: usize,
    len: usize,
    geometry: CacheGeometry,
    /// Number of cache lines we target per node.
    lines_per_node: usize,
}

impl CacheBTree {
    /// Create a B-Tree auto-tuned to the detected cache geometry.
    pub fn new() -> Self {
        Self::with_geometry(CacheGeometry::detect())
    }

    /// Create a B-Tree tuned to the given cache geometry.
    pub fn with_geometry(geometry: CacheGeometry) -> Self {
        // Pick how many cache lines a node spans based on L1D size:
        // small cores stay narrower so working sets fit in L1; bigger cores
        // can absorb wider nodes and shorter trees.
        let lines_per_node = if geometry.l1d_size <= 32 * 1024 {
            2
        } else if geometry.l1d_size <= 48 * 1024 {
            3
        } else {
            4
        };
        let target_bytes = lines_per_node * geometry.line_size;
        let max_keys = (target_bytes / ENTRY_BYTES).max(3);
        let t = (max_keys / 2).max(T_MIN);

        CacheBTree {
            root: Node::new(),
            t,
            len: 0,
            geometry,
            lines_per_node,
        }
    }

    /// Maximum number of keys per node (== 2t - 1 in CLRS notation).
    pub fn fanout(&self) -> usize {
        2 * self.t - 1
    }
    /// Minimum degree `t`.
    pub fn min_degree(&self) -> usize {
        self.t
    }
    /// Bytes occupied by a fully-packed node's key+value arrays.
    pub fn node_payload_bytes(&self) -> usize {
        self.fanout() * ENTRY_BYTES
    }
    /// Cache lines targeted per node.
    pub fn lines_per_node(&self) -> usize {
        self.lines_per_node
    }
    /// The cache geometry used to size this tree.
    pub fn geometry(&self) -> &CacheGeometry {
        &self.geometry
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Look up a key.
    pub fn get(&self, key: i64) -> Option<u64> {
        let mut node: &Node = &self.root;
        loop {
            match find_key(&node.keys, key) {
                Ok(i) => return Some(node.values[i]),
                Err(i) => {
                    if node.is_leaf() {
                        return None;
                    }
                    node = &node.children[i];
                }
            }
        }
    }

    /// Insert (key, value). Returns the previous value if `key` was already
    /// present, otherwise `None`.
    pub fn insert(&mut self, key: i64, value: u64) -> Option<u64> {
        let t = self.t;
        if self.root.n() == 2 * t - 1 {
            let old_root = std::mem::replace(&mut self.root, Node::new());
            self.root.children.push(old_root);
            Self::split_child(&mut self.root, 0, t);
        }
        let prev = Self::insert_nonfull(&mut self.root, key, value, t);
        if prev.is_none() {
            self.len += 1;
        }
        prev
    }

    /// Inclusive range scan: returns all (key, value) pairs with
    /// `low <= key <= high`, in sorted order.
    pub fn range(&self, low: i64, high: i64) -> Vec<(i64, u64)> {
        let mut out = Vec::new();
        Self::range_walk(&self.root, low, high, &mut out);
        out
    }

    fn range_walk(node: &Node, low: i64, high: i64, out: &mut Vec<(i64, u64)>) {
        if node.is_leaf() {
            for i in 0..node.n() {
                let k = node.keys[i];
                if k > high {
                    return;
                }
                if k >= low {
                    out.push((k, node.values[i]));
                }
            }
            return;
        }
        for i in 0..node.n() {
            let k = node.keys[i];
            if low <= k {
                Self::range_walk(&node.children[i], low, high, out);
            }
            if k > high {
                return;
            }
            if k >= low {
                out.push((k, node.values[i]));
            }
        }
        // Last child holds keys > node.keys[n-1].
        Self::range_walk(&node.children[node.n()], low, high, out);
    }

    fn split_child(parent: &mut Box<Node>, i: usize, t: usize) {
        // child has 2t-1 keys at indices 0..2t-1.
        // Split into:  left = [0..t-1]   median = [t-1]   right = [t..2t-1]
        let child = &mut parent.children[i];
        let mut new_child = Node::new();

        new_child.keys = child.keys.split_off(t);
        new_child.values = child.values.split_off(t);
        let median_key = child.keys.pop().expect("full node has median");
        let median_val = child.values.pop().expect("full node has median");
        if !child.is_leaf() {
            new_child.children = child.children.split_off(t);
        }

        parent.children.insert(i + 1, new_child);
        parent.keys.insert(i, median_key);
        parent.values.insert(i, median_val);
    }

    fn insert_nonfull(node: &mut Box<Node>, key: i64, value: u64, t: usize) -> Option<u64> {
        match find_key(&node.keys, key) {
            Ok(i) => Some(std::mem::replace(&mut node.values[i], value)),
            Err(mut i) => {
                if node.is_leaf() {
                    node.keys.insert(i, key);
                    node.values.insert(i, value);
                    None
                } else {
                    if node.children[i].n() == 2 * t - 1 {
                        Self::split_child(node, i, t);
                        // After split, parent has a new key at index i.
                        if key > node.keys[i] {
                            i += 1;
                        } else if key == node.keys[i] {
                            return Some(std::mem::replace(&mut node.values[i], value));
                        }
                    }
                    Self::insert_nonfull(&mut node.children[i], key, value, t)
                }
            }
        }
    }

    /// Crude estimate of bytes held by the tree (heap-only, ignoring vec
    /// capacity overhead). Useful for the bench harness.
    pub fn size_in_bytes(&self) -> usize {
        fn walk(node: &Node) -> usize {
            let mut s = std::mem::size_of::<Node>()
                + node.keys.len() * 8
                + node.values.len() * 8
                + node.children.len() * std::mem::size_of::<Box<Node>>();
            for c in &node.children {
                s += walk(c);
            }
            s
        }
        walk(&self.root)
    }
}

impl Default for CacheBTree {
    fn default() -> Self {
        Self::new()
    }
}

// ── Key search ────────────────────────────────────────────────────

/// Find `target` in a sorted `i64` slice.
///
/// Hybrid strategy:
///   - For small slices (typical B-Tree node), linear scan with NEON when
///     available — the whole node fits in 1-2 cache lines and the scan is
///     branch-predictor friendly.
///   - For large slices, fall back to `binary_search`.
#[inline]
pub fn find_key(keys: &[i64], target: i64) -> Result<usize, usize> {
    if keys.len() <= 32 {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            return find_key_neon(keys, target);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            return find_key_linear(keys, target);
        }
    }
    keys.binary_search(&target)
}

#[inline]
fn find_key_linear(keys: &[i64], target: i64) -> Result<usize, usize> {
    for (i, &k) in keys.iter().enumerate() {
        if k == target {
            return Ok(i);
        }
        if k > target {
            return Err(i);
        }
    }
    Err(keys.len())
}

/// NEON linear scan: 2 i64 lanes per iteration via `vceqq_s64` / `vcgtq_s64`.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn find_key_neon(keys: &[i64], target: i64) -> Result<usize, usize> {
    use core::arch::aarch64::*;
    let target_v = vdupq_n_s64(target);
    let mut i = 0;
    while i + 2 <= keys.len() {
        let v = vld1q_s64(keys.as_ptr().add(i));
        let eq = vceqq_s64(v, target_v);
        let gt = vcgtq_s64(v, target_v);

        let eq0 = vgetq_lane_u64(eq, 0);
        if eq0 != 0 {
            return Ok(i);
        }
        let gt0 = vgetq_lane_u64(gt, 0);
        if gt0 != 0 {
            return Err(i);
        }
        let eq1 = vgetq_lane_u64(eq, 1);
        if eq1 != 0 {
            return Ok(i + 1);
        }
        let gt1 = vgetq_lane_u64(gt, 1);
        if gt1 != 0 {
            return Err(i + 1);
        }
        i += 2;
    }
    while i < keys.len() {
        if keys[i] == target {
            return Ok(i);
        }
        if keys[i] > target {
            return Err(i);
        }
        i += 1;
    }
    Err(keys.len())
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_returns_sane_geometry() {
        let g = CacheGeometry::detect();
        assert!(g.line_size >= 32 && g.line_size <= 256);
        assert!(g.l1d_size >= 16 * 1024);
    }

    #[test]
    fn fanout_scales_with_line_size() {
        let small = CacheBTree::with_geometry(CacheGeometry::default_small_arm());
        let big = CacheBTree::with_geometry(CacheGeometry {
            line_size: 128,
            l1d_size: 64 * 1024,
            l2_size: 1024 * 1024,
            source: "synthetic-128B-lines",
        });
        assert!(big.fanout() >= small.fanout());
    }

    #[test]
    fn insert_get_basic() {
        let mut t = CacheBTree::new();
        for k in 0..1000i64 {
            t.insert(k, (k * 3) as u64);
        }
        for k in 0..1000i64 {
            assert_eq!(t.get(k), Some((k * 3) as u64));
        }
        assert_eq!(t.get(-1), None);
        assert_eq!(t.get(1000), None);
        assert_eq!(t.len(), 1000);
    }

    #[test]
    fn insert_replaces_existing_value() {
        let mut t = CacheBTree::new();
        assert_eq!(t.insert(7, 100), None);
        assert_eq!(t.insert(7, 200), Some(100));
        assert_eq!(t.get(7), Some(200));
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn insert_handles_descending_and_random_orders() {
        let mut t = CacheBTree::new();
        // Descending insertion stresses the right spine.
        for k in (0..500i64).rev() {
            t.insert(k, k as u64);
        }
        // Pseudo-random insertion stresses internal splits.
        let mut x: u64 = 0xC0FFEE;
        for _ in 0..500 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let k = 500 + (x as i64 & 0xFFFF);
            t.insert(k, x);
        }
        // Spot checks.
        for k in 0..500i64 {
            assert_eq!(t.get(k), Some(k as u64));
        }
    }

    #[test]
    fn range_returns_sorted_inclusive() {
        let mut t = CacheBTree::new();
        for k in 0..200i64 {
            t.insert(k * 2, k as u64); // even keys
        }
        let r = t.range(50, 100);
        // Even keys in [50, 100]: 50,52,...,100 → 26 entries
        assert_eq!(r.len(), 26);
        for w in r.windows(2) {
            assert!(w[0].0 < w[1].0);
        }
        assert_eq!(r.first().unwrap().0, 50);
        assert_eq!(r.last().unwrap().0, 100);
    }

    #[test]
    fn find_key_paths_agree() {
        let keys: Vec<i64> = (0..40i64).map(|i| i * 3).collect();
        for probe in -3..130i64 {
            let std_result = keys.binary_search(&probe);
            let our_result = find_key(&keys, probe);
            // Note: binary_search is allowed to return any matching index;
            // ours is too. For a strictly-sorted unique slice they must agree.
            assert_eq!(std_result, our_result, "probe={}", probe);
        }
    }
}
