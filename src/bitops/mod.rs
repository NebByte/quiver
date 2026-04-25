pub mod scalar;
pub mod neon;

/// Dispatch layer: selects optimal implementation at compile time.
/// On aarch64: NEON intrinsics
/// Everywhere else: scalar fallback
///
/// All public functions go through `neon` module, which internally
/// delegates to scalar on non-ARM targets via cfg.

/// Count total set bits in a block of u64 words.
#[inline]
pub fn popcount_block(data: &[u64]) -> u64 {
    neon::popcount_block(data)
}

/// Count set bits in positions [0, pos) across a block of u64 words.
#[inline]
pub fn rank(data: &[u64], pos: usize) -> u64 {
    neon::rank(data, pos)
}

/// Find position of the n-th set bit (0-indexed) in a block.
#[inline]
pub fn select(data: &[u64], n: u64) -> Option<usize> {
    neon::select(data, n)
}

/// Returns true if this build is using NEON intrinsics.
pub fn is_neon_active() -> bool {
    cfg!(target_arch = "aarch64")
}
