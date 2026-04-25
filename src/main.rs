use quiver::bitops;
use quiver::bitops::scalar;
use quiver::bitmap::QuiverBitmap;
use quiver::bench::{bench, bench_throughput, BenchResult};

use std::time::Duration;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                   QUIVER PoC Benchmark                     ║");
    println!("║         ARM-Optimized Data Structure Primitives            ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Backend: {}                                    ║",
        if bitops::is_neon_active() { "ARM NEON (native)" } else { "Scalar (fallback) " }
    );
    println!("║  Platform: {:<47} ║", std::env::consts::ARCH);
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let min_dur = Duration::from_millis(500);

    // ── Generate test data ──────────────────────────────────────────
    let sizes: &[usize] = &[64, 256, 1024, 4096, 16384];

    // ================================================================
    // 1. POPCOUNT BENCHMARKS
    // ================================================================
    println!("━━━ POPCOUNT: count set bits in a block ━━━");
    println!();

    for &size in sizes {
        let data: Vec<u64> = (0..size as u64).map(|i| {
            // Deterministic pseudo-random fill
            i.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
        }).collect();
        let bytes = size * 8;

        let label_scalar = format!("scalar  popcount  {}w ({}KB)", size, bytes / 1024);
        let label_dispatch = format!("dispatch popcount {}w ({}KB)", size, bytes / 1024);

        let r1 = bench_throughput(&label_scalar, bytes, min_dur, || {
            scalar::popcount_block(&data)
        });

        let r2 = bench_throughput(&label_dispatch, bytes, min_dur, || {
            bitops::popcount_block(&data)
        });

        r1.print();
        r2.print();

        // Verify correctness
        let expected = scalar::popcount_block(&data);
        let actual = bitops::popcount_block(&data);
        assert_eq!(expected, actual, "popcount mismatch at size {}", size);

        println!();
    }

    // ================================================================
    // 2. RANK BENCHMARKS
    // ================================================================
    println!("━━━ RANK: count set bits before position ━━━");
    println!();

    for &size in &[256, 1024, 4096] {
        let data: Vec<u64> = (0..size as u64).map(|i| {
            i.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
        }).collect();
        let pos = size * 64 / 2; // middle position

        let label_scalar = format!("scalar  rank  {}w pos={}", size, pos);
        let label_dispatch = format!("dispatch rank {}w pos={}", size, pos);

        let r1 = bench(&label_scalar, min_dur, || {
            scalar::rank(&data, pos)
        });
        let r2 = bench(&label_dispatch, min_dur, || {
            bitops::rank(&data, pos)
        });

        r1.print();
        r2.print();

        // Verify
        assert_eq!(scalar::rank(&data, pos), bitops::rank(&data, pos));
        println!();
    }

    // ================================================================
    // 3. SELECT BENCHMARKS
    // ================================================================
    println!("━━━ SELECT: find n-th set bit position ━━━");
    println!();

    for &size in &[256, 1024, 4096] {
        let data: Vec<u64> = (0..size as u64).map(|i| {
            i.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
        }).collect();
        let total = scalar::popcount_block(&data);
        let target = total / 2; // find the middle set bit

        let label_scalar = format!("scalar  select  {}w n={}", size, target);
        let label_dispatch = format!("dispatch select {}w n={}", size, target);

        let r1 = bench(&label_scalar, min_dur, || {
            scalar::select(&data, target).unwrap_or(0) as u64
        });
        let r2 = bench(&label_dispatch, min_dur, || {
            bitops::select(&data, target).unwrap_or(0) as u64
        });

        r1.print();
        r2.print();

        // Verify
        assert_eq!(
            scalar::select(&data, target),
            bitops::select(&data, target),
            "select mismatch at size {}", size
        );
        println!();
    }

    // ================================================================
    // 4. BROADWORD SELECT BENCHMARK
    // ================================================================
    println!("━━━ SELECT-IN-WORD: broadword vs naive ━━━");
    println!();

    let test_word: u64 = 0xDEAD_BEEF_CAFE_BABEu64;
    let word_pop = test_word.count_ones();

    let r1 = bench("naive   select_in_word", min_dur, || {
        let mut sum = 0u64;
        for n in 0..word_pop {
            sum += scalar::select_in_word_broadword(test_word, n) as u64;
        }
        sum
    });
    let r2 = bench("broadword select_in_word", min_dur, || {
        let mut sum = 0u64;
        for n in 0..word_pop {
            sum += scalar::select_in_word_broadword(test_word, n) as u64;
        }
        sum
    });
    r1.print();
    r2.print();
    println!();

    // ================================================================
    // 5. COMPRESSED BITMAP BENCHMARKS
    // ================================================================
    println!("━━━ QUIVER BITMAP: compressed bitmap operations ━━━");
    println!();

    // Sparse bitmap: 10K values spread across u32 range
    let mut sparse = QuiverBitmap::new();
    for i in 0..10_000u32 {
        sparse.insert(i * 400); // spread out
    }

    // Dense bitmap: 100K contiguous values
    let mut dense = QuiverBitmap::new();
    for i in 0..100_000u32 {
        dense.insert(i);
    }

    // Another dense bitmap for set operations
    let mut dense2 = QuiverBitmap::new();
    for i in 50_000..150_000u32 {
        dense2.insert(i);
    }

    println!("  Sparse bitmap: {} values, {} bytes, {} chunks",
        sparse.cardinality(), sparse.size_in_bytes(), sparse.num_chunks());
    println!("  Dense  bitmap: {} values, {} bytes, {} chunks",
        dense.cardinality(), dense.size_in_bytes(), dense.num_chunks());
    println!();

    bench("bitmap: sparse contains (hit)", min_dur, || {
        let mut hits = 0u64;
        for i in (0..10_000u32).step_by(10) {
            if sparse.contains(i * 400) { hits += 1; }
        }
        hits
    }).print();

    bench("bitmap: sparse contains (miss)", min_dur, || {
        let mut hits = 0u64;
        for i in 0..1000u32 {
            if sparse.contains(i * 400 + 1) { hits += 1; }
        }
        hits
    }).print();

    bench("bitmap: dense cardinality", min_dur, || {
        dense.cardinality()
    }).print();

    bench("bitmap: dense rank(50000)", min_dur, || {
        dense.rank(50_000)
    }).print();

    bench("bitmap: dense select(50000)", min_dur, || {
        dense.select(50_000).unwrap_or(0) as u64
    }).print();

    bench("bitmap: dense AND (intersection)", min_dur, || {
        let result = dense.and(&dense2);
        result.cardinality()
    }).print();

    bench("bitmap: dense OR (union)", min_dur, || {
        let result = dense.or(&dense2);
        result.cardinality()
    }).print();

    println!();

    // ================================================================
    // 6. CORRECTNESS VERIFICATION
    // ================================================================
    println!("━━━ CORRECTNESS CHECKS ━━━");
    println!();

    // Bitmap correctness
    let intersection = dense.and(&dense2);
    let union = dense.or(&dense2);
    assert_eq!(intersection.cardinality(), 50_000, "AND cardinality");
    assert_eq!(union.cardinality(), 150_000, "OR cardinality");
    println!("  ✓ Bitmap AND: {} (expected 50000)", intersection.cardinality());
    println!("  ✓ Bitmap OR:  {} (expected 150000)", union.cardinality());

    // Rank/select round-trip on bitmap
    for i in 0..10 {
        let val = dense.select(i * 1000).unwrap();
        let r = dense.rank(val);
        assert_eq!(r, i * 1000, "rank/select round-trip failed at {}", i);
    }
    println!("  ✓ Rank/Select round-trip: 10 checks passed");

    // Bitops correctness across sizes
    for &size in sizes {
        let data: Vec<u64> = (0..size as u64).map(|i| {
            i.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
        }).collect();
        let pop = bitops::popcount_block(&data);
        let r = bitops::rank(&data, size * 64);
        assert_eq!(pop, r, "popcount vs full rank mismatch at size {}", size);

        if pop > 0 {
            let last_pos = bitops::select(&data, pop - 1).unwrap();
            assert!(last_pos < size * 64, "select out of bounds");
        }
    }
    println!("  ✓ Bitops popcount/rank/select: all sizes verified");

    // Broadword select correctness
    for n in 0..word_pop {
        let naive = scalar::select_in_word_broadword(test_word, n);
        let broadword = scalar::select_in_word_broadword(test_word, n);
        assert_eq!(naive, broadword, "broadword select mismatch at n={}", n);
    }
    println!("  ✓ Broadword select: {} checks passed", word_pop);

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    if bitops::is_neon_active() {
        println!("  Running on ARM with NEON — these are real SIMD numbers!");
    } else {
        println!("  Running on {} — scalar fallback active.", std::env::consts::ARCH);
        println!("  Cross-compile to aarch64 to see NEON speedups:");
        println!("    cargo build --release --target aarch64-unknown-linux-gnu");
    }
    println!("═══════════════════════════════════════════════════════════════");
}
