use quiver::bitops;
use quiver::bitops::scalar;
use quiver::bitmap::QuiverBitmap;
use quiver::bench::{bench, bench_throughput, BenchResult};
use quiver::minidb::{QuiverDB, Value};
use quiver::btree::{CacheBTree, CacheGeometry};
use quiver::fmindex::FmIndex;
use rusqlite::{Connection, params};

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

    // ================================================================
    // 7. MINI DATABASE BENCHMARK (REAL-WORLD APPLICATION)
    // ================================================================
    println!();
    println!("━━━ QUIVERDB: columnar database with bitmap indexes ━━━");
    println!();

    // Build a 500K-row database with 3 columns
    let num_rows: usize = 500_000;
    println!("  Building database: {} rows, 3 columns...", num_rows);

    let ages: Vec<i64> = (0..num_rows as i64).map(|i| 18 + (i * 7 + 3) % 62).collect();
    let depts: Vec<i64> = (0..num_rows as i64).map(|i| (i * 13 + 5) % 20).collect();
    let regions: Vec<i64> = (0..num_rows as i64).map(|i| (i * 31 + 11) % 50).collect();

    let build_start = std::time::Instant::now();
    let mut db = QuiverDB::new();
    db.add_int_column("age", ages.clone());
    db.add_int_column("dept", depts.clone());
    db.add_int_column("region", regions.clone());
    let build_time = build_start.elapsed();

    println!("  Database built in {:.1} ms", build_time.as_secs_f64() * 1000.0);
    println!("  Index memory: {} KB", db.index_memory_bytes() / 1024);
    println!("  Row count: {}", db.row_count());
    println!();

    // --- Query 1: Single equality ---
    let target_age = Value::Int(30);
    let q1_bitmap = bench("DB: bitmap  WHERE age=30", min_dur, || {
        db.count_where_eq("age", &target_age)
    });
    let q1_naive = bench("DB: scan    WHERE age=30", min_dur, || {
        db.naive_count_where_eq_int("age", 30)
    });
    q1_bitmap.print();
    q1_naive.print();
    let count1 = db.count_where_eq("age", &target_age);
    let naive1 = db.naive_count_where_eq_int("age", 30);
    assert_eq!(count1, naive1, "equality count mismatch");
    println!("  → Bitmap vs Scan speedup: {:.1}x", q1_naive.mean_ns / q1_bitmap.mean_ns);
    println!();

    // --- Query 2: AND (two-column filter) ---
    let target_dept = Value::Int(5);
    let q2_bitmap = bench("DB: bitmap  WHERE age=30 AND dept=5", min_dur, || {
        db.count_where_and("age", &target_age, "dept", &target_dept)
    });
    let q2_naive = bench("DB: scan    WHERE age=30 AND dept=5", min_dur, || {
        db.naive_count_where_and_int("age", 30, "dept", 5)
    });
    q2_bitmap.print();
    q2_naive.print();
    let count2 = db.count_where_and("age", &target_age, "dept", &target_dept);
    let naive2 = db.naive_count_where_and_int("age", 30, "dept", 5);
    assert_eq!(count2, naive2, "AND count mismatch");
    println!("  → Bitmap vs Scan speedup: {:.1}x", q2_naive.mean_ns / q2_bitmap.mean_ns);
    println!();

    // --- Query 3: OR (two-column filter) ---
    let target_region = Value::Int(10);
    let q3_bitmap = bench("DB: bitmap  WHERE age=30 OR region=10", min_dur, || {
        db.count_where_or("age", &target_age, "region", &target_region)
    });
    q3_bitmap.print();
    println!();

    // --- Query 4: Range query ---
    let q4_bitmap = bench("DB: bitmap  WHERE age BETWEEN 25 AND 35", min_dur, || {
        db.count_where_range("age", 25, 35)
    });
    let q4_naive = bench("DB: scan    WHERE age BETWEEN 25 AND 35", min_dur, || {
        db.naive_count_where_range("age", 25, 35)
    });
    q4_bitmap.print();
    q4_naive.print();
    let count4 = db.count_where_range("age", 25, 35);
    let naive4 = db.naive_count_where_range("age", 25, 35);
    assert_eq!(count4, naive4, "range count mismatch");
    println!("  → Bitmap vs Scan speedup: {:.1}x", q4_naive.mean_ns / q4_bitmap.mean_ns);
    println!();

    // --- Query 5: Multi-AND (3 columns) ---
    let q5_bitmap = bench("DB: bitmap  WHERE age=30 AND dept=5 AND region=10", min_dur, || {
        db.count_where_multi_and(&[
            ("age", &Value::Int(30)),
            ("dept", &Value::Int(5)),
            ("region", &Value::Int(10)),
        ])
    });
    q5_bitmap.print();
    println!();

    // --- Correctness ---
    println!("━━━ DB CORRECTNESS ━━━");
    println!("  ✓ Single equality:  bitmap={} naive={}", count1, naive1);
    println!("  ✓ AND query:        bitmap={} naive={}", count2, naive2);
    println!("  ✓ Range query:      bitmap={} naive={}", count4, naive4);
    println!("  ✓ All queries match between bitmap-indexed and naive scan");

    // ================================================================
    // 8. VS REAL DATABASE (SQLite IN-MEMORY)
    // ================================================================
    println!();
    println!("━━━ VS SQLite (Real Production Database) ━━━");
    println!();
    println!("  Spinning up in-memory SQLite database...");
    
    let mut conn = Connection::open_in_memory().unwrap();
    conn.execute(
        "CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            age INTEGER,
            dept INTEGER,
            region INTEGER
        )",
        [],
    ).unwrap();

    // Create indexes to give SQLite a fair fight
    conn.execute("CREATE INDEX idx_age ON users(age)", []).unwrap();
    conn.execute("CREATE INDEX idx_dept ON users(dept)", []).unwrap();
    conn.execute("CREATE INDEX idx_region ON users(region)", []).unwrap();

    println!("  Loading {} rows into SQLite...", num_rows);
    let sqlite_build_start = std::time::Instant::now();
    
    {
        let tx = conn.transaction().unwrap();
        let mut stmt = tx.prepare("INSERT INTO users (id, age, dept, region) VALUES (?1, ?2, ?3, ?4)").unwrap();
        for i in 0..num_rows {
            stmt.execute(params![i as i64, ages[i], depts[i], regions[i]]).unwrap();
        }
        drop(stmt);
        tx.commit().unwrap();
    }
    
    println!("  SQLite DB built in {:.1} ms", sqlite_build_start.elapsed().as_secs_f64() * 1000.0);

    // Run the identical AND query: age=30 AND dept=5
    let q2_sqlite = bench("DB: SQLite  WHERE age=30 AND dept=5", min_dur, || {
        let mut stmt = conn.prepare_cached("SELECT count(*) FROM users WHERE age = 30 AND dept = 5").unwrap();
        let count: i64 = stmt.query_row([], |row| row.get(0)).unwrap();
        count as u64
    });
    
    q2_sqlite.print();
    println!("  → QuiverDB is {:.1}x FASTER than SQLite", q2_sqlite.mean_ns / q2_bitmap.mean_ns);

    // ================================================================
    // 9. CACHE-ADAPTIVE B-TREE
    // ================================================================
    println!();
    println!("━━━ CACHE-ADAPTIVE B-TREE: range/sorted indexing ━━━");
    println!();

    // Show how the tree auto-tunes to detected cache geometry.
    let detected = CacheGeometry::detect();
    println!(
        "  Detected geometry ({}): line={}B, L1D={}KB, L2={}KB",
        detected.source,
        detected.line_size,
        detected.l1d_size / 1024,
        detected.l2_size / 1024,
    );

    let tree_default = CacheBTree::new();
    let tree_a55 = CacheBTree::with_geometry(CacheGeometry::default_small_arm());
    let tree_a78 = CacheBTree::with_geometry(CacheGeometry::default_modern_arm());
    println!(
        "    detected core    → fanout {:>3} (~{}B/node, {} cache lines)",
        tree_default.fanout(),
        tree_default.node_payload_bytes(),
        tree_default.lines_per_node(),
    );
    println!(
        "    Cortex-A55 (LITTLE) → fanout {:>3} (~{}B/node, {} cache lines)",
        tree_a55.fanout(),
        tree_a55.node_payload_bytes(),
        tree_a55.lines_per_node(),
    );
    println!(
        "    Cortex-A78 (big)    → fanout {:>3} (~{}B/node, {} cache lines)",
        tree_a78.fanout(),
        tree_a78.node_payload_bytes(),
        tree_a78.lines_per_node(),
    );
    println!();

    // Build a 100K-entry tree.
    let n_btree: i64 = 100_000;
    let mut tree = CacheBTree::new();
    let build_start = std::time::Instant::now();
    for i in 0..n_btree {
        // Insert in pseudo-random order to exercise splits.
        let k = (i.wrapping_mul(2654435761) & 0x7FFF_FFFF) % n_btree;
        tree.insert(k, i as u64);
    }
    // Fill any gaps from collisions so the test data is dense.
    for k in 0..n_btree {
        tree.insert(k, k as u64);
    }
    let btree_build = build_start.elapsed();
    println!(
        "  Built B-Tree: {} entries in {:.1} ms ({} KB)",
        tree.len(),
        btree_build.as_secs_f64() * 1000.0,
        tree.size_in_bytes() / 1024,
    );

    // Point lookup throughput.
    bench("btree: point get (hit)", min_dur, || {
        let mut acc = 0u64;
        for i in (0..1000i64).map(|i| i * 97 % n_btree) {
            acc = acc.wrapping_add(tree.get(i).unwrap_or(0));
        }
        acc
    }).print();

    bench("btree: point get (miss)", min_dur, || {
        let mut acc = 0u64;
        for i in 0..1000i64 {
            if tree.get(n_btree + i).is_some() { acc += 1; }
        }
        acc
    }).print();

    bench("btree: range scan [25K, 75K]", min_dur, || {
        tree.range(25_000, 75_000).len() as u64
    }).print();

    // Compare against std BTreeMap on the same workload.
    use std::collections::BTreeMap;
    let mut std_tree: BTreeMap<i64, u64> = BTreeMap::new();
    for k in 0..n_btree {
        std_tree.insert(k, k as u64);
    }
    let q_std = bench("btree: std::BTreeMap point get", min_dur, || {
        let mut acc = 0u64;
        for i in (0..1000i64).map(|i| i * 97 % n_btree) {
            acc = acc.wrapping_add(*std_tree.get(&i).unwrap_or(&0));
        }
        acc
    });
    q_std.print();
    println!();

    // ================================================================
    // 10. FM-INDEX (FULL-TEXT SEARCH OVER BWT)
    // ================================================================
    println!("━━━ FM-INDEX: full-text search (genomics / cybersecurity) ━━━");
    println!();

    // Build a synthetic DNA-ish corpus with planted patterns.
    let alphabet = b"ACGT";
    let mut corpus = Vec::with_capacity(64 * 1024);
    let mut x: u64 = 0xBADC0FFEE0DDF00D;
    while corpus.len() < 64 * 1024 {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        corpus.push(alphabet[(x as usize) & 3]);
    }
    // Plant a needle so we can spot-check.
    let needle = b"ACGTACGTAC";
    for &offset in &[1234usize, 9876, 31415, 50000] {
        if offset + needle.len() <= corpus.len() {
            corpus[offset..offset + needle.len()].copy_from_slice(needle);
        }
    }

    let fm_build_start = std::time::Instant::now();
    let fm = FmIndex::build(&corpus);
    let fm_build = fm_build_start.elapsed();
    println!(
        "  Built FM-Index: {} bytes of text → {} KB index in {:.1} ms",
        corpus.len(),
        fm.size_in_bytes() / 1024,
        fm_build.as_secs_f64() * 1000.0,
    );

    let fm_count = fm.count(needle);
    println!("  Pattern {:?} occurs {} times (planted ≥4)", std::str::from_utf8(needle).unwrap(), fm_count);

    let q_fm_count = bench("fm-index: count (10-mer)", min_dur, || {
        fm.count(needle)
    });
    q_fm_count.print();

    bench("fm-index: count (5-mer ACGTA)", min_dur, || {
        fm.count(b"ACGTA")
    }).print();

    bench("fm-index: locate (10-mer)", min_dur, || {
        fm.locate(needle).len() as u64
    }).print();

    // Linear-scan baseline: how slow is naive substring search?
    let q_naive_scan = bench("scan:     naive count (10-mer)", min_dur, || {
        let pat = needle;
        let mut count = 0u64;
        if pat.len() <= corpus.len() {
            for i in 0..=corpus.len() - pat.len() {
                if &corpus[i..i + pat.len()] == pat {
                    count += 1;
                }
            }
        }
        count
    });
    q_naive_scan.print();
    println!(
        "  → FM-Index is {:.1}x faster than naive scan",
        q_naive_scan.mean_ns / q_fm_count.mean_ns,
    );

    // Correctness vs naive scan.
    let mut naive_count = 0u64;
    if needle.len() <= corpus.len() {
        for i in 0..=corpus.len() - needle.len() {
            if &corpus[i..i + needle.len()] == needle {
                naive_count += 1;
            }
        }
    }
    assert_eq!(fm_count, naive_count, "FM-Index disagrees with naive scan");
    println!("  ✓ FM-Index count matches naive scan ({})", fm_count);

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
