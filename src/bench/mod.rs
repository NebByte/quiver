/// Minimal benchmarking harness — no external dependencies.
/// Measures wall-clock time with std::time::Instant.

use std::time::{Duration, Instant};

pub struct BenchResult {
    pub name: String,
    pub iterations: u64,
    pub total_time: Duration,
    pub mean_ns: f64,
    pub throughput: Option<f64>, // operations per second or bytes per second
}

impl BenchResult {
    pub fn print(&self) {
        print!(
            "  {:<35} {:>10.1} ns/op  ({} iters, {:.2} ms total)",
            self.name,
            self.mean_ns,
            self.iterations,
            self.total_time.as_secs_f64() * 1000.0,
        );
        if let Some(tp) = self.throughput {
            if tp > 1_000_000_000.0 {
                print!("  [{:.2} GB/s]", tp / 1_000_000_000.0);
            } else if tp > 1_000_000.0 {
                print!("  [{:.2} MB/s]", tp / 1_000_000.0);
            } else {
                print!("  [{:.2} Kops/s]", tp / 1_000.0);
            }
        }
        println!();
    }
}

/// Run a benchmark: executes `f` repeatedly for at least `min_duration`,
/// returns timing stats.
pub fn bench<F>(name: &str, min_duration: Duration, mut f: F) -> BenchResult
where
    F: FnMut() -> u64, // returns a value to prevent dead-code elimination
{
    // Warmup
    for _ in 0..10 {
        std::hint::black_box(f());
    }

    let mut iterations: u64 = 0;
    let start = Instant::now();

    while start.elapsed() < min_duration {
        std::hint::black_box(f());
        iterations += 1;
    }

    let total_time = start.elapsed();
    let mean_ns = total_time.as_nanos() as f64 / iterations as f64;

    BenchResult {
        name: name.to_string(),
        iterations,
        total_time,
        mean_ns,
        throughput: None,
    }
}

/// Same as bench but calculates throughput in bytes/sec.
pub fn bench_throughput<F>(
    name: &str,
    bytes_per_op: usize,
    min_duration: Duration,
    mut f: F,
) -> BenchResult
where
    F: FnMut() -> u64,
{
    let mut result = bench(name, min_duration, f);
    let ops_per_sec = 1_000_000_000.0 / result.mean_ns;
    result.throughput = Some(ops_per_sec * bytes_per_op as f64);
    result
}
