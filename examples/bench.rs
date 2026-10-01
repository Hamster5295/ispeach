//! Benchmark the `ispeach` predicate.
//!
//! Usage: cargo run --release --example bench -- <img> [iters]
//!
//! Reports both:
//!   - `predicate(&DynamicImage)` : preprocessing + inference (image already decoded)
//!   - `predicate_path(path)`     : file read + decode + the above

use std::time::Instant;

use ispeach::{predicate, predicate_path};

fn stats(mut v: Vec<f64>) -> (f64, f64, f64, f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    let mean = v.iter().sum::<f64>() / n as f64;
    let p = |q: f64| v[((n as f64 - 1.0) * q).round() as usize];
    (v[0], p(0.5), p(0.9), v[n - 1], mean)
}

fn report(label: &str, times: Vec<f64>) {
    let (min, med, p90, max, mean) = stats(times);
    println!(
        "{label:<28} n mean={mean:.3}ms  min={min:.3}  median={med:.3}  p90={p90:.3}  max={max:.3}"
    );
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: bench <img> [iters]");
    let iters: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(50);

    // Warm-up (also triggers the one-time lazy model load/optimisation).
    let img = image::open(&path).expect("failed to open image");
    let t_load = Instant::now();
    let _ = predicate(&img).unwrap();
    println!(
        "first call (incl. model parse+optimise): {:.1}ms",
        t_load.elapsed().as_secs_f64() * 1e3
    );

    let mut a = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        let _ = predicate(&img).unwrap();
        a.push(t.elapsed().as_secs_f64() * 1e3);
    }
    report("predicate(&Image)", a);

    let mut b = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        let _ = predicate_path(&path).unwrap();
        b.push(t.elapsed().as_secs_f64() * 1e3);
    }
    report("predicate_path(path)", b);
}
