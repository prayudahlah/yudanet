use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use yudanet::Tensor;

fn bench_matmul(c: &mut Criterion) {
    let sizes = [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (512, 1024, 256),
        (256, 1024, 512),
    ];

    for &(m, k, n) in &sizes {
        let mut group = c.benchmark_group(format!("matmul_{}x{}x{}", m, k, n));
        group.sample_size(30);

        let a = Tensor::new((0..(m * k)).map(|i| i as f32).collect(), vec![m, k]);
        let b = Tensor::new((0..(k * n)).map(|i| i as f32).collect(), vec![k, n]);

        group.bench_function("naive", |bencher| {
            bencher.iter(|| black_box(a.matmul_naive(&b).unwrap()))
        });
        group.bench_function("tiled32", |bencher| {
            bencher.iter(|| black_box(a.matmul_tiled(&b, 32).unwrap()))
        });
        unsafe {
            group.bench_function("simd", |bencher| {
                bencher.iter(|| black_box(a.matmul_simd(&b, 32).unwrap()))
            });
            group.bench_function("rayon_simd", |bencher| {
                bencher.iter(|| black_box(a.matmul_rayon_simd(&b, 32).unwrap()))
            });
        }
        group.finish();
    }
}

fn bench_tile_experiment(c: &mut Criterion) {
    let tile_sizes = [8, 16, 32, 48, 64, 96, 128];
    let sizes = [(128, 128, 128), (256, 256, 256), (512, 512, 512)];

    for &(m, k, n) in &sizes {
        let mut group = c.benchmark_group(format!("tile_exp_{}x{}x{}", m, k, n));
        group.sample_size(10);

        let a = Tensor::new((0..(m * k)).map(|i| i as f32).collect(), vec![m, k]);
        let b = Tensor::new((0..(k * n)).map(|i| i as f32).collect(), vec![k, n]);

        for &ts in &tile_sizes {
            group.bench_function(format!("tiled{}", ts), |bencher| {
                bencher.iter(|| black_box(a.matmul_tiled(&b, ts).unwrap()))
            });
        }
        group.finish();
    }
}

criterion_group!(benches, bench_matmul, bench_tile_experiment);
criterion_main!(benches);
