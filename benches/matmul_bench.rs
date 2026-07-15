use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use yudanet::Tensor;

fn bench_matmul_naive(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_naive");
    group.sample_size(30);

    let sizes = [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (512, 1024, 256),
        (256, 1024, 512),
    ];

    for &(m, k, n) in &sizes {
        let a = Tensor::new((0..(m * k)).map(|i| i as f32).collect(), vec![m, k]);
        let b = Tensor::new((0..(k * n)).map(|i| i as f32).collect(), vec![k, n]);

        group.bench_function(&format!("matmul_naive_{}x{}x{}", m, k, n), |bencher| {
            bencher.iter(|| black_box(a.matmul(&b).unwrap()))
        });
    }
}

criterion_group!(benches, bench_matmul_naive);
criterion_main!(benches);
