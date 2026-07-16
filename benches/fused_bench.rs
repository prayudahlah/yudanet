use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use yudanet::Tensor;

fn fused_matmul(c: &mut Criterion) {
    let sizes = [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (512, 1024, 256),
        (256, 1024, 512),
    ];

    for &(m, k, n) in &sizes {
        let mut group = c.benchmark_group(format!("fused_add_relu{}x{}x{}", m, k, n));
        group.sample_size(30);

        let a = Tensor::new((0..(m * k)).map(|i| i as f32).collect(), vec![m, k]);
        let b = Tensor::new((0..(k * n)).map(|i| i as f32).collect(), vec![k, n]);
        let bias = Tensor::new((0..n).map(|i| i as f32).collect(), vec![n]);

        group.bench_function("independent", |bencher| {
            bencher.iter(|| black_box(a.matmul(&b).unwrap().add(&bias).relu()))
        });
        group.bench_function("fused", |bencher| {
            bencher.iter(|| black_box(a.matmul_add_relu(&b, &bias).unwrap()))
        });

        group.finish();
    }
}

criterion_group!(benches, fused_matmul);
criterion_main!(benches);
