use yuda_learn::tensor::Tensor;

fn divider() {
    println!("\n{}", "-".repeat(30));
}

fn main() {
    divider();

    let a = Tensor::new(vec![-1., 2., -3., 4.], vec![2, 2]);
    let b = Tensor::new(vec![1., 2., -7., -1.], vec![2, 2]);

    println!("\n# Binary Dimensi sama");
    println!("{}\n+\n{}\n=\n{}", a, b, a.add(&b));

    let a = Tensor::new(vec![10.], vec![1]);
    let b = Tensor::new(vec![-1., 2., -3., 4.], vec![2, 2]);

    println!("\n# Binary Dimensi beda");
    println!("{}\n+\n{}\n=\n{}", a, b, a.add(&b));

    divider();

    let a = Tensor::new((1..=16).map(|x| x as f32).collect(), vec![4, 4]);
    let b = a.slice(vec![(1..=2), (1..=2)].as_slice());

    println!("\n# Slice:\n{}", b);
    println!("\n# Slice negated:\n{}", b.neg());

    divider();

    let a = Tensor::new((1..=6).map(|x| x as f32).collect(), vec![2, 3]);

    println!("\n# Sum of:\n{}\n=\n{}", a, a.sum());
    println!(
        "\n# Sum with axis 0 of:\n{}\n=\n{}",
        a,
        a.sum_axis(0).unwrap()
    );
    println!(
        "\n# Sum with axis 1 of:\n{}\n=\n{}",
        a,
        a.sum_axis(1).unwrap()
    );

    println!("\n# Mean of:\n{}\n=\n{}", a, a.mean());
    println!(
        "\n# Mean with axis 0 of:\n{}\n=\n{}",
        a,
        a.mean_axis(0).unwrap()
    );
    println!(
        "\n# Mean with axis 1 of:\n{}\n=\n{}",
        a,
        a.mean_axis(1).unwrap()
    );

    println!("\n# Max of:\n{}\n=\n{}", a, a.max());
    println!(
        "\n# Max with axis 0 of:\n{}\n=\n{}",
        a,
        a.max_axis(0).unwrap()
    );
    println!(
        "\n# Max with axis 1 of:\n{}\n=\n{}",
        a,
        a.max_axis(1).unwrap()
    );

    divider();
}
