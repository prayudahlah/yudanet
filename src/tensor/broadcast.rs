use crate::Tensor;

impl Tensor {
    pub(crate) fn get_broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
        let broadcast_length: usize = a.len().max(b.len());
        let mut broadcast_shape: Vec<usize> = vec![0; broadcast_length];

        let mut a_rev = a.iter().rev();
        let mut b_rev = b.iter().rev();

        for i in (0..broadcast_length).rev() {
            let a_dim = a_rev.next().unwrap_or(&1);
            let b_dim = b_rev.next().unwrap_or(&1);

            let dim: usize;

            match (a_dim, b_dim) {
                (1, &d) => dim = d,
                (&d, 1) => dim = d,
                (&d1, &d2) if d1 == d2 => dim = d1,
                _ => panic!(
                    "[ERROR] Mismatch dimension ({} != {}) and neither equals to 1",
                    a_dim, b_dim
                ),
            }

            broadcast_shape[i] = dim;
        }

        broadcast_shape
    }

    pub(crate) fn update_reverse_broadcast(
        original_indices: &mut [usize],
        broadcast_indices: &[usize],
        original_shape: &[usize],
        broadcast_shape: &[usize],
    ) {
        let mut og_pointer = original_shape.len();
        let mut bc_pointer = broadcast_shape.len();

        assert!(og_pointer <= bc_pointer);

        for _ in (0..original_shape.len()).rev() {
            og_pointer -= 1;
            bc_pointer -= 1;

            let og_dim: usize = original_shape[og_pointer];
            let bc_dim: usize = broadcast_shape[bc_pointer];

            if og_dim == 1 {
                original_indices[og_pointer] = 0;
            } else if og_dim == bc_dim {
                original_indices[og_pointer] = broadcast_indices[bc_pointer];
            } else {
                panic!("Mismatch dimension at {} != {}", og_dim, bc_dim);
            }
        }
    }
}
