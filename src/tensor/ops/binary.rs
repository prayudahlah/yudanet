use crate::Tensor;

impl Tensor {
    fn apply_binary<F>(&self, other: &Tensor, f: F) -> Self
    where
        F: Fn(f32, f32) -> f32,
    {
        let a_shape: &[usize] = self.shape();
        let mut a_indices: Vec<usize> = vec![0; self.ndim()];

        let b_shape: &[usize] = other.shape();
        let mut b_indices: Vec<usize> = vec![0; other.ndim()];

        let broadcast_shape: Vec<usize> = Self::get_broadcast_shape(a_shape, b_shape);

        let mut broadcast_data: Vec<f32> = vec![0.0; broadcast_shape.iter().product()];
        let mut broadcast_indices: Vec<usize> = vec![0; broadcast_shape.len()];

        for broadcast_idx in 0..broadcast_data.len() {
            Self::update_indices_from_index(
                broadcast_indices.as_mut_slice(),
                broadcast_idx,
                broadcast_shape.as_slice(),
            );

            Self::update_reverse_broadcast(
                a_indices.as_mut_slice(),
                broadcast_indices.as_slice(),
                a_shape,
                broadcast_shape.as_slice(),
            );

            Self::update_reverse_broadcast(
                b_indices.as_mut_slice(),
                broadcast_indices.as_slice(),
                b_shape,
                broadcast_shape.as_slice(),
            );

            broadcast_data[broadcast_idx] = f(
                self.get_unchecked(a_indices.as_slice()),
                other.get_unchecked(b_indices.as_slice()),
            );
        }

        Self::new(broadcast_data, broadcast_shape)
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        self.apply_binary(other, |a, b| a + b)
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        self.apply_binary(other, |a, b| a - b)
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        self.apply_binary(other, |a, b| a * b)
    }

    pub fn div(&self, other: &Tensor) -> Tensor {
        self.apply_binary(other, |a, b| a / b)
    }
}
