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

#[cfg(test)]
mod tests {
    use crate::Tensor;

    #[test]
    fn test_add_same_shape() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let r = a.add(&b);
        let vals: Vec<f32> = r.iter().collect();
        assert_eq!(vals, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_sub_same_shape() {
        let a = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let r = a.sub(&b);
        let vals: Vec<f32> = r.iter().collect();
        assert_eq!(vals, vec![4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_mul_same_shape() {
        let a = Tensor::new(vec![2.0, 3.0], vec![2]);
        let b = Tensor::new(vec![4.0, 5.0], vec![2]);
        let r = a.mul(&b);
        let vals: Vec<f32> = r.iter().collect();
        assert_eq!(vals, vec![8.0, 15.0]);
    }

    #[test]
    fn test_div_same_shape() {
        let a = Tensor::new(vec![10.0, 20.0, 30.0], vec![3]);
        let b = Tensor::new(vec![2.0, 4.0, 5.0], vec![3]);
        let r = a.div(&b);
        let vals: Vec<f32> = r.iter().collect();
        assert_eq!(vals, vec![5.0, 5.0, 6.0]);
    }

    #[test]
    fn test_broadcast_scalar() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::new(vec![10.0], vec![1]);
        let r = a.add(&b);
        let vals: Vec<f32> = r.iter().collect();
        assert_eq!(vals, vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_broadcast_2d_scalar() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![10.0], vec![1]);
        let r = a.mul(&b);
        let vals: Vec<f32> = r.iter().collect();
        assert_eq!(vals, vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_broadcast_1d_to_2d() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![10.0, 20.0], vec![2]);
        let r = a.add(&b);
        assert_eq!(r.shape(), &[2, 2]);
        let vals: Vec<f32> = r.iter().collect();
        assert_eq!(vals, vec![11.0, 22.0, 13.0, 24.0]);
    }
}
