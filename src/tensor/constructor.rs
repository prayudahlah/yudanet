use std::sync::{Arc, RwLock};

use crate::Tensor;

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        assert!(data.len() == shape.iter().product());

        let strides: Vec<usize> = Self::compute_strides(&shape);

        Self {
            data: Arc::new(RwLock::new(data)),
            shape,
            strides,
            offset: 0,
        }
    }

    pub fn full(shape: Vec<usize>, fill_value: f32) -> Self {
        let size: usize = shape.iter().product();

        let data: Vec<f32> = vec![fill_value; size];
        let strides: Vec<usize> = Self::compute_strides(&shape);

        Self {
            data: Arc::new(RwLock::new(data)),
            shape,
            strides,
            offset: 0,
        }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        Self::full(shape, 0f32)
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        Self::full(shape, 1f32)
    }
}

#[cfg(test)]
mod tests {
    use crate::Tensor;

    #[test]
    fn test_new_valid() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.strides(), &[2, 1]);
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.len(), 4);
    }

    #[test]
    #[should_panic]
    fn test_new_shape_mismatch() {
        Tensor::new(vec![1.0, 2.0, 3.0], vec![2, 2]);
    }

    #[test]
    fn test_zeros() {
        let t = Tensor::zeros(vec![2, 3]);
        assert_eq!(t.len(), 6);
        assert!(t.iter().all(|x| x == 0.0));
    }

    #[test]
    fn test_ones() {
        let t = Tensor::ones(vec![1, 4]);
        assert_eq!(t.len(), 4);
        assert!(t.iter().all(|x| x == 1.0));
    }

    #[test]
    fn test_full() {
        let t = Tensor::full(vec![3], 5.5);
        assert_eq!(t.len(), 3);
        assert!(t.iter().all(|x| (x - 5.5).abs() < 1e-6));
    }
}
