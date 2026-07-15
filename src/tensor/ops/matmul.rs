use crate::Tensor;

impl Tensor {
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, String> {
        if self.ndim() != 2 || other.ndim() != 2 {
            return Err(
                "Both tensors must be 2-dimensional for matrix multiplication.".to_string(),
            );
        }

        if self.shape()[1] != other.shape()[0] {
            return Err("Inner dimensions must match for matrix multiplication.".to_string());
        }

        let matmul_shape: Vec<usize> = vec![self.shape()[0], other.shape()[1]];
        let mut matmul_data: Vec<f32> = vec![0.0; matmul_shape.iter().product()];

        for i in 0..matmul_shape[0] {
            for j in 0..matmul_shape[1] {
                let matmul_idx = i * matmul_shape[1] + j;

                for k in 0..self.shape()[1] {
                    let a_idx = [i, k];
                    let b_idx = [k, j];

                    matmul_data[matmul_idx] +=
                        self.get_unchecked(&a_idx) * other.get_unchecked(&b_idx);
                }
            }
        }

        Ok(Self::new(matmul_data, matmul_shape))
    }
}

#[cfg(test)]
mod tests {
    use crate::Tensor;

    #[test]
    fn test_matmul_2x3_3x2() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        let vals: Vec<f32> = c.iter().collect();
        assert_eq!(vals, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_matmul_1x1() {
        let a = Tensor::new(vec![3.0], vec![1, 1]);
        let b = Tensor::new(vec![4.0], vec![1, 1]);
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.get_unchecked(&[0, 0]), 12.0);
    }

    #[test]
    fn test_matmul_not_2d() {
        let a = Tensor::ones(vec![2, 2, 2]);
        let b = Tensor::ones(vec![2, 2]);
        assert!(a.matmul(&b).is_err());
    }

    #[test]
    fn test_matmul_inner_dim_mismatch() {
        let a = Tensor::ones(vec![2, 3]);
        let b = Tensor::ones(vec![4, 5]);
        assert!(a.matmul(&b).is_err());
    }
}
