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
