use core::arch::x86_64::*;
use rayon::prelude::*;

use crate::Tensor;

const DEFAULT_TILE_SIZE: usize = 32;

impl Tensor {
    pub fn matmul_add(&self, other: &Tensor, bias: &Tensor) -> Result<Tensor, String> {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { self.matmul_add_rayon_simd(other, DEFAULT_TILE_SIZE, bias) }
        } else {
            self.matmul_add_tiled(other, DEFAULT_TILE_SIZE, bias)
        }
    }

    pub fn matmul_add_tiled(
        &self,
        other: &Tensor,
        tile_size: usize,
        bias: &Tensor,
    ) -> Result<Tensor, String> {
        if self.ndim() != 2 || other.ndim() != 2 {
            return Err(
                "Both tensors must be 2-dimensional for matrix multiplication.".to_string(),
            );
        }

        if self.shape()[1] != other.shape()[0] {
            return Err("Inner dimensions must match for matrix multiplication.".to_string());
        }

        if bias.shape() != &[other.shape()[1]] {
            return Err(
                "Bias tensor must be 1-dimensional and match the number of columns in the result."
                    .to_string(),
            );
        }

        let matmul_shape: Vec<usize> = vec![self.shape()[0], other.shape()[1]];
        let mut matmul_data: Vec<f32> = vec![0.0; matmul_shape.iter().product()];

        let a_data = self.data();
        let b_data = other.data();
        let bias_data = bias.data();

        for ii in (0..matmul_shape[0]).step_by(tile_size) {
            let row_len = (ii + tile_size).min(matmul_shape[0]);

            for jj in (0..matmul_shape[1]).step_by(tile_size) {
                let col_len = (jj + tile_size).min(matmul_shape[1]);

                for kk in (0..self.shape()[1]).step_by(tile_size) {
                    let k_len = (kk + tile_size).min(self.shape()[1]);

                    for i in ii..row_len {
                        let matmul_idx = i * matmul_shape[1];

                        for k in kk..k_len {
                            for j in jj..col_len {
                                let a_idx =
                                    self.offset() + self.strides()[0] * i + self.strides()[1] * k;
                                let b_idx = other.offset()
                                    + other.strides()[0] * k
                                    + other.strides()[1] * j;

                                matmul_data[matmul_idx + j] += a_data[a_idx] * b_data[b_idx];
                            }
                        }
                    }
                }

                for i in ii..row_len {
                    let matmul_idx = i * matmul_shape[1];

                    for j in jj..col_len {
                        matmul_data[matmul_idx + j] += bias_data[j];
                    }
                }
            }
        }

        Ok(Self::new(matmul_data, matmul_shape))
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn matmul_add_rayon_simd(
        &self,
        other: &Tensor,
        tile_size: usize,
        bias: &Tensor,
    ) -> Result<Tensor, String> {
        if self.ndim() != 2 || other.ndim() != 2 {
            return Err(
                "Both tensors must be 2-dimensional for matrix multiplication.".to_string(),
            );
        }

        if self.shape()[1] != other.shape()[0] {
            return Err("Inner dimensions must match for matrix multiplication.".to_string());
        }

        if bias.shape() != &[other.shape()[1]] {
            return Err(
                "Bias tensor must be 1-dimensional and match the number of columns in the result."
                    .to_string(),
            );
        }

        let matmul_shape: Vec<usize> = vec![self.shape()[0], other.shape()[1]];
        let mut matmul_data: Vec<f32> = vec![0.0; matmul_shape.iter().product()];

        let a_data = self.data();
        let b_data = other.data();
        let bias_data = bias.data();

        let chunk_len = matmul_shape[1] * tile_size;

        matmul_data
            .par_chunks_mut(chunk_len)
            .enumerate()
            .for_each(|(tile_idx, chunk)| {
                let ii = tile_idx * tile_size;
                let row_len = (ii + tile_size).min(matmul_shape[0]);

                for jj in (0..matmul_shape[1]).step_by(tile_size) {
                    let col_len = (jj + tile_size).min(matmul_shape[1]);
                    let simd_bound: usize = col_len & !7;

                    for kk in (0..self.shape()[1]).step_by(tile_size) {
                        let k_len = (kk + tile_size).min(self.shape()[1]);

                        for i in ii..row_len {
                            let chunk_idx = (i - ii) * matmul_shape[1];

                            for k in kk..k_len {
                                for j in (jj..simd_bound).step_by(8) {
                                    let a_idx = self.offset()
                                        + self.strides()[0] * i
                                        + self.strides()[1] * k;
                                    let b_idx = other.offset()
                                        + other.strides()[0] * k
                                        + other.strides()[1] * j;

                                    let va = _mm256_set1_ps(a_data[a_idx]);
                                    let vb = _mm256_loadu_ps(&b_data[b_idx]);
                                    let mut vc = _mm256_loadu_ps(&chunk[chunk_idx + j]);
                                    vc = _mm256_fmadd_ps(va, vb, vc);
                                    _mm256_storeu_ps(&mut chunk[chunk_idx + j], vc);
                                }

                                for j in simd_bound..col_len {
                                    let a_idx = self.offset()
                                        + self.strides()[0] * i
                                        + self.strides()[1] * k;
                                    let b_idx = other.offset()
                                        + other.strides()[0] * k
                                        + other.strides()[1] * j;

                                    chunk[chunk_idx + j] += a_data[a_idx] * b_data[b_idx];
                                }
                            }
                        }
                    }

                    for i in ii..row_len {
                        let chunk_idx = (i - ii) * matmul_shape[1];

                        for j in (jj..simd_bound).step_by(8) {
                            let mut vc = _mm256_loadu_ps(&chunk[chunk_idx + j]);
                            let vbias = _mm256_loadu_ps(&bias_data[j]);
                            vc = _mm256_add_ps(vc, vbias);
                            _mm256_storeu_ps(&mut chunk[chunk_idx + j], vc);
                        }

                        for j in simd_bound..col_len {
                            chunk[chunk_idx + j] += bias_data[j];
                        }
                    }
                }
            });

        Ok(Self::new(matmul_data, matmul_shape))
    }
}

#[cfg(test)]
mod tests {
    use crate::Tensor;

    #[test]
    fn test_matmul_add_small() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
        let bias = Tensor::new(vec![1.0, 2.0], vec![2]);

        let c = a.matmul_add(&b, &bias).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        let vals: Vec<f32> = c.iter().collect();
        assert_eq!(vals, vec![59.0, 66.0, 140.0, 156.0]);
    }

    #[test]
    fn test_matmul_add_multi_tile() {
        let a = Tensor::ones(vec![64, 64]);
        let b = Tensor::ones(vec![64, 64]);
        let bias = Tensor::new((0..64).map(|i| i as f32).collect(), vec![64]);

        let c = a.matmul_add(&b, &bias).unwrap();
        for i in 0..64 {
            for j in 0..64 {
                assert_eq!(c.get_unchecked(&[i, j]), 64.0 + j as f32);
            }
        }
    }

    #[test]
    fn test_matmul_add_wrong_bias_shape() {
        let a = Tensor::ones(vec![2, 3]);
        let b = Tensor::ones(vec![3, 2]);
        let bias = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        assert!(a.matmul_add(&b, &bias).is_err());
    }
}
