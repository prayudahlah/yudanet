use crate::Tensor;

impl Tensor {
    pub fn is_contiguous(&self) -> bool {
        self.offset == 0 && self.data().len() == self.shape.iter().product()
    }

    pub fn to_contiguous(&self) -> Tensor {
        if self.is_contiguous() {
            Self {
                data: self.data.clone(),
                shape: self.shape.clone(),
                strides: self.strides.clone(),
                offset: self.offset,
            }
        } else {
            let data: Vec<f32> = self.iter().collect();

            Self::new(data, self.shape.clone())
        }
    }

    pub(crate) fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides: Vec<usize> = vec![1; shape.len()];

        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        strides
    }

    pub(crate) fn update_indices_from_index(indices: &mut [usize], idx: usize, shape: &[usize]) {
        let ndim: usize = shape.len();
        let mut remaining = idx;

        for j in (0..ndim).rev() {
            indices[j] = remaining % shape[j];
            remaining /= shape[j];
        }
    }

    pub(crate) fn flat_index(&self, indices: &[usize]) -> Option<usize> {
        if self.shape.len() != indices.len() {
            None
        } else {
            let mut flat_index: usize = self.offset;

            for i in 0..indices.len() {
                if indices[i] >= self.shape[i] {
                    return None;
                } else {
                    flat_index += self.strides[i] * indices[i];
                }
            }

            Some(flat_index)
        }
    }

    pub(crate) fn flat_index_unchecked(&self, indices: &[usize]) -> usize {
        assert!(self.shape.len() == indices.len());

        let mut flat_index: usize = self.offset;

        for i in 0..indices.len() {
            assert!(indices[i] < self.shape[i]);

            flat_index += self.strides[i] * indices[i];
        }

        flat_index
    }
}

#[cfg(test)]
mod tests {
    use crate::Tensor;

    #[test]
    fn test_compute_strides() {
        assert_eq!(Tensor::compute_strides(&[2, 3]), vec![3, 1]);
        assert_eq!(Tensor::compute_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(Tensor::compute_strides(&[5]), vec![1]);
    }

    #[test]
    fn test_is_contiguous() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert!(t.is_contiguous());
    }

    #[test]
    fn test_is_not_contiguous_after_slice() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let s = t.slice(&[0..2, 0..1]);
        assert!(!s.is_contiguous());
    }

    #[test]
    fn test_to_contiguous() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let s = t.slice(&[0..2, 0..1]);
        let c = s.to_contiguous();
        assert!(c.is_contiguous());
        assert_eq!(c.shape(), &[2, 1]);
        assert_eq!(c.strides(), &[1, 1]);
    }

    #[test]
    fn test_update_indices_from_index() {
        let mut indices = vec![0, 0, 0];
        Tensor::update_indices_from_index(&mut indices, 10, &[2, 3, 4]);
        // 10 = 0*12 + 2*4 + 2*1 → (0, 2, 2)
        assert_eq!(indices, vec![0, 2, 2]);

        let mut indices = vec![0, 0];
        Tensor::update_indices_from_index(&mut indices, 5, &[2, 3]);
        // 5 = 1*3 + 2*1 → (1, 2)
        assert_eq!(indices, vec![1, 2]);
    }

    #[test]
    fn test_flat_index() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(t.flat_index(&[0, 0]), Some(0));
        assert_eq!(t.flat_index(&[1, 0]), Some(2));
        assert_eq!(t.flat_index(&[1, 1]), Some(3));
    }

    #[test]
    fn test_flat_index_out_of_bounds() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(t.flat_index(&[2, 0]), None);
        assert_eq!(t.flat_index(&[0, 2]), None);
        assert_eq!(t.flat_index(&[0]), None);
        assert_eq!(t.flat_index(&[0, 0, 0]), None);
    }
}
