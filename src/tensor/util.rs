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
