use crate::Tensor;

impl Tensor {
    pub fn iter<'a>(&'a self) -> Iter<'a> {
        Iter {
            tensor: &self,
            idx: 0,
            indices: vec![0; self.ndim()],
        }
    }
}

pub struct Iter<'a> {
    tensor: &'a Tensor,
    idx: usize,
    indices: Vec<usize>,
}

impl<'a> Iterator for Iter<'a> {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.tensor.len() {
            Tensor::update_indices_from_index(
                self.indices.as_mut_slice(),
                self.idx,
                self.tensor.shape(),
            );
            let value = self.tensor.get_unchecked(self.indices.as_slice());
            self.idx += 1;

            Some(value)
        } else {
            None
        }
    }
}
