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

#[cfg(test)]
mod tests {
    use crate::Tensor;

    #[test]
    fn test_iter_count() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let collected: Vec<f32> = t.iter().collect();
        assert_eq!(collected, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_iter_strided() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let s = t.slice(&[0..2, 0..1]);
        let collected: Vec<f32> = s.iter().collect();
        assert_eq!(collected, vec![1.0, 4.0]);
    }

    #[test]
    fn test_iter_empty_after_end() {
        let t = Tensor::ones(vec![3]);
        let mut iter = t.iter();
        assert!(iter.next().is_some());
        assert!(iter.next().is_some());
        assert!(iter.next().is_some());
        assert!(iter.next().is_none());
    }
}
