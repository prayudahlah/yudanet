use std::{
    ops::{Bound, RangeBounds},
    sync::RwLockReadGuard,
};

use crate::Tensor;

impl Tensor {
    pub fn data(&self) -> RwLockReadGuard<'_, Vec<f32>> {
        self.data.read().unwrap()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn get(&self, indices: &[usize]) -> Option<f32> {
        if let Some(idx) = self.flat_index(indices) {
            let guard = self.data.read().unwrap();
            Some(guard[idx])
        } else {
            None
        }
    }

    pub fn get_unchecked(&self, indices: &[usize]) -> f32 {
        let idx = self.flat_index_unchecked(indices);
        let guard = self.data.read().unwrap();
        guard[idx]
    }

    pub fn slice<R>(&self, ranges: &[R]) -> Tensor
    where
        R: RangeBounds<usize>,
    {
        assert!(ranges.len() == self.shape.len());

        let mut shape: Vec<usize> = Vec::with_capacity(self.shape.len());
        let mut offset: usize = 0;

        for (i, range) in ranges.iter().enumerate() {
            let mut start: usize = 0;
            let mut end: usize = self.shape[i];

            match range.start_bound() {
                Bound::Included(s) => start = *s,
                Bound::Excluded(s) => start = *s + 1,
                Bound::Unbounded => (),
            }

            match range.end_bound() {
                Bound::Included(e) => end = *e + 1,
                Bound::Excluded(e) => end = *e,
                Bound::Unbounded => (),
            }

            shape.push(end - start);
            offset += start * self.strides[i]
        }

        Self {
            data: self.data.clone(),
            shape,
            strides: self.strides.clone(),
            offset,
        }
    }
}
