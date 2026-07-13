use std::fmt::{Display, Formatter, Result};
use std::ops::{Bound, RangeBounds};
use std::sync::{Arc, RwLock, RwLockReadGuard};

pub struct Tensor {
    data: Arc<RwLock<Vec<f32>>>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

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

    pub fn is_contiguous(&self) -> bool {
        self.offset == 0 && self.data().len() == self.shape.iter().product()
    }

    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        assert!(data.len() == shape.iter().product());

        let strides: Vec<usize> = Self::get_strides(&shape);

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
        let strides: Vec<usize> = Self::get_strides(&shape);

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

    pub fn iter<'a>(&'a self) -> Iter<'a> {
        Iter {
            tensor: &self,
            idx: 0,
            indices: vec![0; self.ndim()],
        }
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

    fn update_indices_from_index(indices: &mut [usize], idx: usize, shape: &[usize]) {
        let ndim: usize = shape.len();
        let mut remaining = idx;

        for j in (0..ndim).rev() {
            indices[j] = remaining % shape[j];
            remaining /= shape[j];
        }
    }

    fn reduce<F>(&self, init: f32, f: F) -> f32
    where
        F: Fn(f32, f32) -> f32,
    {
        let mut acc = init;

        for val in self.iter() {
            acc = f(acc, val);
        }

        acc
    }

    fn reduce_axis<F>(&self, axis: usize, init: f32, f: F) -> Option<Self>
    where
        F: Fn(f32, f32) -> f32,
    {
        let shape: Vec<usize> = self
            .shape
            .iter()
            .enumerate()
            .filter(|(i, _)| i != &axis)
            .map(|x| *x.1)
            .collect();

        let mut og_indices: Vec<usize> = vec![0; self.ndim()];
        let mut indices: Vec<usize> = vec![0; shape.len()];
        let mut data: Vec<f32> = vec![0.0; shape.iter().product()];

        for idx in 0..data.len() {
            Self::update_indices_from_index(indices.as_mut_slice(), idx, shape.as_slice());
            let mut acc = init;

            for i in 0..indices.len() {
                if i < axis {
                    og_indices[i] = indices[i]
                } else {
                    og_indices[i + 1] = indices[i]
                }
            }

            for i in 0..self.shape[axis] {
                og_indices[axis] = i;
                acc = f(acc, self.get_unchecked(og_indices.as_slice()));
            }

            data[idx] = acc;
        }

        Some(Self::new(data, shape))
    }

    pub fn sum(&self) -> f32 {
        self.reduce(0.0, |a, b| a + b)
    }

    pub fn sum_axis(&self, axis: usize) -> Option<Tensor> {
        self.reduce_axis(axis, 0.0, |a, b| a + b)
    }

    pub fn product(&self) -> f32 {
        self.reduce(1.0, |a, b| a * b)
    }

    pub fn product_axis(&self, axis: usize) -> Option<Tensor> {
        self.reduce_axis(axis, 1.0, |a, b| a * b)
    }

    pub fn mean(&self) -> f32 {
        self.sum() / self.len() as f32
    }

    pub fn mean_axis(&self, axis: usize) -> Option<Tensor> {
        if let Some(t) = self.sum_axis(axis) {
            Some(t.div_scalar(self.shape[axis] as f32))
        } else {
            None
        }
    }

    pub fn max(&self) -> f32 {
        self.reduce(f32::NEG_INFINITY, |a, b| a.max(b))
    }

    pub fn max_axis(&self, axis: usize) -> Option<Tensor> {
        self.reduce_axis(axis, f32::NEG_INFINITY, |a, b| a.max(b))
    }

    pub fn min(&self) -> f32 {
        self.reduce(f32::INFINITY, |a, b| a.min(b))
    }

    pub fn min_axis(&self, axis: usize) -> Option<Tensor> {
        self.reduce_axis(axis, f32::INFINITY, |a, b| a.min(b))
    }

    fn get_broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
        let broadcast_length: usize = a.len().max(b.len());
        let mut broadcast_shape: Vec<usize> = vec![0; broadcast_length];

        let mut a_rev = a.iter().rev();
        let mut b_rev = b.iter().rev();

        for i in (0..broadcast_length).rev() {
            let a_dim = a_rev.next().unwrap_or(&1);
            let b_dim = b_rev.next().unwrap_or(&1);

            let dim: usize;

            match (a_dim, b_dim) {
                (1, &d) => dim = d,
                (&d, 1) => dim = d,
                (&d1, &d2) if d1 == d2 => dim = d1,
                _ => panic!(
                    "[ERROR] Mismatch dimension ({} != {}) and neither equals to 1",
                    a_dim, b_dim
                ),
            }

            broadcast_shape[i] = dim;
        }

        broadcast_shape
    }

    fn update_reverse_broadcast(
        original_indices: &mut [usize],
        broadcast_indices: &[usize],
        original_shape: &[usize],
        broadcast_shape: &[usize],
    ) {
        let mut og_pointer = original_shape.len();
        let mut bc_pointer = broadcast_shape.len();

        assert!(og_pointer <= bc_pointer);

        for _ in (0..original_shape.len()).rev() {
            og_pointer -= 1;
            bc_pointer -= 1;

            let og_dim: usize = original_shape[og_pointer];
            let bc_dim: usize = broadcast_shape[bc_pointer];

            if og_dim == 1 {
                original_indices[og_pointer] = 0;
            } else if og_dim == bc_dim {
                original_indices[og_pointer] = broadcast_indices[bc_pointer];
            } else {
                panic!("Mismatch dimension at {} != {}", og_dim, bc_dim);
            }
        }
    }

    fn apply_binary<F>(&self, other: &Tensor, f: F) -> Self
    where
        F: Fn(f32, f32) -> f32,
    {
        let a_shape: &[usize] = self.shape.as_slice();
        let mut a_indices: Vec<usize> = vec![0; self.ndim()];

        let b_shape: &[usize] = other.shape.as_slice();
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

    fn apply_unary<F>(&self, f: F) -> Self
    where
        F: Fn(f32) -> f32,
    {
        let mut data = vec![0f32; self.len()];

        if self.is_contiguous() {
            let original_data = self.data();
            for idx in 0..self.len() {
                data[idx] = f(original_data[idx]);
            }
        } else {
            for (idx, val) in self.iter().enumerate() {
                data[idx] = f(val);
            }
        }

        Self::new(data, self.shape.clone())
    }

    pub fn relu(&self) -> Self {
        self.apply_unary(|x| x.max(0f32))
    }

    pub fn exp(&self) -> Self {
        self.apply_unary(|x| x.exp())
    }

    pub fn neg(&self) -> Self {
        self.apply_unary(|x| -x)
    }

    pub fn log(&self) -> Self {
        self.apply_unary(|x| x.ln())
    }

    pub fn sigmoid(&self) -> Self {
        self.apply_unary(|x| 1.0 / (1.0 + (-x).exp()))
    }

    pub fn tanh(&self) -> Self {
        self.apply_unary(|x| x.tanh())
    }

    pub fn add_scalar(&self, scalar: f32) -> Self {
        self.apply_unary(|x| x + scalar)
    }

    pub fn sub_scalar(&self, scalar: f32) -> Self {
        self.apply_unary(|x| x - scalar)
    }

    pub fn mult_scalar(&self, scalar: f32) -> Self {
        self.apply_unary(|x| x * scalar)
    }

    pub fn div_scalar(&self, scalar: f32) -> Self {
        self.apply_unary(|x| x / scalar)
    }

    fn get_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides: Vec<usize> = vec![1; shape.len()];

        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        strides
    }

    fn flat_index(&self, indices: &[usize]) -> Option<usize> {
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

    fn flat_index_unchecked(&self, indices: &[usize]) -> usize {
        assert!(self.shape.len() == indices.len());

        let mut flat_index: usize = self.offset;

        for i in 0..indices.len() {
            assert!(indices[i] < self.shape[i]);

            flat_index += self.strides[i] * indices[i];
        }

        flat_index
    }

    fn format(
        data: &[f32],
        shape: &[usize],
        strides: &[usize],
        offset: usize,
        max_dim: usize,
    ) -> String {
        assert!(max_dim >= strides.len());

        let mut formatted: String = String::new();

        if max_dim == strides.len() {
            formatted.push_str("Tensor");
        }

        if shape.len() == 1 {
            formatted.push('[');

            for i in 0..shape[0] {
                let idx = offset + strides[0] * i;

                if i > 0 {
                    formatted.push_str(", ");
                }

                formatted.push_str(&format!("{:.2}", data[idx]));
            }

            formatted.push(']');
        } else {
            formatted.push('[');

            for i in 0..shape[0] {
                let sub_offset = offset + strides[0] * i;

                formatted.push_str(&Self::format(
                    data,
                    &shape[1..shape.len()],
                    &strides[1..strides.len()],
                    sub_offset,
                    max_dim,
                ));

                if i < shape[0] - 1 {
                    formatted.push_str(&format!(
                        ",{}{}",
                        "\n".repeat((shape.len() - 1).max(1)),
                        " ".repeat(6 + max_dim - shape.len() + 1)
                    ));
                }
            }

            formatted.push(']');
        }

        formatted
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
                self.tensor.shape.as_slice(),
            );
            let value = self.tensor.get_unchecked(self.indices.as_slice());
            self.idx += 1;

            Some(value)
        } else {
            None
        }
    }
}

impl Display for Tensor {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let guard = self.data.read().unwrap();
        let data: &[f32] = guard.as_slice();

        write!(
            f,
            "{}",
            Self::format(
                data,
                &self.shape,
                &self.strides,
                self.offset,
                self.strides.len()
            )
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;

    // Broadcast Tests
    #[test]
    fn test_broadcast_same_shape() {
        let a: Vec<usize> = vec![4, 3];
        let b: Vec<usize> = vec![4, 3];

        let bc: Vec<usize> = Tensor::get_broadcast_shape(a.as_slice(), b.as_slice());
        assert_eq!(bc, vec![4, 3]);
    }

    #[test]
    fn test_broadcast_with_ones_and_different_ndim() {
        let a: Vec<usize> = vec![5, 1, 4];
        let b: Vec<usize> = vec![3, 1];

        let bc: Vec<usize> = Tensor::get_broadcast_shape(a.as_slice(), b.as_slice());
        assert_eq!(bc, vec![5, 3, 4]);
    }

    #[test]
    #[should_panic]
    fn test_broadcast_different_no_ones() {
        let a: Vec<usize> = vec![5, 6, 4];
        let b: Vec<usize> = vec![3, 3];

        Tensor::get_broadcast_shape(a.as_slice(), b.as_slice());
    }

    // Reverse Broadcast Tests
    #[test]
    fn test_reverse_broadcast_same_shape() {
        let mut original_indices: Vec<usize> = vec![0; 2];
        let broadcast_indices: Vec<usize> = vec![2, 2];

        let original_shape: Vec<usize> = vec![3, 1];
        let broadcast_shape: Vec<usize> = vec![3, 3];

        Tensor::update_reverse_broadcast(
            original_indices.as_mut_slice(),
            broadcast_indices.as_slice(),
            original_shape.as_slice(),
            broadcast_shape.as_slice(),
        );

        println!("original_indices: {:?}", original_indices);

        assert_eq!(original_indices, vec![2, 0]);
    }

    #[test]
    fn test_reverse_broadcast_different_shape() {
        let mut original_indices: Vec<usize> = vec![0; 2];
        let broadcast_indices: Vec<usize> = vec![2, 3, 4];

        let original_shape: Vec<usize> = vec![5, 1];
        let broadcast_shape: Vec<usize> = vec![3, 5, 5];

        Tensor::update_reverse_broadcast(
            original_indices.as_mut_slice(),
            broadcast_indices.as_slice(),
            original_shape.as_slice(),
            broadcast_shape.as_slice(),
        );

        println!("original_indices: {:?}", original_indices);

        assert_eq!(original_indices, vec![3, 0]);
    }
}
