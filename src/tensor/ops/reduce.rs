use crate::Tensor;

impl Tensor {
    fn apply_reduce<F>(&self, init: f32, f: F) -> f32
    where
        F: Fn(f32, f32) -> f32,
    {
        let mut acc = init;

        for val in self.iter() {
            acc = f(acc, val);
        }

        acc
    }

    fn apply_reduce_axis<F>(&self, axis: usize, init: f32, f: F) -> Result<Self, String>
    where
        F: Fn(f32, f32) -> f32,
    {
        if axis >= self.ndim() {
            return Err("Axis out of bounds".to_string());
        }

        let shape: Vec<usize> = self
            .shape()
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

            for i in 0..self.shape()[axis] {
                og_indices[axis] = i;
                acc = f(acc, self.get_unchecked(og_indices.as_slice()));
            }

            data[idx] = acc;
        }

        Ok(Self::new(data, shape))
    }

    pub fn sum(&self) -> f32 {
        self.apply_reduce(0.0, |a, b| a + b)
    }

    pub fn sum_axis(&self, axis: usize) -> Result<Tensor, String> {
        self.apply_reduce_axis(axis, 0.0, |a, b| a + b)
    }

    pub fn product(&self) -> f32 {
        self.apply_reduce(1.0, |a, b| a * b)
    }

    pub fn product_axis(&self, axis: usize) -> Result<Tensor, String> {
        self.apply_reduce_axis(axis, 1.0, |a, b| a * b)
    }

    pub fn mean(&self) -> f32 {
        self.sum() / self.len() as f32
    }

    pub fn mean_axis(&self, axis: usize) -> Result<Tensor, String> {
        let sum = self.sum_axis(axis)?;
        Ok(sum.div_scalar(self.shape()[axis] as f32))
    }

    pub fn max(&self) -> f32 {
        self.apply_reduce(f32::NEG_INFINITY, |a, b| a.max(b))
    }

    pub fn max_axis(&self, axis: usize) -> Result<Tensor, String> {
        self.apply_reduce_axis(axis, f32::NEG_INFINITY, |a, b| a.max(b))
    }

    pub fn min(&self) -> f32 {
        self.apply_reduce(f32::INFINITY, |a, b| a.min(b))
    }

    pub fn min_axis(&self, axis: usize) -> Result<Tensor, String> {
        self.apply_reduce_axis(axis, f32::INFINITY, |a, b| a.min(b))
    }
}
