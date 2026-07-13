use std::collections::HashSet;

use crate::Tensor;

impl Tensor {
    pub fn transpose(&self, a: usize, b: usize) -> Result<Tensor, String> {
        if a >= self.ndim() {
            return Err("Axis a out of bounds".to_string());
        }

        if b >= self.ndim() {
            return Err("Axis b out of bounds".to_string());
        }

        let mut shape = self.shape().to_vec();
        shape[a] = self.shape()[b];
        shape[b] = self.shape()[a];

        let mut strides = self.strides().to_vec();
        strides[a] = self.strides()[b];
        strides[b] = self.strides()[a];

        Ok(Self {
            data: self.data.clone(),
            shape,
            strides,
            offset: self.offset(),
        })
    }

    pub fn permute(&self, axes: &[usize]) -> Result<Tensor, String> {
        if axes.len() != self.ndim() {
            return Err("Axes length must match tensor dimensions".to_string());
        }

        if axes.iter().any(|&x| x >= self.ndim()) {
            return Err("Axis out of bounds".to_string());
        }

        let mut seen = HashSet::new();
        if axes.iter().any(|&x| !seen.insert(x)) {
            return Err("Axes must be unique".to_string());
        }

        let mut shape = self.shape().to_vec();
        let mut strides = self.strides().to_vec();

        for i in 0..axes.len() {
            shape[i] = self.shape()[axes[i]];
            strides[i] = self.strides()[axes[i]];
        }

        Ok(Self {
            data: self.data.clone(),
            shape,
            strides,
            offset: self.offset(),
        })
    }

    pub fn reshape(&self, new_shape: &[usize]) -> Result<Tensor, String> {
        if new_shape.iter().product::<usize>() != self.len() {
            return Err("New shape must have the same number of elements".to_string());
        }

        if self.is_contiguous() {
            let shape = new_shape.to_vec();
            let strides = Self::compute_strides(new_shape);

            Ok(Self {
                data: self.data.clone(),
                shape,
                strides,
                offset: self.offset(),
            })
        } else {
            let data = self.iter().collect();

            Ok(Self::new(data, new_shape.to_vec()))
        }
    }
}
