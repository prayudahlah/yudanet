use std::sync::{Arc, RwLock};

use crate::Tensor;

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        assert!(data.len() == shape.iter().product());

        let strides: Vec<usize> = Self::compute_strides(&shape);

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
        let strides: Vec<usize> = Self::compute_strides(&shape);

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
}
