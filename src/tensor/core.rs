use std::sync::{Arc, RwLock};

pub struct Tensor {
    pub(crate) data: Arc<RwLock<Vec<f32>>>,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) offset: usize,
}
