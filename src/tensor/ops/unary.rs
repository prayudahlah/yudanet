use crate::Tensor;

impl Tensor {
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

        Self::new(data, self.shape().to_vec())
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
}
