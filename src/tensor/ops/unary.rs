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

#[cfg(test)]
mod tests {
    use crate::Tensor;

    #[test]
    fn test_relu_neg() {
        let t = Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
        let r = t.relu();
        let vals: Vec<f32> = r.iter().collect();
        assert_eq!(vals, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_relu_strided() {
        let t = Tensor::new(vec![-2.0, 1.0, -3.0, 4.0], vec![2, 2]);
        let s = t.slice(&[0..2, 0..1]);
        let r = s.relu();
        // slice of first column: [[-2], [-3]] → relu → [[0], [0]]
        assert_eq!(r.get_unchecked(&[0, 0]), 0.0);
        assert_eq!(r.get_unchecked(&[1, 0]), 0.0);
    }

    #[test]
    fn test_neg() {
        let t = Tensor::new(vec![1.0, -2.0, 3.0], vec![3]);
        let r = t.neg();
        let vals: Vec<f32> = r.iter().collect();
        assert_eq!(vals, vec![-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_exp() {
        let t = Tensor::new(vec![0.0, 1.0], vec![2]);
        let r = t.exp();
        assert!((r.get_unchecked(&[0]) - 1.0).abs() < 1e-5);
        assert!((r.get_unchecked(&[1]) - std::f32::consts::E).abs() < 1e-5);
    }

    #[test]
    fn test_log() {
        let t = Tensor::new(vec![1.0, std::f32::consts::E], vec![2]);
        let r = t.log();
        assert!((r.get_unchecked(&[0]) - 0.0).abs() < 1e-5);
        assert!((r.get_unchecked(&[1]) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_sigmoid() {
        let t = Tensor::new(vec![0.0, 100.0], vec![2]);
        let r = t.sigmoid();
        assert!((r.get_unchecked(&[0]) - 0.5).abs() < 1e-5);
        assert!((r.get_unchecked(&[1]) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_tanh() {
        let t = Tensor::new(vec![0.0], vec![1]);
        let r = t.tanh();
        assert!((r.get_unchecked(&[0]) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_add_scalar() {
        let t = Tensor::new(vec![1.0, 2.0], vec![2]);
        let r = t.add_scalar(10.0);
        assert_eq!(r.get_unchecked(&[0]), 11.0);
        assert_eq!(r.get_unchecked(&[1]), 12.0);
    }

    #[test]
    fn test_mult_scalar() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let r = t.mult_scalar(3.0);
        let vals: Vec<f32> = r.iter().collect();
        assert_eq!(vals, vec![3.0, 6.0, 9.0]);
    }

    #[test]
    fn test_div_scalar() {
        let t = Tensor::new(vec![2.0, 4.0, 6.0], vec![3]);
        let r = t.div_scalar(2.0);
        let vals: Vec<f32> = r.iter().collect();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }
}
