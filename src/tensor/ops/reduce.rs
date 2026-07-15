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

#[cfg(test)]
mod tests {
    use crate::Tensor;

    #[test]
    fn test_sum_all() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(t.sum(), 10.0);
    }

    #[test]
    fn test_sum_axis_0() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let r = t.sum_axis(0).unwrap();
        let vals: Vec<f32> = r.iter().collect();
        assert_eq!(vals, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sum_axis_1() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let r = t.sum_axis(1).unwrap();
        let vals: Vec<f32> = r.iter().collect();
        assert_eq!(vals, vec![6.0, 15.0]);
    }

    #[test]
    fn test_product_all() {
        let t = Tensor::new(vec![2.0, 3.0, 4.0], vec![3]);
        assert_eq!(t.product(), 24.0);
    }

    #[test]
    fn test_product_axis() {
        let t = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]);
        let r = t.product_axis(1).unwrap();
        let vals: Vec<f32> = r.iter().collect();
        assert_eq!(vals, vec![6.0, 20.0]);
    }

    #[test]
    fn test_mean() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(t.mean(), 2.5);
    }

    #[test]
    fn test_mean_axis() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let r = t.mean_axis(0).unwrap();
        let vals: Vec<f32> = r.iter().collect();
        assert_eq!(vals, vec![2.0, 3.0]);
    }

    #[test]
    fn test_max() {
        let t = Tensor::new(vec![-5.0, 0.0, 3.0, 7.0], vec![4]);
        assert_eq!(t.max(), 7.0);
    }

    #[test]
    fn test_max_axis() {
        let t = Tensor::new(vec![1.0, 5.0, 3.0, 2.0], vec![2, 2]);
        let r = t.max_axis(0).unwrap();
        let vals: Vec<f32> = r.iter().collect();
        assert_eq!(vals, vec![3.0, 5.0]);
    }

    #[test]
    fn test_min() {
        let t = Tensor::new(vec![-5.0, 0.0, 3.0, 7.0], vec![4]);
        assert_eq!(t.min(), -5.0);
    }

    #[test]
    fn test_min_axis() {
        let t = Tensor::new(vec![1.0, 5.0, 3.0, 2.0], vec![2, 2]);
        let r = t.min_axis(1).unwrap();
        let vals: Vec<f32> = r.iter().collect();
        assert_eq!(vals, vec![1.0, 2.0]);
    }

    #[test]
    fn test_reduce_axis_out_of_bounds() {
        let t = Tensor::ones(vec![2, 3]);
        assert!(t.sum_axis(2).is_err());
        assert!(t.sum_axis(3).is_err());
    }
}
