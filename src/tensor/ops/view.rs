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

#[cfg(test)]
mod tests {
    use crate::Tensor;

    #[test]
    fn test_transpose_shape() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let r = t.transpose(0, 1).unwrap();
        assert_eq!(r.shape(), &[3, 2]);
        assert_eq!(r.strides(), &[1, 3]);
    }

    #[test]
    fn test_transpose_value() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let r = t.transpose(0, 1).unwrap();
        assert_eq!(r.get_unchecked(&[0, 0]), 1.0);
        assert_eq!(r.get_unchecked(&[1, 0]), 2.0);
        assert_eq!(r.get_unchecked(&[0, 1]), 3.0);
    }

    #[test]
    fn test_transpose_out_of_bounds() {
        let t = Tensor::ones(vec![2, 3]);
        assert!(t.transpose(0, 2).is_err());
        assert!(t.transpose(2, 0).is_err());
    }

    #[test]
    fn test_permute_shape() {
        let t = Tensor::new((0..24).map(|x| x as f32).collect(), vec![2, 3, 4]);
        let r = t.permute(&[2, 0, 1]).unwrap();
        assert_eq!(r.shape(), &[4, 2, 3]);
    }

    #[test]
    fn test_permute_duplicate_axes() {
        let t = Tensor::ones(vec![2, 3, 4]);
        assert!(t.permute(&[0, 0, 1]).is_err());
        assert!(t.permute(&[0, 1, 0]).is_err());
    }

    #[test]
    fn test_permute_axis_out_of_bounds() {
        let t = Tensor::ones(vec![2, 3]);
        assert!(t.permute(&[0, 2]).is_err());
    }

    #[test]
    fn test_reshape_contiguous() {
        let t = Tensor::new((0..6).map(|x| x as f32).collect(), vec![2, 3]);
        let r = t.reshape(&[6]).unwrap();
        assert_eq!(r.shape(), &[6]);
        assert_eq!(r.strides(), &[1]);
    }

    #[test]
    fn test_reshape_non_contiguous() {
        let t = Tensor::new((0..6).map(|x| x as f32).collect(), vec![2, 3]);
        let s = t.slice(&[0..2, 0..2]);
        let r = s.reshape(&[4]).unwrap();
        assert_eq!(r.shape(), &[4]);
        assert_eq!(r.iter().collect::<Vec<f32>>(), vec![0.0, 1.0, 3.0, 4.0]);
    }

    #[test]
    fn test_reshape_wrong_size() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert!(t.reshape(&[3]).is_err());
        assert!(t.reshape(&[2, 3]).is_err());
    }
}
