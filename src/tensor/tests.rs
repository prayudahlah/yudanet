#[cfg(test)]
mod broadcast {
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
