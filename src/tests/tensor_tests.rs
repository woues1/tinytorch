use crate::tensor::Tensor;
use std::ops::Add;
mod tests {
    use num_traits::Float;

    use super::*;

    #[test]
    fn test_strides() {
        let data: Vec<f64> = (1..=24).map(|x| x as f64).collect();
        let shape = vec![2, 3, 4];
        let res = Tensor::new(data, shape).unwrap();

        // The expected strides for a 2x3x4 tensor are [12, 4, 1]
        assert_eq!(res.strides, vec![12, 4, 1]);

        let data1: Vec<f64> = (1..=6).map(|x| x as f64).collect();
        let shape1 = vec![2, 3];
        let res1 = Tensor::new(data1, shape1).unwrap();
        assert_eq!(res1.strides, vec![3, 1]);

        // The expected strides for a shape of [3, 3, 4, 5] are [60, 20, 5, 1]
        let data2: Vec<f64> = (1..=180).map(|x| x as f64).collect();
        let shape2 = vec![3, 3, 4, 5];
        let res2 = Tensor::new(data2, shape2).unwrap();
        assert_eq!(res2.strides, vec![60, 20, 5, 1]);
    }
    #[test]
    fn test_add() {
        let data1: Vec<f64> = (1..=6).map(|x| x as f64).collect();
        let shape1 = vec![2, 3];
        let tensor1 = Tensor::new(data1, shape1).unwrap();

        let data2: Vec<f64> = (7..=12).map(|x| x as f64).collect();
        let shape2 = vec![2, 3];
        let tensor2 = Tensor::new(data2, shape2).unwrap();

        let result = tensor1 + tensor2;
        assert_eq!(result.shape, vec![2, 3]);
        assert_eq!(result.data, vec![8.0, 10.0, 12.0, 14.0, 16.0, 18.0]);
    }

    #[test]
    fn test_sub() {
        let data1: Vec<f64> = (1..=6).map(|x| x as f64).collect();
        let shape1 = vec![2, 3];
        let tensor1 = Tensor::new(data1, shape1).unwrap();

        let data2: Vec<f64> = (7..=12).map(|x| x as f64).collect();
        let shape2 = vec![2, 3];
        let tensor2 = Tensor::new(data2, shape2).unwrap();

        let result = tensor1 - tensor2;
        assert_eq!(result.shape, vec![2, 3]);
        assert_eq!(result.data, vec![-6.0, -6.0, -6.0, -6.0, -6.0, -6.0]);
    }

    #[test]
    fn test_matmul() {
        let data1: Vec<f64> = (1..=6).map(|x| x as f64).collect();
        let shape1 = vec![2, 3];
        let tensor1 = Tensor::new(data1, shape1).unwrap();

        let data2: Vec<f64> = (7..=12).map(|x| x as f64).collect();
        let shape2 = vec![3, 2];
        let tensor2 = Tensor::new(data2, shape2).unwrap();

        let result = tensor1.matmul(tensor2);
        assert_eq!(result.shape, vec![2, 2]);
        assert_eq!(result.data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_broadcast_1d_to_2d() {
        let data1: Vec<f64> = vec![1.0, 2.0];
        let shape1 = vec![2];
        let tensor1 = Tensor::new(data1, shape1).unwrap();

        let data2: Vec<f64> = (7..=12).map(|x| x as f64).collect();
        let shape2 = vec![3, 2];
        let tensor2 = Tensor::new(data2, shape2).unwrap();

        println!("{:?}", tensor1 + tensor2)
    }
    #[test]
    fn test_tensor_forward_pass_chain() {
        let a = Tensor {
            data: vec![1.0, -2.0],
            shape: vec![1, 2],
            strides: vec![2, 1],
        };

        let b = Tensor {
            data: vec![2.0, 1.0, -1.0, 3.0],
            shape: vec![2, 2],
            strides: vec![2, 1],
        };

        let bias = Tensor {
            data: vec![-1.0, 2.0],
            shape: vec![1, 2],
            strides: vec![2, 1],
        };

        let output = a.matmul(b).add(bias).relu();

        let expected_data = vec![3.0, 0.0];
        let expected_shape = vec![1, 2];
        let expected_strides = vec![2, 1];

        assert_eq!(output.shape, expected_shape, "Shape mismatch");
        assert_eq!(output.strides, expected_strides, "Stride mismatch");

        for (out_val, exp_val) in output.data.iter().zip(expected_data.iter()) {
            assert!(
                (out_val - exp_val).abs() < 1e-5,
                "Data mismatch: expected {:?}, got {:?}",
                expected_data,
                output.data
            );
        }
    }
    #[test]
    fn test_mean() {
        let a = Tensor {
            data: vec![1.0, -2.0],
            shape: vec![1, 2],
            strides: vec![2, 1],
        };
        assert_eq!(a.mean(), -0.5);
    }
    #[test]
    fn test_max_all() {
        let a = Tensor {
            data: vec![1.0, -2.0],
            shape: vec![1, 2],
            strides: vec![2, 1],
        };
        assert_eq!(a.max_all(), 1.0)
    }
    #[test]
    fn test_sum_all() {
        let a = Tensor {
            data: vec![1.0, -2.0],
            shape: vec![1, 2],
            strides: vec![2, 1],
        };
        assert_eq!(a.sum_all(), -1.0)
    }
    #[test]
    fn test_sum_dim() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let tensor = Tensor::new(data, shape).unwrap();

        println!("\n=== Original Tensor ===");
        println!("Shape: {:?}", tensor.shape);
        println!("Data:  {:?}\n", tensor.data);

        // 1. Test collapsing the rows (dim = 0)
        println!("=== Testing sum_dim(0) ===");
        let sum_rows = tensor.sum_dim(0);
        println!("Expected Shape: [1, 3]");
        println!("Actual Shape:   {:?}", sum_rows.shape);
        println!("Expected Data:  [5.0, 7.0, 9.0]");
        println!("Actual Data:    {:?}\n", sum_rows.data);

        assert_eq!(sum_rows.shape, vec![1, 3]);
        assert_eq!(sum_rows.data, vec![5.0, 7.0, 9.0]);

        // 2. Test collapsing the columns (dim = 1)
        println!("=== Testing sum_dim(1) ===");
        let sum_cols = tensor.sum_dim(1);
        println!("Expected Shape: [2, 1]");
        println!("Actual Shape:   {:?}", sum_cols.shape);
        println!("Expected Data:  [6.0, 15.0]");
        println!("Actual Data:    {:?}\n", sum_cols.data);

        assert_eq!(sum_cols.shape, vec![2, 1]);
        assert_eq!(sum_cols.data, vec![6.0, 15.0]);
    }
}
