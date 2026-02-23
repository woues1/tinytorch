use std::ops::*;

#[derive(Debug, Clone)]
pub struct Tensor<T> {
    pub data: Vec<T>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl<T> Tensor<T> {
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Result<Self, String> {
        if data.len() != shape.iter().product() {
            return Err("Data length does not match shape".to_string());
        }

        let mut strides: Vec<usize> = Vec::with_capacity(shape.len());
        let mut current_stride = 1;

        for &dim in shape.iter().rev() {
            strides.push(current_stride);

            current_stride *= dim;
        }
        strides.reverse();

        Ok(Self {
            data,
            shape,
            strides,
        })
    }

    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn broadcast_shapes(&self, other: &Tensor<T>) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        let max_rank = self.shape.len().max(other.shape.len());

        let shape_a = self.shape.iter().rev().chain(std::iter::repeat(&1));
        let shape_b = other.shape.iter().rev().chain(std::iter::repeat(&1));
        let stride_a = self.strides.iter().rev().chain(std::iter::repeat(&0));
        let stride_b = other.strides.iter().rev().chain(std::iter::repeat(&0));

        let mut new_shape = Vec::with_capacity(max_rank);
        let mut new_strides_a = Vec::with_capacity(max_rank);
        let mut new_strides_b = Vec::with_capacity(max_rank);

        let iter = shape_a
            .zip(shape_b)
            .zip(stride_a)
            .zip(stride_b)
            .take(max_rank);

        for (((dim_a, dim_b), str_a), str_b) in iter {
            let (dim_a, dim_b, str_a, str_b) = (*dim_a, *dim_b, *str_a, *str_b);

            if dim_a == dim_b {
                new_shape.push(dim_a);
                new_strides_a.push(str_a);
                new_strides_b.push(str_b);
            } else if dim_b == 1 {
                new_shape.push(dim_a);
                new_strides_a.push(str_a);
                new_strides_b.push(0);
            } else if dim_a == 1 {
                new_shape.push(dim_b);
                new_strides_a.push(0);
                new_strides_b.push(str_b);
            } else {
                panic!("Shapes are not broadcastable: {} and {}", dim_a, dim_b);
            }
        }

        new_shape.reverse();
        new_strides_a.reverse();
        new_strides_b.reverse();

        (new_shape, new_strides_a, new_strides_b)
    }

    pub fn matmul(self, other: Tensor<T>) -> Tensor<T>
    where
        T: Default + Copy,
        T: Add<Output = T>,
        T: Mul<Output = T>,
    {
        let x = self.shape.split_at(self.shape.len() - 2);
        let y = other.shape.split_at(other.shape.len() - 2);
        assert_eq!(y.1[0], x.1[1]);

        let mut new_shape: Vec<usize> = Vec::new();
        new_shape.append(&mut x.0.to_vec());
        new_shape.push(x.1[0]);
        new_shape.push(y.1[1]);

        let mut res = Vec::new();
        let batch_count = x.0.iter().product();
        let matrix_size_a: usize = x.1.iter().product();
        let matrix_size_b: usize = y.1.iter().product();

        for b in 0..batch_count {
            let a_offset = b * matrix_size_a;
            let b_offset = b * matrix_size_b;
            for i in 0..x.1[0] {
                for j in 0..y.1[1] {
                    let mut sum = T::default();
                    for k in 0..x.1[1] {
                        let a_index = i * self.strides[self.strides.len() - 2]
                            + k * self.strides[self.strides.len() - 1]
                            + a_offset;

                        let b_index = k * other.strides[other.strides.len() - 2]
                            + j * other.strides[other.strides.len() - 1]
                            + b_offset;

                        sum = sum + self.data[a_index] * other.data[b_index];
                    }
                    res.push(sum);
                }
            }
        }
        Tensor::new(res, new_shape).unwrap()
    }
    pub fn reshape(mut self, new_shape: Vec<usize>) -> Self {
        assert_eq!(
            self.size(),
            new_shape.iter().product(),
            "Cannot reshape: total number of elements must remain the same."
        );
        let mut strides: Vec<usize> = Vec::with_capacity(new_shape.len());
        let mut current_stride = 1;

        for &dim in new_shape.iter().rev() {
            strides.push(current_stride);

            current_stride *= dim;
        }
        self.shape = new_shape;
        self.strides = strides;
        self
    }
    pub fn transpose(mut self) -> Self {
        let size = self.shape.len();
        if size >= 2 {
            self.shape.swap(size - 1, size - 2);
            self.strides.swap(size - 1, size - 2);
        }
        self
    }
}

// Arithmetic Operations
impl_elementwise_op!(Add, add, +);
impl_elementwise_op!(Sub, sub, -);
impl_elementwise_op!(Div, div, /);
impl_elementwise_op!(Mul, mul, *);

#[cfg(test)]
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
}
