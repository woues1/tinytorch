use num_traits::Float;
use std::ops::{Add, Mul};
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

    pub fn sum_dim(&self, dim: usize) -> Self
    where
        T: num_traits::Float + std::ops::AddAssign,
    {
        assert!(dim < self.shape.len(), "Dimension out of bounds");

        let mut new_shape = self.shape.clone();
        new_shape[dim] = 1;

        let mut new_strides = vec![0; new_shape.len()];
        let mut current_stride = 1;
        for i in (0..new_shape.len()).rev() {
            new_strides[i] = current_stride;
            current_stride *= new_shape[i];
        }

        let total_elements = new_shape.iter().product();
        let mut new_data = vec![T::zero(); total_elements];

        for i in 0..self.data.len() {
            let mut current_val = i;
            let mut target_idx = 0;

            for d in (0..self.shape.len()).rev() {
                let coord = current_val % self.shape[d];
                current_val /= self.shape[d];

                if d != dim {
                    target_idx += coord * new_strides[d];
                }
            }

            new_data[target_idx] += self.data[i];
        }
        Tensor {
            data: new_data,
            shape: new_shape,
            strides: new_strides,
        }
    }

    pub fn mul_scalar(self, scalar: T) -> Self
    where
        T: Float,
    {
        let new_data = self.data.into_iter().map(|a| a * scalar).collect();

        Tensor {
            data: new_data,
            shape: self.shape,
            strides: self.strides,
        }
    }
}

// Arithmetic Operations
impl_elementwise_op!(Add, add, +);
impl_elementwise_op!(Sub, sub, -);
impl_elementwise_op!(Div, div, /);
impl_elementwise_op!(Mul, mul, *);
