use std::sync::{Arc, RwLock};

pub trait BackwardOp<T>: Send + Sync {
    fn backward(&self, grad_output: &Tensor<T>);
}

pub struct TensorInner<T> {
    pub data: Vec<T>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub requires_grad: bool,
    pub grad: Option<Vec<T>>, // Stores the accumulated gradient
    pub grad_fn: Option<Arc<dyn BackwardOp<T>>>, // The operation that made this
}

pub trait TensorType: Copy + Default + Send + Sync + 'static {}

impl<T> TensorType for T where T: Copy + Default + Send + Sync + 'static {}

// 3. The Public Wrapper
#[derive(Clone)]
pub struct Tensor<T> {
    // Cloning this just clones the Arc pointer.
    pub inner: Arc<RwLock<TensorInner<T>>>,
}

impl<T> Tensor<T> {
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Result<Self, String> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err("Data length does not match shape".to_string());
        }

        let mut strides: Vec<usize> = Vec::with_capacity(shape.len());
        let mut current_stride = 1;

        for &dim in shape.iter().rev() {
            strides.push(current_stride);
            current_stride *= dim;
        }
        strides.reverse();

        // 1. Create the Inner struct
        let inner = TensorInner {
            data,
            shape,
            strides,
            requires_grad: false, // Default to false (only weights usually need true)
            grad: None,           // No gradient yet
            grad_fn: None,        // No parent operation yet
        };

        // 2. Wrap it in the thread-safe pointers
        Ok(Self {
            inner: Arc::new(RwLock::new(inner)),
        })
    }

    pub fn size(&self) -> usize {
        // THE READ PATTERN: Lock it, read it, drop the lock
        let inner = self.inner.read().unwrap();
        inner.shape.iter().product()
    }

    pub fn broadcast_shapes(&self, other: &Tensor<T>) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        // 1. Acquire read locks for both tensors
        let inner_self = self.inner.read().unwrap();
        let inner_other = other.inner.read().unwrap();

        // 2. Use the inner shapes and strides
        let max_rank = inner_self.shape.len().max(inner_other.shape.len());

        let shape_a = inner_self.shape.iter().rev().chain(std::iter::repeat(&1));

        let shape_b = inner_other.shape.iter().rev().chain(std::iter::repeat(&1));

        let stride_a = inner_self.strides.iter().rev().chain(std::iter::repeat(&0));

        let stride_b = inner_other
            .strides
            .iter()
            .rev()
            .chain(std::iter::repeat(&0));

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

        // 3. Locks are automatically dropped here at the end of the scope
        (new_shape, new_strides_a, new_strides_b)
    }

    pub fn matmul(&self, other: &Tensor<T>) -> Tensor<T>
    where
        T: Default + Copy,
        T: std::ops::Add<Output = T>,
        T: std::ops::Mul<Output = T>,
    {
        // 1. Lock both tensors for reading
        let inner_self = self.inner.read().unwrap();
        let inner_other = other.inner.read().unwrap();

        let x = inner_self.shape.split_at(inner_self.shape.len() - 2);
        let y = inner_other.shape.split_at(inner_other.shape.len() - 2);
        assert_eq!(
            y.1[0], x.1[1],
            "Inner matrix dimensions must match for matmul."
        );

        let mut new_shape: Vec<usize> = Vec::new();
        new_shape.append(&mut x.0.to_vec());
        new_shape.push(x.1[0]);
        new_shape.push(y.1[1]);

        let mut res = Vec::new();
        let batch_count = x.0.iter().product();
        let matrix_size_a: usize = x.1.iter().product();
        let matrix_size_b: usize = y.1.iter().product();

        // Extract lengths outside the loop for performance
        let self_strides_len = inner_self.strides.len();
        let other_strides_len = inner_other.strides.len();

        // 2. Compute the matrix multiplication using the inner data
        for b in 0..batch_count {
            let a_offset = b * matrix_size_a;
            let b_offset = b * matrix_size_b;
            for i in 0..x.1[0] {
                for j in 0..y.1[1] {
                    let mut sum = T::default();
                    for k in 0..x.1[1] {
                        let a_index = i * inner_self.strides[self_strides_len - 2]
                            + k * inner_self.strides[self_strides_len - 1]
                            + a_offset;

                        let b_index = k * inner_other.strides[other_strides_len - 2]
                            + j * inner_other.strides[other_strides_len - 1]
                            + b_offset;

                        sum = sum + inner_self.data[a_index] * inner_other.data[b_index];
                    }
                    res.push(sum);
                }
            }
        }

        // 3. Explicitly drop the locks before allocating the new tensor
        drop(inner_self);
        drop(inner_other);

        // 4. Return the new tensor
        Tensor::new(res, new_shape).unwrap()
    }

    pub fn reshape(self, new_shape: Vec<usize>) -> Self {
        let mut inner = self.inner.write().unwrap();

        let current_size: usize = inner.shape.iter().product();
        let new_size: usize = new_shape.iter().product();
        assert_eq!(
            current_size, new_size,
            "Cannot reshape: element count mismatch."
        );

        let mut strides: Vec<usize> = Vec::with_capacity(new_shape.len());
        let mut current_stride = 1;

        for &dim in new_shape.iter().rev() {
            strides.push(current_stride);
            current_stride *= dim;
        }

        inner.shape = new_shape;
        inner.strides = strides;

        drop(inner);
        self
    }

    pub fn transpose(&self) -> Self
    where
        T: Clone,
    {
        // 1. Acquire the read lock
        let inner = self.inner.read().unwrap();

        let mut new_shape = inner.shape.clone();
        let mut new_strides = inner.strides.clone();
        let size = new_shape.len();

        // 2. Swap the last two dimensions
        if size >= 2 {
            new_shape.swap(size - 1, size - 2);
            new_strides.swap(size - 1, size - 2);
        }

        // 3. Clone the data so the new tensor has its own memory
        let new_data = inner.data.clone();

        // 4. Drop the lock early
        drop(inner);

        // 5. Construct the wrapper manually to preserve the swapped strides!
        Self {
            inner: Arc::new(RwLock::new(TensorInner {
                data: new_data,
                shape: new_shape,
                strides: new_strides,
                requires_grad: false, // Default for new operations
                grad: None,
                grad_fn: None, // We will inject TransposeBackward here later!
            })),
        }
    }

    pub fn sum_dim(&self, dim: usize) -> Self
    where
        T: num_traits::Float + std::ops::AddAssign,
    {
        // 1. Acquire the read lock
        let inner = self.inner.read().unwrap();

        assert!(dim < inner.shape.len(), "Dimension out of bounds");

        let mut new_shape = inner.shape.clone();
        new_shape[dim] = 1;

        // We still need new_strides locally to calculate the target_idx
        let mut new_strides = vec![0; new_shape.len()];
        let mut current_stride = 1;
        for i in (0..new_shape.len()).rev() {
            new_strides[i] = current_stride;
            current_stride *= new_shape[i];
        }

        let total_elements: usize = new_shape.iter().product();
        let mut new_data = vec![T::zero(); total_elements];

        // 2. Read from inner.data and inner.shape
        for i in 0..inner.data.len() {
            let mut current_val = i;
            let mut target_idx = 0;

            for d in (0..inner.shape.len()).rev() {
                let coord = current_val % inner.shape[d];
                current_val /= inner.shape[d];

                if d != dim {
                    target_idx += coord * new_strides[d];
                }
            }

            new_data[target_idx] += inner.data[i];
        }

        // 3. Drop the lock explicitly before allocating the new wrapper
        drop(inner);

        // 4. Use the constructor to ensure Arc<RwLock> and Autograd fields are set up
        Tensor::new(new_data, new_shape).unwrap()
    }

    pub fn mul_scalar(&self, scalar: T) -> Self
    where
        T: num_traits::Float,
    {
        // 1. Acquire the read lock
        let inner = self.inner.read().unwrap();

        // 2. Iterate over references to avoid consuming the original data
        let new_data: Vec<T> = inner.data.iter().map(|&a| a * scalar).collect();

        // 3. Clone the shape to pass to the constructor
        let new_shape = inner.shape.clone();

        // 4. Drop the lock early (good concurrency practice)
        drop(inner);

        // 5. Build the new tensor wrapper with the correct Autograd initialization
        Tensor::new(new_data, new_shape).unwrap()
    }

    pub fn sub_scalar(&self, scalar: T) -> Self
    where
        T: num_traits::Float, // Kept your Float bound
    {
        // 1. Acquire the read lock
        let inner = self.inner.read().unwrap();

        // 2. Iterate over references instead of consuming the vector
        let new_data: Vec<T> = inner.data.iter().map(|&a| scalar - a).collect();

        // 3. Clone the shape before we drop the lock
        let new_shape = inner.shape.clone();

        // 4. Drop the lock as early as possible
        drop(inner);

        // 5. Use the constructor to properly initialize the Arc and Autograd fields
        Tensor::new(new_data, new_shape).unwrap()
    }
}

// Arithmetic Operations
impl_elementwise_op!(Add, add, +, AddBackward);
impl_elementwise_op!(Sub, sub, -, SubBackward);
impl_elementwise_op!(Div, div, /, DivBackward);
impl_elementwise_op!(Mul, mul, *, MulBackward);
