use crate::tensor::Tensor;
use num_traits::{Float, FromPrimitive};

impl<T> Tensor<T>
where
    T: Float + FromPrimitive,
{
    /// Sums all elements in the tensor into a single scalar value.
    pub fn sum_all(&self) -> T {
        let inner = self.inner.read().unwrap();
        inner.data.iter().fold(T::zero(), |acc, &x| acc + x)
    }

    /// Finds the maximum value across all elements in the tensor.
    pub fn max_all(&self) -> T {
        let inner = self.inner.read().unwrap();
        inner.data.iter().fold(T::min_value(), |acc, &x| acc.max(x))
    }

    pub fn mean(&self) -> T {
        let count = self.size();
        let number_of_items = FromPrimitive::from_usize(count).unwrap();
        self.sum_all() / number_of_items
    }

    /// Element-wise exponential (e^x)
    pub fn exp(&self) -> Self {
        let inner = self.inner.read().unwrap();
        let new_shape = inner.shape.clone();
        let new_data = inner.data.iter().map(|x| x.exp()).collect();
        drop(inner);
        Tensor::new(new_data, new_shape.clone()).unwrap()
    }

    pub fn max_dim(&self, dim: usize) -> Self
    where
        T: num_traits::Float,
    {
        let inner = self.inner.read().unwrap();
        let mut new_shape = inner.shape.clone();
        new_shape[dim] = 1;

        let mut new_strides = vec![0; new_shape.len()];
        let mut current_stride = 1;
        for i in (0..new_shape.len()).rev() {
            new_strides[i] = current_stride;
            current_stride *= new_shape[i];
        }

        let total_elements = new_shape.iter().product();
        let mut new_data = vec![T::min_value(); total_elements];

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

            new_data[target_idx] = new_data[target_idx].max(inner.data[i]);
        }

        // Drop lock before constructing the new tensor
        drop(inner);
        Tensor::new(new_data, new_shape).unwrap()
    }

    pub fn ln(&self) -> Self
    where
        T: num_traits::Float,
    {
        let inner = self.inner.read().unwrap();

        // Out-of-place mapping using references
        let new_data: Vec<T> = inner.data.iter().map(|&val| val.ln()).collect();
        let new_shape = inner.shape.clone();

        drop(inner);

        Tensor::new(new_data, new_shape).unwrap()
    }
}
