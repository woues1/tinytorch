use crate::tensor::Tensor;
use num_traits::{Float, FromPrimitive};

impl<T> Tensor<T>
where
    T: Float + FromPrimitive,
{
    /// Sums all elements in the tensor into a single scalar value.
    pub fn sum_all(&self) -> T {
        self.data.iter().fold(T::zero(), |acc, &x| acc + x)
    }

    /// Finds the maximum value across all elements in the tensor.
    pub fn max_all(&self) -> T {
        self.data.iter().fold(T::min_value(), |acc, &x| acc.max(x))
    }

    pub fn mean(&self) -> T {
        let count = self.size();
        let number_of_items = FromPrimitive::from_usize(count).unwrap();
        self.sum_all() / number_of_items
    }

    /// Element-wise exponential (e^x)
    pub fn exp(&self) -> Self {
        let new_data = self.data.iter().map(|x| x.exp()).collect();
        Tensor::new(new_data, self.shape.clone()).unwrap()
    }

    pub fn max_dim(&self, dim: usize) -> Self {
        let mut new_shape = self.shape.clone();
        new_shape[dim] = 1;

        let mut new_strides = vec![0; new_shape.len()];
        let mut current_stride = 1;
        for i in (0..new_shape.len()).rev() {
            new_strides[i] = current_stride;
            current_stride *= new_shape[i];
        }

        let total_elements = new_shape.iter().product();
        let mut new_data = vec![T::min_value(); total_elements];

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

            new_data[target_idx] = new_data[target_idx].max(self.data[i]);
        }

        Tensor {
            data: new_data,
            shape: new_shape,
            strides: new_strides,
        }
    }
}
