use crate::Tensor;
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
}
