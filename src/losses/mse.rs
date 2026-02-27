use crate::tensor::{Tensor, tensor::TensorType};
use num_traits::{Float, FromPrimitive};
pub struct MSELoss;

impl MSELoss {
    pub fn new() -> Self {
        Self
    }

    pub fn forward<T>(&self, predictions: &Tensor<T>, targets: &Tensor<T>) -> T
    where
        T: Float + FromPrimitive + TensorType,
    {
        let p_shape = { predictions.inner.read().unwrap().shape.clone() };
        let t_shape = { targets.inner.read().unwrap().shape.clone() };

        assert_eq!(
            p_shape, t_shape,
            "Predictions and targets must have the same shape for MSE"
        );

        let n_elements = predictions.size();
        let n = T::from_usize(n_elements).unwrap();

        let diff = predictions.clone() - targets.clone();
        let squared = diff.clone() * diff;
        let sum = squared.sum_all();

        sum / n
    }
}
