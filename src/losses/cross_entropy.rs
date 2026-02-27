use crate::tensor::{Tensor, tensor::TensorType};
use num_traits::{Float, FromPrimitive};

pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    pub fn new() -> Self {
        Self
    }

    pub fn forward<T>(&self, predictions: &Tensor<T>, targets: &Tensor<T>) -> T
    where
        T: Float + FromPrimitive + std::ops::AddAssign + TensorType,
    {
        // 1. Apply log_softmax along the class dimension (usually dim 1)
        let log_probs = predictions.log_softmax(1);

        // 2. Element-wise multiply (Targets should be one-hot encoded)
        let loss_tensor = targets.clone() * log_probs;

        // 3. Sum the values
        let total_loss = loss_tensor.sum_all();

        // 4. Convert batch size to generic type T and calculate the mean
        let predictions = predictions.inner.read().unwrap();
        let batch_size = T::from_usize(predictions.shape[0]).unwrap();

        // Return negative log-likelihood
        -total_loss / batch_size
    }
}
