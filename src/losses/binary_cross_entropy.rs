use crate::tensor::{Tensor, tensor::TensorType};
use num_traits::{Float, FromPrimitive};

pub struct BinaryCrossEntropy;

impl BinaryCrossEntropy {
    pub fn new() -> Self {
        Self
    }

    pub fn forward<T>(&self, predictions: &Tensor<T>, targets: &Tensor<T>) -> T
    where
        T: Float + FromPrimitive + TensorType,
    {
        let first_term = targets.clone() * predictions.clone().ln();

        let one_minus_y = targets
            .clone()
            .sub_scalar(FromPrimitive::from_f32(1.0).unwrap());

        let one_minus_y_hat = predictions
            .clone()
            .sub_scalar(FromPrimitive::from_f32(1.0).unwrap());

        let second_term = one_minus_y * one_minus_y_hat.ln();

        let combined = first_term + second_term;

        let n = predictions.size() as f32;

        -combined.sum_all() / FromPrimitive::from_f32(n).unwrap()
    }
}
