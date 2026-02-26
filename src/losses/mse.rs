use crate::tensor::Tensor;
pub struct MSELoss;

impl MSELoss {
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, predictions: &Tensor<f32>, targets: &Tensor<f32>) -> f32 {
        mse_loss(predictions, targets)
    }
}

pub fn mse_loss(predictions: &Tensor<f32>, targets: &Tensor<f32>) -> f32 {
    assert_eq!(
        predictions.shape, targets.shape,
        "Predictions and targets must have the same shape for MSE"
    );

    let n = predictions.shape.iter().product::<usize>() as f32;

    let diff = predictions.clone() - targets.clone();
    let squared = diff.clone() * diff;
    let sum = squared.sum_all();

    sum / n
}
