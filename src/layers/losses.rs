use crate::tensor::Tensor;

pub fn mse_loss(predictions: Tensor<f32>, targets: Tensor<f32>) -> f32 {
    assert_eq!(
        predictions.shape, targets.shape,
        "Perdictions and targets must have the same shape for MSE"
    );

    let n = predictions.shape.iter().product::<usize>() as f32;

    let diff = predictions - targets;

    let squared = diff.clone() * diff;

    let sum = squared.sum_all();

    sum / n
}
