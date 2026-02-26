use crate::losses::mse::mse_loss;
use crate::tensor::Tensor;
mod tests {

    use super::*;

    #[test]
    fn test_mse_loss() {
        let targets = Tensor::new(vec![1.0, 0.0, 1.0, 0.0], vec![2, 2]).unwrap();

        // Perfect predictions should yield 0.0 loss
        let perfect_preds = Tensor::new(vec![1.0, 0.0, 1.0, 0.0], vec![2, 2]).unwrap();
        let loss_perfect = mse_loss(&perfect_preds, &targets.clone());
        assert_eq!(loss_perfect, 0.0);

        // Off by 0.5 everywhere
        // Diff = 0.5 -> Squared = 0.25 -> Sum = 1.0 -> Mean = 0.25
        let bad_preds = Tensor::new(vec![1.5, -0.5, 1.5, -0.5], vec![2, 2]).unwrap();
        let loss_bad = mse_loss(&bad_preds, &targets);
        assert_eq!(loss_bad, 0.25);
    }
}
