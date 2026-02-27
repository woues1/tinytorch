use crate::tensor::Tensor;
// Assuming your struct is exported from this module
use crate::losses::mse::MSELoss;

mod tests {
    use super::*;

    #[test]
    fn test_mse_loss() {
        // 1. Instantiate the loss function (the "criterion")
        let criterion = MSELoss::new();

        let targets = Tensor::new(vec![1.0, 0.0, 1.0, 0.0], vec![2, 2]).unwrap();

        // Perfect predictions should yield 0.0 loss
        let perfect_preds = Tensor::new(vec![1.0, 0.0, 1.0, 0.0], vec![2, 2]).unwrap();

        // 2. Call forward on the criterion
        let loss_perfect = criterion.forward(&perfect_preds, &targets);
        assert_eq!(loss_perfect, 0.0);

        // Off by 0.5 everywhere
        // Diff = 0.5 -> Squared = 0.25 -> Sum = 1.0 -> Mean = 0.25
        let bad_preds = Tensor::new(vec![1.5, -0.5, 1.5, -0.5], vec![2, 2]).unwrap();

        // No need to .clone() targets if forward() takes references!
        let loss_bad = criterion.forward(&bad_preds, &targets);
        assert_eq!(loss_bad, 0.25);
    }
}
