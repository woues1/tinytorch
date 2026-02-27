use crate::nn::{Dropout, Linear, Sequential};
use crate::tensor::Tensor;
mod tests {
    use crate::nn::Layer;

    use super::*;

    #[test]
    fn test_sequential_integration() {
        // 1. Build a small 2-layer MLP: [4 -> 8 -> 2]
        // We use Box::new to "erase" the specific types so they fit in the same Vec
        let model: Sequential<f32> = Sequential {
            layers: vec![
                Box::new(Linear::new(4, 8, true)),
                Box::new(Dropout::new(0.5)),
                Box::new(Linear::new(8, 2, true)),
            ],
        };

        let input = Tensor::<f32>::randn(vec![1, 4]);

        // 2. Test Training Pass (Dropout active)
        let out_train = model.forward(&input, true);

        // 3. Test Inference Pass (Dropout inactive)
        let out_eval = model.forward(&input, false);

        let out_train_inner = out_train.inner.read().unwrap();
        let out_eval_inner = out_eval.inner.read().unwrap();

        assert_eq!(out_train_inner.shape, vec![1, 2]);
        assert_eq!(out_eval_inner.shape, vec![1, 2]);

        // In theory, out_train and out_eval should be different
        // because Dropout was randomly zeroing things in the first one!
        println!("Training output: {:?}", out_train_inner.data);
        println!("Eval output: {:?}", out_eval_inner.data);
    }
}
