use crate::{nn::Layer, tensor::Tensor};
use rand::RngExt;

pub struct Dropout {
    pub p: f32, // The probability of dropping an element (e.g., 0.5)
}
impl Dropout {
    pub fn new(p: f32) -> Self {
        assert!(p >= 0.0 && p < 1.0);
        Self { p }
    }
}
impl Layer<f32> for Dropout {
    fn forward(&self, input: &Tensor<f32>, training: bool) -> Tensor<f32> {
        if !training {
            return input.clone();
        }
        let keep_probability = 1.0 - self.p;
        let scale = 1.0 / keep_probability;
        let mut rng = rand::rng();

        let inner = input.inner.read().unwrap();
        let new_shape = inner.shape.clone();

        let new_data = inner
            .data
            .iter()
            .map(|a| {
                a * if rng.random_range(0.0..1.0) < keep_probability {
                    scale
                } else {
                    0.0
                }
            })
            .collect();
        drop(inner);
        Tensor::new(new_data, new_shape).unwrap()
    }

    /// Dropout has no learnable weights
    fn parameters(&self) -> Vec<Tensor<f32>> {
        vec![]
    }
}
