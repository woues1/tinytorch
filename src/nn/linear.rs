use num_traits::{Float, FromPrimitive};
use rand_distr::StandardNormal;

use crate::nn::layer::Layer;
use crate::tensor::tensor::{Tensor, TensorType};

pub struct Linear<T> {
    pub weight: Tensor<T>,
    pub bias: Option<Tensor<T>>,
}

impl<T> Linear<T>
where
    T: From<f32> + TensorType,
    T: Float + FromPrimitive,
    StandardNormal: rand_distr::Distribution<T>,
{
    pub fn new(in_features: usize, out_features: usize, use_bias: bool) -> Self
    where
        Tensor<T>: From<Tensor<f32>>,
    {
        // 1. Initialize random weights and scale them
        let scale = (1.0 / in_features as f32).sqrt();
        let weight_f32 = Tensor::<f32>::randn(vec![in_features, out_features]).mul_scalar(scale);

        // Convert to our generic type T
        let weight = Tensor::<T>::from(weight_f32);

        // 2. Initialize biases to zero if requested
        let bias = if use_bias {
            Some(Tensor::<T>::zeros(vec![1, out_features]))
        } else {
            None
        };

        Self { weight, bias }
    }
}

impl<T> Layer<T> for Linear<T>
where
    T: TensorType,
{
    fn forward(&self, input: &Tensor<T>, _is_training: bool) -> Tensor<T> {
        // x @ W
        let mut output = input.matmul(&self.weight);

        if let Some(bias) = &self.bias {
            output = output + bias.clone();
        }

        output
    }

    fn parameters(&self) -> Vec<Tensor<T>> {
        let mut params = Vec::with_capacity(2);

        params.push(self.weight.clone());
        if let Some(b) = &self.bias {
            params.push(b.clone());
        }

        params
    }
}
