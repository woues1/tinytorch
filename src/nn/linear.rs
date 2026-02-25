use crate::tensor::tensor::Tensor;

pub struct Linear {
    // Keeping it specific to f32 for simplicity right now
    pub weight: Tensor<f32>,
    pub bias: Option<Tensor<f32>>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, use_bias: bool) -> Self {
        // 1. Initialize random weights [IN, OUT]
        let weight = Tensor::randn(vec![in_features, out_features]);

        let scale = (1.0 / in_features as f32).sqrt();
        let weight = weight.mul_scalar(scale);
        // 2. Initialize biases to zero [1, OUT] if requested
        let bias = if use_bias {
            Some(Tensor::zeros(vec![1, out_features]))
        } else {
            None
        };

        Self { weight, bias }
    }

    pub fn forward(&self, input: &Tensor<f32>) -> Tensor<f32> {
        // x @ W
        let mut output = input.clone().matmul(self.weight.clone());

        // + b
        if let Some(bias) = &self.bias {
            output = output + bias.clone();
        }

        output
    }
    pub fn parameters(&mut self) -> Vec<&mut Tensor<f32>> {
        let mut params = Vec::new();

        params.push(&mut self.weight);

        if let Some(b) = &mut self.bias {
            params.push(b);
        }
        params
    }
}
