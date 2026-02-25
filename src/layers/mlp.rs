use crate::layers::Linear;
use crate::tensor::Tensor;

pub struct MLP {
    pub layers: Vec<Linear>,
}

impl MLP {
    /// Creates an N-layer neural network.
    /// `sizes` should contain the input size, followed by all hidden sizes,
    /// and ending with the output size.
    /// Example: `[3, 16, 16, 2]` creates a 3-layer network.
    pub fn new(sizes: &[usize]) -> Self {
        assert!(
            sizes.len() >= 2,
            "An MLP needs at least an input and output size"
        );

        let mut layers = Vec::new();

        // Loop through the sizes in overlapping windows of 2
        // e.g., for [3, 16, 16], this gives (3, 16) then (16, 16)
        for i in 0..sizes.len() - 1 {
            let in_feat = sizes[i];
            let out_feat = sizes[i + 1];
            layers.push(Linear::new(in_feat, out_feat, true));
        }

        Self { layers }
    }

    /// The forward pass loops through all layers dynamically
    pub fn forward(&self, input: &Tensor<f32>) -> Tensor<f32> {
        // We start with the input tensor
        let mut current = input.clone();
        let total_layers = self.layers.len();

        for (i, layer) in self.layers.iter().enumerate() {
            // 1. Apply the linear transformation
            current = layer.forward(&current);

            // 2. Apply ReLU activation to hidden layers ONLY
            // We do NOT activate the final output layer, because the final
            // layer needs to output raw numbers (logits) for our Loss function!
            if i < total_layers - 1 {
                current = current.relu();
            }
        }

        current
    }
}
