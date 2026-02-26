use std::ops::AddAssign;

use crate::nn::Linear;
use crate::nn::layer::Layer; // 1. IMPORTANT: Must bring Trait into scope to use .forward()
use crate::tensor::tensor::Tensor;
use num_traits::{Float, FromPrimitive};
use rand_distr::StandardNormal;

pub struct MLP<T> {
    pub layers: Vec<Linear<T>>,
}

impl<T> MLP<T>
where
    // 2. Add bounds so T can handle the math and initialization
    T: Float
        + FromPrimitive
        + From<f32>
        + Default
        + std::ops::Add<Output = T>
        + std::ops::Mul<Output = T>
        + AddAssign,
    StandardNormal: rand_distr::Distribution<T>,
{
    pub fn new(sizes: &[usize]) -> Self
    where
        Tensor<T>: From<Tensor<f32>>, // Required by your Linear::new logic
    {
        assert!(
            sizes.len() >= 2,
            "An MLP needs at least an input and output size"
        );

        let mut layers = Vec::new();

        for i in 0..sizes.len() - 1 {
            let in_feat = sizes[i];
            let out_feat = sizes[i + 1];
            // 3. This now creates Linear<T> instead of Linear<f32>
            layers.push(Linear::<T>::new(in_feat, out_feat, true));
        }

        Self { layers }
    }

    // 4. Input and Output must be Tensor<T>, not Tensor<f32>
    pub fn forward(&self, input: &Tensor<T>) -> Tensor<T> {
        let mut current = input.clone();
        let total_layers = self.layers.len();

        for (i, layer) in self.layers.iter().enumerate() {
            // Now works because Layer trait is in scope and types match
            current = layer.forward(&current, true); // Assuming training mode for now

            if i < total_layers - 1 {
                current = current.relu(); // Ensure your Tensor class has a generic .relu()
            }
        }

        current
    }
}
