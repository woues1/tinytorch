use crate::nn::layer::Layer;
use crate::tensor::Tensor;
use crate::tensor::tensor::TensorType;
pub struct Sequential<T> {
    // Note: We use Box<dyn Layer<T>> to allow a mix of different layer types
    pub layers: Vec<Box<dyn Layer<T>>>,
}

impl<T> Layer<T> for Sequential<T>
where
    T: TensorType,
{
    fn forward(&self, input: &Tensor<T>, training: bool) -> Tensor<T> {
        self.layers.iter().fold(input.clone(), |acc, layer| {
            // Pass 'training' into every layer in the stack
            layer.forward(&acc, training)
        })
    }

    fn parameters(&self) -> Vec<Tensor<T>> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}
