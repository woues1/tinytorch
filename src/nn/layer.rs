use crate::tensor::Tensor;
pub trait Layer<T> {
    fn forward(&self, input: &Tensor<T>, is_training: bool) -> Tensor<T>;
    fn parameters(&self) -> Vec<Tensor<T>>; // for gradient updates
}
