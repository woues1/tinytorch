use crate::nn::layer::Layer;
use crate::tensor::Tensor;
use num_traits::{Float, FromPrimitive, real::Real};
use std::ops::AddAssign;

pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    GELU,
    Softmax { dim: usize },
    LogSoftmax { dim: usize },
}

impl<T> Layer<T> for Activation
where
    T: Float + FromPrimitive + AddAssign + Default,
    Tensor<T>: Real,
{
    fn forward(&self, input: &Tensor<T>, _is_training: bool) -> Tensor<T> {
        let t = input.clone();
        match self {
            Activation::ReLU => t.relu(),
            Activation::Sigmoid => t.sigmoid(),
            Activation::Tanh => t.tanh(),
            Activation::GELU => t.gelu(),
            Activation::Softmax { dim } => t.softmax(*dim),
            Activation::LogSoftmax { dim } => t.log_softmax(*dim),
        }
    }

    fn parameters(&self) -> Vec<Tensor<T>> {
        vec![]
    }
}
