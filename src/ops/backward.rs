use crate::tensor::Tensor;
use crate::tensor::tensor::BackwardOp;

// 1. Define the struct to hold the parent tensors
pub struct AddBackward<T> {
    pub a: Tensor<T>,
    pub b: Tensor<T>,
}
impl<T> BackwardOp<T> for AddBackward<T>
where
    T: Copy + std::ops::Add<Output = T> + Send + Sync,
{
    fn backward(&self, grad_output: &Tensor<T>) {
        // Here we will add the incoming grad_output to a and b
    }
}

pub struct SubBackward<T> {
    pub a: Tensor<T>,
    pub b: Tensor<T>,
}
impl<T> BackwardOp<T> for SubBackward<T>
where
    T: Copy + std::ops::Sub<Output = T> + Send + Sync,
{
    fn backward(&self, grad_output: &Tensor<T>) {
        // Here we will add the incoming grad_output to a and b
    }
}

pub struct DivBackward<T> {
    pub a: Tensor<T>,
    pub b: Tensor<T>,
}
impl<T> BackwardOp<T> for DivBackward<T>
where
    T: Copy + std::ops::Div<Output = T> + Send + Sync,
{
    fn backward(&self, grad_output: &Tensor<T>) {
        // Here we will add the incoming grad_output to a and b
    }
}

pub struct MulBackward<T> {
    pub a: Tensor<T>,
    pub b: Tensor<T>,
}
impl<T> BackwardOp<T> for MulBackward<T>
where
    T: Copy + std::ops::Mul<Output = T> + Send + Sync,
{
    fn backward(&self, grad_output: &Tensor<T>) {
        // Here we will add the incoming grad_output to a and b
    }
}
