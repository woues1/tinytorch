use crate::tensor::Tensor;
use crate::tensor::tensor::{BackwardOp, TensorType};

// 1. Define the struct to hold the parent tensors
pub struct AddBackward<T> {
    pub a: Tensor<T>,
    pub b: Tensor<T>,
}
impl<T> BackwardOp<T> for AddBackward<T>
where
    T: TensorType,
{
    fn backward(&self, grad_output: &Tensor<T>) {
        let mut inner_a = self.a.inner.write().unwrap();
        if let Some(existing_grad) = &inner_a.grad {
            inner_a.grad = Some(existing_grad.clone() + grad_output.clone());
        } else {
            inner_a.grad = Some(grad_output.clone());
        }

        let mut inner_b = self.b.inner.write().unwrap();
        if let Some(existing_grad) = &inner_b.grad {
            inner_b.grad = Some(existing_grad.clone() + grad_output.clone());
        } else {
            inner_b.grad = Some(grad_output.clone());
        }
    }
}

pub struct SubBackward<T> {
    pub a: Tensor<T>,
    pub b: Tensor<T>,
}
impl<T> BackwardOp<T> for SubBackward<T>
where
    T: TensorType,
{
    fn backward(&self, grad_output: &Tensor<T>) {
        let grad_for_a = grad_output.clone();
        let grad_for_b = -grad_output.clone();

        let mut inner_a = self.a.inner.write().unwrap();
        if let Some(existing_grad) = &inner_a.grad {
            inner_a.grad = Some(existing_grad.clone() + grad_for_a.clone());
        } else {
            inner_a.grad = Some(grad_for_a);
        }

        let mut inner_b = self.b.inner.write().unwrap();
        if let Some(existing_grad) = &inner_b.grad {
            inner_b.grad = Some(existing_grad.clone() + grad_for_b.clone());
        } else {
            inner_b.grad = Some(grad_for_b);
        }
    }
}

pub struct DivBackward<T> {
    pub a: Tensor<T>,
    pub b: Tensor<T>,
}
impl<T> BackwardOp<T> for DivBackward<T>
where
    T: TensorType,
{
    fn backward(&self, grad_output: &Tensor<T>) {
        let grad_for_a = grad_output.clone() / self.b.clone();

        let grad_for_b =
            -(grad_output.clone() * self.a.clone() / (self.b.clone() * self.b.clone()));

        let mut inner_a = self.a.inner.write().unwrap();

        if let Some(existing_grad) = &inner_a.grad {
            inner_a.grad = Some(existing_grad.clone() + grad_for_a.clone())
        } else {
            inner_a.grad = Some(grad_for_a)
        }

        let mut inner_b = self.b.inner.write().unwrap();

        if let Some(existing_grad) = &inner_b.grad {
            inner_b.grad = Some(existing_grad.clone() + grad_for_b.clone())
        } else {
            inner_b.grad = Some(grad_for_b)
        }
    }
}

pub struct MulBackward<T> {
    pub a: Tensor<T>,
    pub b: Tensor<T>,
}
impl<T> BackwardOp<T> for MulBackward<T>
where
    T: TensorType,
{
    fn backward(&self, grad_output: &Tensor<T>) {
        let grad_for_a = grad_output.clone() * self.b.clone();
        let grad_for_b = grad_output.clone() * self.a.clone();

        let mut inner_a = self.a.inner.write().unwrap();

        if let Some(existing_grad) = &inner_a.grad {
            inner_a.grad = Some(existing_grad.clone() + grad_for_a.clone());
        } else {
            inner_a.grad = Some(grad_for_a.clone());
        }

        let mut inner_b = self.b.inner.write().unwrap();

        if let Some(existing_grad) = &inner_b.grad {
            inner_b.grad = Some(existing_grad.clone() + grad_for_b.clone());
        } else {
            inner_b.grad = Some(grad_for_b.clone());
        }
    }
}
pub struct NegBackward<T> {
    pub a: Tensor<T>,
}
impl<T> BackwardOp<T> for NegBackward<T>
where
    T: TensorType,
{
    fn backward(&self, grad_output: &Tensor<T>) {
        let mut inner_a = self.a.inner.write().unwrap();

        let grad_for_a = -grad_output.clone();

        if let Some(existing_grad) = &inner_a.grad {
            inner_a.grad = Some(existing_grad.clone() + grad_for_a.clone())
        } else {
            inner_a.grad = Some(grad_for_a)
        }
    }
}
