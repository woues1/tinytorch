use std::ops::AddAssign;

use crate::tensor::{Tensor, tensor::TensorType};
use num_traits::{Float, FromPrimitive};

impl<T> Tensor<T>
where
    T: Float + FromPrimitive + AddAssign + TensorType,
{
    pub fn relu(&self) -> Self {
        let zero = T::zero();

        let inner = self.inner.read().unwrap();

        let new_data: Vec<T> = inner.data.iter().map(|&a| a.max(zero)).collect();

        let new_shape = inner.shape.clone();

        drop(inner);

        Tensor::new(new_data, new_shape).unwrap()
    }

    pub fn sigmoid(&self) -> Self {
        let one = T::one();

        let inner = self.inner.read().unwrap();

        let new_data: Vec<T> = inner
            .data
            .iter()
            .map(|&a| one / (one + (-a).exp()))
            .collect();

        let new_shape = inner.shape.clone();

        drop(inner);

        Tensor::new(new_data, new_shape).unwrap()
    }

    pub fn tanh(&self) -> Self {
        let inner = self.inner.read().unwrap();

        let new_data: Vec<T> = inner.data.iter().map(|&a| a.tanh()).collect();

        let new_shape = inner.shape.clone();

        drop(inner);

        Tensor::new(new_data, new_shape).unwrap()
    }

    pub fn softmax(&self, dim: usize) -> Self {
        let max_vals = self.max_dim(dim);

        let shifted = self.clone() - max_vals;

        let exp_vals = shifted.exp();

        let sum_exp = exp_vals.sum_dim(dim);

        exp_vals / sum_exp
    }

    pub fn log_softmax(&self, dim: usize) -> Self {
        let max_vals = self.max_dim(dim);
        let shifted = self.clone() - max_vals;

        let exp_vals = shifted.clone().exp();
        let sum_exp = exp_vals.sum_dim(dim);
        let log_sum_exp = sum_exp.ln();

        shifted - log_sum_exp
    }

    pub fn gelu(&self) -> Self {
        let half = T::from_f32(0.5).unwrap();
        let sqrt_2_over_pi = T::from_f32(0.7978845608).unwrap();
        let coef = T::from_f32(0.044715).unwrap();
        let one = T::one();

        let inner = self.inner.read().unwrap();

        let new_data = inner
            .data
            .iter()
            .map(|&a| half * a * (one + (sqrt_2_over_pi * (a + coef * (a * a * a))).tanh()))
            .collect();

        let new_shape = inner.shape.clone();

        drop(inner);

        Tensor::new(new_data, new_shape).unwrap()
    }
}
