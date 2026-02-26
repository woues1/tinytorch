use std::ops::AddAssign;

use crate::tensor::Tensor;
use num_traits::{Float, FromPrimitive, real::Real};

impl<T> Tensor<T>
where
    T: Float + FromPrimitive + AddAssign + Default,
{
    pub fn relu(mut self) -> Self {
        let zero = T::zero();
        for val in self.data.iter_mut() {
            *val = val.max(zero);
        }
        self
    }

    pub fn sigmoid(mut self) -> Self {
        let one = T::one();
        for val in self.data.iter_mut() {
            *val = one / (one + (-*val).exp());
        }
        self
    }

    pub fn tanh(mut self) -> Self {
        for val in self.data.iter_mut() {
            *val = val.tanh();
        }
        self
    }

    pub fn softmax(&self, dim: usize) -> Self {
        let max_vals = self.max_dim(dim);

        let shifted = self.clone() - max_vals;

        let exp_vals = shifted.exp();

        let sum_exp = exp_vals.sum_dim(dim);

        exp_vals / sum_exp
    }

    pub fn log_softmax(&self, dim: usize) -> Self
    where
        Tensor<T>: Real,
    {
        let max_vals = self.max_dim(dim);
        let shifted = self.clone() - max_vals;

        let exp_vals = shifted.clone().exp();
        let sum_exp = exp_vals.sum_dim(dim);
        let log_sum_exp = sum_exp.ln();

        shifted - log_sum_exp
    }

    pub fn gelu(mut self) -> Self {
        let half = T::from_f32(0.5).unwrap();
        let sqrt_2_over_pi = T::from_f32(0.7978845608).unwrap();
        let coef = T::from_f32(0.044715).unwrap();
        let one = T::one();

        for val in self.data.iter_mut() {
            let x = *val;
            let x_cubed = x * x * x;

            let inner = sqrt_2_over_pi * (x + coef * x_cubed);

            *val = half * x * (one + inner.tanh());
        }

        self
    }
}
