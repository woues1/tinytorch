use crate::tensor::Tensor;
use num_traits::Float;

impl<T> Tensor<T>
where
    T: Float,
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
}
