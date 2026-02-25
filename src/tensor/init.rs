use crate::tensor::Tensor;
use num_traits::{Float, FromPrimitive};
use rand::RngExt;
use rand_distr::StandardNormal;

impl<T> Tensor<T>
where
    // T must be a Float, be convertible from f32, be sampleable by rand, and be clonable
    T: Float + FromPrimitive + Clone,
    StandardNormal: rand_distr::Distribution<T>,
{
    /// Creates a tensor filled with random numbers drawn from a
    /// standard normal distribution (mean = 0, std = 1).
    pub fn randn(shape: Vec<usize>) -> Self {
        let total_elements = shape.iter().product();
        let mut rng = rand::rng();

        let data: Vec<T> = (0..total_elements)
            .map(|_| rng.sample(StandardNormal))
            .collect();

        Tensor::new(data, shape).unwrap()
    }

    /// Creates a new tensor filled entirely with zeros
    pub fn zeros(shape: Vec<usize>) -> Self {
        let total_elements: usize = shape.iter().product();

        // Use T::zero() instead of hardcoding 0.0
        let data = vec![T::zero(); total_elements];

        Tensor::new(data, shape).unwrap()
    }
}
