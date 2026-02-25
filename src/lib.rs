#[macro_use]
pub mod macros;
pub mod activations;
pub mod reductions;
pub mod tensor;
pub use tensor::Tensor;
pub mod layers;
#[cfg(test)]
pub mod tests;
