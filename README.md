# tinytorch

A minimal deep learning framework written in Rust. This is a personal challenge project to implement core neural network primitives from scratch, taking inspiration from PyTorch.

## Overview

tinytorch provides a generic `Tensor` type along with a set of neural network layers, activation functions, and loss functions sufficient to define and run inference on multi-layer perceptrons.

The release profile is configured for size optimization (`opt-level = 'z'`, LTO, stripped symbols), making the compiled binary suitable for resource-constrained environments.

## Features

### Tensor

- Generic `Tensor<T>` supporting any floating-point numeric type
- Multi-dimensional shape and stride management
- Broadcasting for element-wise operations
- Matrix multiplication (`matmul`)
- `reshape`, `transpose`
- Reductions: `sum_all`, `sum_dim`, `max_all`, `max_dim`, `mean`
- Built-in weight initialization utilities

### Neural Network Layers

- `Linear` - fully connected layer with optional bias
- `MLP` - stack of linear layers
- `Sequential` - generic layer container
- `Dropout` - regularization layer
- `Layer` trait for defining custom layers

### Activation Functions

- ReLU
- Sigmoid
- Tanh
- Softmax (numerically stable via max-shifting)
- Log-Softmax
- GELU

### Loss Functions

- Mean Squared Error (MSE)

## Project Structure

```
src/
├── tensor/           # Core tensor type and operations
├── nn/               # Neural network layers and traits
├── activations/      # Activation functions
├── losses/           # Loss functions
├── macros.rs         # impl_elementwise_op macro for broadcast ops
├── lib.rs            # Public API exports
└── tests/            # Integration and unit tests
```

## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `num-traits` | 0.2 | Generic numeric traits |
| `rand` | 0.10 | Random number generation |
| `rand_distr` | 0.6 | Statistical distributions for weight init |

## Building

```bash
cargo build
cargo build --release
```

## Testing

```bash
cargo test
```

The test suite covers tensor operations, broadcasting, linear layer forward passes, MLP inference, activation correctness, and MSE loss computation.

## Usage Example

```rust
use tinytorch::tensor::Tensor;
use tinytorch::nn::{Linear, Sequential, Layer};
use tinytorch::losses::mse;

// Build a small network
let mut net = Sequential::new(vec![
    Box::new(Linear::new(4, 8, true)),
    Box::new(Linear::new(8, 2, true)),
]);

// Forward pass
let input: Tensor<f32> = Tensor::zeros(vec![1, 4]);
let output = net.forward(&input);

// Compute loss against a target
let target: Tensor<f32> = Tensor::zeros(vec![1, 2]);
let loss = mse(&output, &target);
```
