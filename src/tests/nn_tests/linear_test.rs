use crate::nn::Linear;
mod tests {
    use super::*;
    use crate::{nn::Layer, tensor::Tensor}; // Adjust if your import path is different

    #[test]
    fn test_linear_forward() {
        // 1. Create a dummy input batch: 4 items, 3 features each.
        // Shape: [4, 3]
        let input_data = vec![
            1.0, 1.0, 1.0, // Item 1
            2.0, 2.0, 2.0, // Item 2
            3.0, 3.0, 3.0, // Item 3
            4.0, 4.0, 4.0, // Item 4
        ];
        let input = Tensor::new(input_data, vec![4, 3]).unwrap();
        let input_inner = input.inner.read().unwrap();
        // 2. Create the Linear Layer
        // In Features: 3, Out Features: 5, Use Bias: true
        // Weight Shape will be [3, 5], Bias Shape will be [1, 5]
        let layer = Linear::new(3, 5, true);
        let layer_inner = layer.weight.inner.read().unwrap();

        // 3. Run the forward pass!
        // Mathematically: ( [4, 3] @ [3, 5] ) + [1, 5]
        let output = layer.forward(&input, false);

        // 4. Print the shapes to visually confirm the broadcasting and matmul

        println!("\n=== Linear Layer Forward Pass ===");
        println!("Input Shape:  {:?}", input_inner.shape);
        println!("Weight Shape: {:?}", layer_inner.shape);
        println!(
            "Bias Shape:   {:?}",
            layer.bias.as_ref().unwrap().inner.read().unwrap().shape
        );
        let output_inner = output.inner.read().unwrap();
        println!("Output Shape: {:?}", output_inner.shape);
        println!("Output Data:  {:?}\n", output_inner.data);

        // 5. Assert the final shape is correct
        // The batch size (4) should remain unchanged,
        // but the features should change from 3 to 5.
        assert_eq!(output_inner.shape, vec![4, 5]);

        // Ensure the underlying data array was sized correctly
        assert_eq!(output_inner.data.len(), 20);
    }
}
