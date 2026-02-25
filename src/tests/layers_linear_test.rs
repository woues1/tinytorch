use crate::layers::Linear;
mod tests {
    use super::*;
    use crate::tensor::Tensor; // Adjust if your import path is different

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

        // 2. Create the Linear Layer
        // In Features: 3, Out Features: 5, Use Bias: true
        // Weight Shape will be [3, 5], Bias Shape will be [1, 5]
        let layer = Linear::new(3, 5, true);

        // 3. Run the forward pass!
        // Mathematically: ( [4, 3] @ [3, 5] ) + [1, 5]
        let output = layer.forward(&input);

        // 4. Print the shapes to visually confirm the broadcasting and matmul
        println!("\n=== Linear Layer Forward Pass ===");
        println!("Input Shape:  {:?}", input.shape);
        println!("Weight Shape: {:?}", layer.weight.shape);
        println!("Bias Shape:   {:?}", layer.bias.as_ref().unwrap().shape);
        println!("Output Shape: {:?}", output.shape);
        println!("Output Data:  {:?}\n", output.data);

        // 5. Assert the final shape is correct
        // The batch size (4) should remain unchanged,
        // but the features should change from 3 to 5.
        assert_eq!(output.shape, vec![4, 5]);

        // Ensure the underlying data array was sized correctly
        assert_eq!(output.data.len(), 20);
    }
}
