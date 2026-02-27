use crate::nn::MLP;
use crate::tensor::Tensor;
mod tests {
    use super::*;

    #[test]
    fn test_deep_mlp_forward() {
        // 1. Create a dummy input batch: Shape [4, 3] (4 items, 3 features)
        let input_data = vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0];
        let input = Tensor::new(input_data, vec![4, 3]).unwrap();

        // 2. Define a 16-Layer Architecture!
        // 3 inputs -> fifteen layers of size 64 -> 2 outputs
        let mut sizes = vec![3];
        for _ in 0..15 {
            sizes.push(64);
        }
        sizes.push(2);

        // 3. Initialize the deep network
        let model = MLP::new(&sizes);
        assert_eq!(model.layers.len(), 16); // Verify we built 16 layers

        // 4. Run the forward pass
        let predictions = model.forward(&input);

        println!("\n=== Deep MLP Forward Pass ===");
        let input_inner = input.inner.read().unwrap();
        let pred_inner = predictions.inner.read().unwrap();
        println!("Input Shape:       {:?}", input_inner.shape);
        println!("Total Layers:      {}", model.layers.len());
        println!("Final Output Shape:{:?}", pred_inner.shape);

        // The final shape should still perfectly resolve to [4, 2]
        assert_eq!(pred_inner.shape, vec![4, 2]);
    }
}
