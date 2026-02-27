use crate::tensor::Tensor;
mod tests {
    use num_traits::Float;

    use super::*;

    #[test]
    fn test_gelu() {
        let data = vec![-3.0, -1.0, 0.0, 1.0, 3.0];
        let tensor = Tensor::new(data, vec![5]).unwrap();
        let result = tensor.gelu();

        println!("\n=== GELU Outputs ===");
        let result_inner = result.inner.read().unwrap();
        println!("{:?}", result_inner.data);

        // Expected approximate values:
        // GELU(-3) ≈ -0.004
        // GELU(-1) ≈ -0.158
        // GELU(0)  = 0.0
        // GELU(1)  ≈ 0.841
        // GELU(3)  ≈ 2.996

        assert!((result_inner.data[2] - 0.0).abs() < 1e-4); // 0 stays 0
        assert!(result_inner.data[1] < 0.0); // -1.0 dips slightly negative
        assert!(result_inner.data[3] > 0.8 && result_inner.data[3] < 0.9); // 1.0 scales up
    }
}
