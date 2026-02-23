#[macro_export]
macro_rules! impl_elementwise_op {
    // We take the Trait name, the method name, and the operator token
    ($trait:ident, $method:ident, $op:tt) => {
        impl<T> std::ops::$trait for Tensor<T>
        where
            T: std::ops::$trait<Output = T> + Copy + Default,
        {
            type Output = Self;

            fn $method(self, rhs: Self) -> Self::Output {
                // FAST PATH
                if self.shape == rhs.shape {
                    let new_shape = self.shape.clone();
                    let new_data = self.data
                        .into_iter()
                        .zip(rhs.data.into_iter())
                        .map(|(a, b)| a $op b) // <-- The operator gets inserted here!
                        .collect();
                    return Tensor::new(new_data, new_shape).unwrap();
                }

                // SLOW PATH (Broadcasting)
                let (new_shape, stride_a, stride_b) = self.broadcast_shapes(&rhs);
                let total_elements: usize = new_shape.iter().product();
                let mut new_data = Vec::with_capacity(total_elements);

                for flat_idx in 0..total_elements {
                    let mut current_val = flat_idx;
                    let mut index_a = 0;
                    let mut index_b = 0;

                    for d in (0..new_shape.len()).rev() {
                        let coord = current_val % new_shape[d];
                        current_val /= new_shape[d];

                        index_a += coord * stride_a[d];
                        index_b += coord * stride_b[d];
                    }


                    new_data.push(self.data[index_a] $op rhs.data[index_b]); // <-- And the operator gets inserted here too!
                }

                Tensor::new(new_data, new_shape).unwrap()
            }
        }
    };
}
