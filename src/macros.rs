#[macro_export]
macro_rules! impl_elementwise_op {
    ($trait:ident, $method:ident, $op:tt, $backward_op:ident) => {
        impl<T> std::ops::$trait for Tensor<T>
        where
            T: std::ops::$trait<Output = T> + Copy + Default + TensorType,
        {
            type Output = Self;

            fn $method(self, rhs: Self) -> Self::Output {
                // 1. Check if we can do the FAST PATH, then immediately drop the locks!
                let (is_fast_path, requires_grad_flag) = {
                    let inner_self = self.inner.read().unwrap();
                    let inner_rhs = rhs.inner.read().unwrap();

                    let req_grad = inner_self.requires_grad || inner_rhs.requires_grad;
                    let fast_path = inner_self.shape == inner_rhs.shape;

                    (fast_path, req_grad)
                };

                if is_fast_path {
                    // Re-acquire locks safely for data processing
                    let inner_self = self.inner.read().unwrap();
                    let inner_rhs = rhs.inner.read().unwrap();
                    let new_shape = inner_self.shape.clone();

                    // Use .iter() and map over references (&a, &b)
                    let new_data = inner_self.data
                        .iter()
                        .zip(inner_rhs.data.iter())
                        .map(|(&a, &b)| a $op b)
                        .collect();

                    drop(inner_self);
                    drop(inner_rhs);
                    return Tensor::new(new_data, new_shape).unwrap();
                }

                // 2. SLOW PATH (Broadcasting)
                let (new_shape, stride_a, stride_b) = self.broadcast_shapes(&rhs);
                let total_elements: usize = new_shape.iter().product();
                let mut new_data = Vec::with_capacity(total_elements);

                // 3. Re-acquire locks to do the actual math
                let inner_self = self.inner.read().unwrap();
                let inner_rhs = rhs.inner.read().unwrap();

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

                    new_data.push(inner_self.data[index_a] $op inner_rhs.data[index_b]);
                }

                drop(inner_self);
                drop(inner_rhs);

                let new_tensor = Tensor::new(new_data, new_shape).unwrap();


                if requires_grad_flag {
                    let mut inner_new = new_tensor.inner.write().unwrap();
                    inner_new.requires_grad = true;
                    inner_new.grad_fn = Some(std::sync::Arc::new(
                        crate::ops::$backward_op {
                            a: self.clone(),
                            b: rhs.clone(),
                        }
                    ));
                }

                return new_tensor;

            }
        }
    };
}
