/// A view into a single convolutional layer's parameters (weights + bias).
///
/// Bundles weights and bias together with compile-time dimensions so they cannot be
/// mismatched. Validated at construction time.
///
/// # Type Parameters
/// - `IN_C`: Number of input channels
/// - `OUT_C`: Number of output channels
/// - `K`: Kernel size (e.g., 3 for 3×3 kernels)
///
/// # Weight Layout
/// Weights are in `[IN_C * K * K][OUT_C]` layout.
#[derive(Debug, Clone, Copy)]
pub struct ConvParams<'a, const IN_C: usize, const OUT_C: usize, const K: usize> {
    weights: &'a [f32],
    bias: &'a [f32],
}

impl<'a, const IN_C: usize, const OUT_C: usize, const K: usize> ConvParams<'a, IN_C, OUT_C, K> {
    /// Workspace stride: number of elements per position in the gathered neighborhood.
    const STRIDE: usize = IN_C * K * K;

    /// Expected weights length: `IN_C * K * K * OUT_C`.
    const EXPECTED_WEIGHTS: usize = Self::STRIDE * OUT_C;

    /// Creates a new `ConvParams` with validated dimensions.
    ///
    /// # Panics
    ///
    /// Panics if `weights.len() != IN_C * K * K * OUT_C` or `bias.len() != OUT_C`.
    #[must_use]
    pub fn new(weights: &'a [f32], bias: &'a [f32]) -> Self {
        assert_eq!(
            weights.len(),
            Self::EXPECTED_WEIGHTS,
            "weights length mismatch: expected {}, got {}",
            Self::EXPECTED_WEIGHTS,
            weights.len()
        );
        assert_eq!(
            bias.len(),
            OUT_C,
            "bias length mismatch: expected {OUT_C}, got {}",
            bias.len()
        );
        Self { weights, bias }
    }

    /// Returns the weight slice in `[IN_C * K * K][OUT_C]` layout.
    #[must_use]
    pub const fn weights(&self) -> &'a [f32] {
        assert!(self.weights.len() == Self::EXPECTED_WEIGHTS);
        self.weights
    }

    /// Returns the bias slice of length `OUT_C`.
    #[must_use]
    pub const fn bias(&self) -> &'a [f32] {
        assert!(self.bias.len() == OUT_C);
        self.bias
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_params_construction() {
        let weights = vec![0.0f32; 2 * 3 * 3 * 4]; // in=2, k=3, out=4
        let bias = vec![0.0f32; 4];

        let params = ConvParams::<2, 4, 3>::new(&weights, &bias);

        assert_eq!(params.weights().len(), 72);
        assert_eq!(params.bias().len(), 4);
    }

    #[test]
    fn one_by_one_kernel_params() {
        let weights = vec![0.0f32; 8]; // in=8, k=1, out=1
        let bias = vec![0.0f32; 1];

        let params = ConvParams::<8, 1, 1>::new(&weights, &bias);

        assert_eq!(params.weights().len(), 8);
        assert_eq!(params.bias().len(), 1);
    }

    #[test]
    #[should_panic(expected = "weights length mismatch")]
    fn rejects_wrong_weights_length() {
        let weights = vec![0.0f32; 10]; // wrong
        let bias = vec![0.0f32; 4];
        let _ = ConvParams::<2, 4, 3>::new(&weights, &bias);
    }

    #[test]
    #[should_panic(expected = "bias length mismatch")]
    fn rejects_wrong_bias_length() {
        let weights = vec![0.0f32; 2 * 3 * 3 * 4];
        let bias = vec![0.0f32; 3]; // wrong
        let _ = ConvParams::<2, 4, 3>::new(&weights, &bias);
    }
}
