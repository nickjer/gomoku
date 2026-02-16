use fastrand_contrib::RngExt;
use serde::{Deserialize, Serialize};

use crate::evolution::crossover::uniform_crossover;
use crate::evolution::mutation::gaussian_mutate;

/// A single convolutional layer's parameters (weights + bias).
///
/// Owns its data with compile-time dimensions so they cannot be mismatched.
/// Validated at construction time.
///
/// # Type Parameters
/// - `IN_C`: Number of input channels
/// - `OUT_C`: Number of output channels
/// - `K`: Kernel size (e.g., 3 for 3×3 kernels)
///
/// # Weight Layout
/// Weights are in `[IN_C * K * K][OUT_C]` layout.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConvParams<const IN_C: usize, const OUT_C: usize, const K: usize> {
    weights: Vec<f32>,
    bias: Vec<f32>,
}

impl<const IN_C: usize, const OUT_C: usize, const K: usize> ConvParams<IN_C, OUT_C, K> {
    /// Workspace stride: number of elements per position in the gathered neighborhood.
    pub const STRIDE: usize = IN_C
        .checked_mul(K)
        .expect("STRIDE overflow")
        .checked_mul(K)
        .expect("STRIDE overflow");

    /// Expected weights length: `IN_C * K * K * OUT_C`.
    const EXPECTED_WEIGHTS: usize = Self::STRIDE
        .checked_mul(OUT_C)
        .expect("EXPECTED_WEIGHTS overflow");

    /// Creates a new `ConvParams` with validated dimensions.
    ///
    /// # Panics
    ///
    /// Panics if `weights.len() != IN_C * K * K * OUT_C` or `bias.len() != OUT_C`.
    #[must_use]
    pub fn new(weights: Vec<f32>, bias: Vec<f32>) -> Self {
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

    /// Creates parameters initialized using He initialization.
    ///
    /// Uses `N(0, √(2/n_in))` which is optimal for `ReLU` networks.
    /// Biases are initialized to zero.
    #[must_use]
    pub fn random(rng: &mut fastrand::Rng) -> Self {
        let std = he_std(Self::STRIDE);
        let weights: Vec<f32> = (0..Self::EXPECTED_WEIGHTS)
            .map(|_| rng.f32_normal(0.0, std))
            .collect();
        let bias = vec![0.0; OUT_C];
        Self::new(weights, bias)
    }

    /// Returns the weight slice in `[IN_C * K * K][OUT_C]` layout.
    #[must_use]
    pub fn weights(&self) -> &[f32] {
        assert!(self.weights.len() == Self::EXPECTED_WEIGHTS);
        &self.weights
    }

    /// Returns the bias slice of length `OUT_C`.
    #[must_use]
    pub fn bias(&self) -> &[f32] {
        assert!(self.bias.len() == OUT_C);
        &self.bias
    }

    /// Performs uniform crossover with another set of parameters.
    #[must_use]
    pub fn uniform_crossover(&self, other: &Self, rng: &mut fastrand::Rng) -> Self {
        Self {
            weights: uniform_crossover(&self.weights, &other.weights, rng),
            bias: uniform_crossover(&self.bias, &other.bias, rng),
        }
    }

    /// Performs Gaussian mutation on these parameters.
    #[must_use]
    pub fn gaussian_mutate(&self, sigma: f32, rng: &mut fastrand::Rng) -> Self {
        Self {
            weights: gaussian_mutate(&self.weights, sigma, rng),
            bias: gaussian_mutate(&self.bias, sigma, rng),
        }
    }
}

/// Computes He initialization standard deviation: `√(2/n_in)`.
fn he_std(n_in: usize) -> f32 {
    // Safely downcast to u16 (valid up to 65,535), then losslessly convert to f32.
    let n_in_u16 = u16::try_from(n_in).expect("n_in exceeds u16::MAX");
    (2.0 / f32::from(n_in_u16)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_params_construction() {
        let weights = vec![0.0f32; 2 * 3 * 3 * 4]; // in=2, k=3, out=4
        let bias = vec![0.0f32; 4];

        let params = ConvParams::<2, 4, 3>::new(weights, bias);

        assert_eq!(params.weights().len(), 72);
        assert_eq!(params.bias().len(), 4);
    }

    #[test]
    fn one_by_one_kernel_params() {
        let weights = vec![0.0f32; 8]; // in=8, k=1, out=1
        let bias = vec![0.0f32; 1];

        let params = ConvParams::<8, 1, 1>::new(weights, bias);

        assert_eq!(params.weights().len(), 8);
        assert_eq!(params.bias().len(), 1);
    }

    #[test]
    #[should_panic(expected = "weights length mismatch")]
    fn rejects_wrong_weights_length() {
        let weights = vec![0.0f32; 10]; // wrong
        let bias = vec![0.0f32; 4];
        let _ = ConvParams::<2, 4, 3>::new(weights, bias);
    }

    #[test]
    #[should_panic(expected = "bias length mismatch")]
    fn rejects_wrong_bias_length() {
        let weights = vec![0.0f32; 2 * 3 * 3 * 4];
        let bias = vec![0.0f32; 3]; // wrong
        let _ = ConvParams::<2, 4, 3>::new(weights, bias);
    }

    #[test]
    fn random_creates_correct_lengths() {
        let mut rng = fastrand::Rng::with_seed(42);
        let params = ConvParams::<2, 4, 3>::random(&mut rng);

        assert_eq!(params.weights().len(), 2 * 3 * 3 * 4);
        assert_eq!(params.bias().len(), 4);
    }

    #[test]
    fn random_biases_initialized_to_zero() {
        let mut rng = fastrand::Rng::with_seed(42);
        let params = ConvParams::<2, 8, 3>::random(&mut rng);

        assert!(params.bias().iter().all(|&b| b == 0.0));
    }

    #[test]
    fn he_initialization_has_reasonable_distribution() {
        let mut rng = fastrand::Rng::with_seed(42);
        let params = ConvParams::<2, 8, 3>::random(&mut rng);

        let weights = params.weights();
        let mean: f32 = weights.iter().sum::<f32>() / weights.len() as f32;
        let variance: f32 =
            weights.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / weights.len() as f32;
        let std_dev = variance.sqrt();

        // Expected std: sqrt(2 / (2 * 9)) = sqrt(1/9) ≈ 0.333
        let expected_std = he_std(2 * 3 * 3);
        assert!(
            (std_dev - expected_std).abs() < 0.1,
            "std_dev {std_dev} not close to expected {expected_std}"
        );
    }

    #[test]
    fn uniform_crossover_produces_valid_child() {
        let mut rng = fastrand::Rng::with_seed(42);
        let parent1 = ConvParams::<2, 4, 3>::random(&mut rng);
        let parent2 = ConvParams::<2, 4, 3>::random(&mut rng);

        let child = parent1.uniform_crossover(&parent2, &mut rng);

        assert_eq!(child.weights().len(), parent1.weights().len());
        assert_eq!(child.bias().len(), parent1.bias().len());
        for (i, &val) in child.weights().iter().enumerate() {
            assert!(val == parent1.weights()[i] || val == parent2.weights()[i]);
        }
    }

    #[test]
    fn gaussian_mutate_changes_values() {
        let params = ConvParams::<2, 4, 3>::new(vec![0.0f32; 2 * 3 * 3 * 4], vec![0.0f32; 4]);
        let mut rng = fastrand::Rng::with_seed(42);

        let result = params.gaussian_mutate(1.0, &mut rng);

        let changed = result.weights().iter().filter(|&&val| val != 0.0).count();
        assert!(changed > 0, "some weights should change");
    }
}
