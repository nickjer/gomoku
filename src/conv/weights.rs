use fastrand_contrib::RngExt;

use crate::evolution::crossover::{Crossover, uniform_crossover};
use crate::evolution::genes::EvolvableGenes;
use crate::evolution::mutation::{Mutation, gaussian_mutate};

use super::encoding::INPUT_CHANNELS;

/// Weights for a convolutional neural network with const-generic architecture.
///
/// # Type Parameters
/// - `K`: Kernel size (e.g., 3 for 3×3 kernels)
/// - `C`: Number of channels in hidden layers
/// - `L`: Number of convolutional layers (must be >= 1)
/// - `R`: Number of residual blocks (must be 0 until implemented)
///
/// # Architecture
/// - First conv: 2 input channels → C channels, K×K kernel
/// - L-1 hidden convs: C → C channels, K×K kernel
/// - Final conv: C → 1 channel, 1×1 kernel (position scoring)
///
/// # Weight Layout
/// Weights are stored in im2col-friendly `[patch_size][OUT_C]` layout, not standard
/// `[OUT_C][patch_size]`. This avoids runtime transposition during convolution.
#[derive(Debug, Clone)]
pub struct ConvWeights<const K: usize, const C: usize, const L: usize, const R: usize> {
    data: Vec<f32>,
}

impl<const K: usize, const C: usize, const L: usize, const R: usize> ConvWeights<K, C, L, R> {
    // First conv layer: INPUT_CHANNELS -> C, K×K kernel
    const FIRST_CONV_WEIGHTS: usize = INPUT_CHANNELS * C * K * K;
    const FIRST_CONV_BIAS: usize = C;
    const FIRST_CONV_TOTAL: usize = Self::FIRST_CONV_WEIGHTS + Self::FIRST_CONV_BIAS;

    // Hidden conv layers: C -> C, K×K kernel (L-1 layers)
    const HIDDEN_CONV_WEIGHTS: usize = C * C * K * K;
    const HIDDEN_CONV_BIAS: usize = C;
    const HIDDEN_CONV_TOTAL: usize = Self::HIDDEN_CONV_WEIGHTS + Self::HIDDEN_CONV_BIAS;
    const ALL_HIDDEN_TOTAL: usize = (L - 1) * Self::HIDDEN_CONV_TOTAL;

    // Final conv layer: C -> 1, 1×1 kernel
    const FINAL_CONV_WEIGHTS: usize = C;
    const FINAL_CONV_BIAS: usize = 1;
    const FINAL_CONV_TOTAL: usize = Self::FINAL_CONV_WEIGHTS + Self::FINAL_CONV_BIAS;

    /// Total number of weights in the network.
    pub const TOTAL: usize =
        Self::FIRST_CONV_TOTAL + Self::ALL_HIDDEN_TOTAL + Self::FINAL_CONV_TOTAL;

    // Offset calculations for slice accessors
    const FIRST_WEIGHTS_START: usize = 0;
    const FIRST_BIAS_START: usize = Self::FIRST_CONV_WEIGHTS;
    const HIDDEN_START: usize = Self::FIRST_CONV_TOTAL;
    const FINAL_START: usize = Self::FIRST_CONV_TOTAL + Self::ALL_HIDDEN_TOTAL;

    /// Creates weights initialized using He initialization.
    ///
    /// Uses `N(0, √(2/n_in))` for each layer, which is optimal for `ReLU` networks.
    ///
    /// # Panics
    ///
    /// Panics if `L < 1` or `R > 0` (residual blocks not yet implemented).
    #[must_use]
    pub fn random(rng: &mut fastrand::Rng) -> Self {
        assert!(L >= 1, "must have at least 1 layer");
        assert!(R == 0, "residual blocks not yet implemented");

        let mut data = Vec::with_capacity(Self::TOTAL);

        // First conv: He init with n_in = INPUT_CHANNELS * K * K
        let first_std = he_std(INPUT_CHANNELS * K * K);
        data.extend((0..Self::FIRST_CONV_WEIGHTS).map(|_| rng.f32_normal(0.0, first_std)));
        data.resize(data.len() + Self::FIRST_CONV_BIAS, 0.0); // Biases initialized to zero

        // Hidden convs: He init with n_in = C * K * K
        let hidden_std = he_std(C * K * K);
        for _ in 0..(L - 1) {
            data.extend((0..Self::HIDDEN_CONV_WEIGHTS).map(|_| rng.f32_normal(0.0, hidden_std)));
            data.resize(data.len() + Self::HIDDEN_CONV_BIAS, 0.0);
        }

        // Final conv: He init with n_in = C (1×1 kernel)
        let final_std = he_std(C);
        data.extend((0..Self::FINAL_CONV_WEIGHTS).map(|_| rng.f32_normal(0.0, final_std)));
        data.push(0.0); // Final bias

        debug_assert_eq!(data.len(), Self::TOTAL);
        Self { data }
    }

    /// Creates weights from a pre-existing vector.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != Self::TOTAL`, `L < 1`, or `R > 0`.
    #[must_use]
    pub fn from_vec(data: Vec<f32>) -> Self {
        assert!(L >= 1, "must have at least 1 layer");
        assert!(R == 0, "residual blocks not yet implemented");
        assert_eq!(
            data.len(),
            Self::TOTAL,
            "expected {} weights, got {}",
            Self::TOTAL,
            data.len()
        );
        Self { data }
    }

    // ===== Layer accessors =====

    /// First conv layer weights in im2col layout: `[INPUT_CHANNELS * K * K][C]`.
    #[must_use]
    pub fn first_conv_weights(&self) -> &[f32] {
        &self.data[Self::FIRST_WEIGHTS_START..][..Self::FIRST_CONV_WEIGHTS]
    }

    /// First conv layer biases: `[C]`.
    #[must_use]
    pub fn first_conv_bias(&self) -> &[f32] {
        &self.data[Self::FIRST_BIAS_START..][..Self::FIRST_CONV_BIAS]
    }

    /// Hidden conv layer weights in im2col layout: `[C * K * K][C]`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= L - 1`.
    #[must_use]
    pub fn hidden_conv_weights(&self, index: usize) -> &[f32] {
        assert!(index < L - 1, "hidden layer index out of bounds");
        let start = Self::HIDDEN_START + index * Self::HIDDEN_CONV_TOTAL;
        &self.data[start..][..Self::HIDDEN_CONV_WEIGHTS]
    }

    /// Hidden conv layer biases for layer `index` (0-indexed): `[C]`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= L - 1`.
    #[must_use]
    pub fn hidden_conv_bias(&self, index: usize) -> &[f32] {
        assert!(index < L - 1, "hidden layer index out of bounds");
        let start =
            Self::HIDDEN_START + index * Self::HIDDEN_CONV_TOTAL + Self::HIDDEN_CONV_WEIGHTS;
        &self.data[start..][..Self::HIDDEN_CONV_BIAS]
    }

    /// Final conv layer weights in im2col layout: `[C][1]` (1×1 kernel, C -> 1).
    #[must_use]
    pub fn final_conv_weights(&self) -> &[f32] {
        &self.data[Self::FINAL_START..][..Self::FINAL_CONV_WEIGHTS]
    }

    /// Final conv layer bias (scalar).
    #[must_use]
    pub fn final_conv_bias(&self) -> f32 {
        self.data[Self::FINAL_START + Self::FINAL_CONV_WEIGHTS]
    }
}

impl<const K: usize, const C: usize, const L: usize, const R: usize> AsRef<[f32]>
    for ConvWeights<K, C, L, R>
{
    fn as_ref(&self) -> &[f32] {
        &self.data
    }
}

impl<const K: usize, const C: usize, const L: usize, const R: usize> AsMut<[f32]>
    for ConvWeights<K, C, L, R>
{
    fn as_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }
}

impl<const K: usize, const C: usize, const L: usize, const R: usize> From<ConvWeights<K, C, L, R>>
    for Vec<f32>
{
    fn from(weights: ConvWeights<K, C, L, R>) -> Self {
        weights.data
    }
}

impl<const K: usize, const C: usize, const L: usize, const R: usize> From<Vec<f32>>
    for ConvWeights<K, C, L, R>
{
    /// Creates weights from a vector.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != Self::TOTAL`, `L < 1`, or `R > 0`.
    fn from(data: Vec<f32>) -> Self {
        Self::from_vec(data)
    }
}

impl<const K: usize, const C: usize, const L: usize, const R: usize> EvolvableGenes
    for ConvWeights<K, C, L, R>
{
    fn crossover(&self, other: &Self, crossover: Crossover, rng: &mut fastrand::Rng) -> Self {
        match crossover {
            Crossover::Uniform => {
                Self::from_vec(uniform_crossover(self.as_ref(), other.as_ref(), rng))
            }
        }
    }

    fn mutate(&self, mutation: Mutation, rng: &mut fastrand::Rng) -> Self {
        match mutation {
            Mutation::Gaussian { sigma } => {
                Self::from_vec(gaussian_mutate(self.as_ref(), sigma, rng))
            }
        }
    }
}

/// Computes He initialization standard deviation: `√(2/n_in)`.
fn he_std(n_in: usize) -> f32 {
    // f32 represents integers exactly up to 2^24 (~16M).
    // n_in is small for any reasonable architecture (e.g., 32 * 3 * 3 = 288).
    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    let n_in_f32 = n_in as f32;
    (2.0 / n_in_f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Type alias for testing: 3×3 kernel, 8 channels, 2 layers, 0 residual
    type TestWeights = ConvWeights<3, 8, 2, 0>;

    #[test]
    fn total_size_calculation() {
        // First: 2 * 8 * 3 * 3 = 144 weights + 8 bias = 152
        // Hidden: (2-1) * (8 * 8 * 3 * 3 + 8) = 1 * (576 + 8) = 584
        // Final: 8 weights + 1 bias = 9
        // Total: 152 + 584 + 9 = 745
        assert_eq!(TestWeights::TOTAL, 745);
    }

    #[test]
    fn total_size_single_layer() {
        // L=1 means no hidden layers
        type SingleLayer = ConvWeights<3, 8, 1, 0>;
        // First: 144 + 8 = 152
        // Hidden: 0
        // Final: 8 + 1 = 9
        // Total: 161
        assert_eq!(SingleLayer::TOTAL, 161);
    }

    #[test]
    fn total_size_conv_tiny() {
        // ConvTiny: K=3, C=32, L=2, R=0
        type ConvTiny = ConvWeights<3, 32, 2, 0>;
        // First: 2 * 32 * 9 = 576 + 32 = 608
        // Hidden: 1 * (32 * 32 * 9 + 32) = 9216 + 32 = 9248
        // Final: 32 + 1 = 33
        // Total: 608 + 9248 + 33 = 9889
        assert_eq!(ConvTiny::TOTAL, 9889);
    }

    #[test]
    fn random_creates_correct_length() {
        let mut rng = fastrand::Rng::with_seed(42);
        let weights = TestWeights::random(&mut rng);

        assert_eq!(weights.as_ref().len(), TestWeights::TOTAL);
    }

    #[test]
    fn from_vec_accepts_correct_length() {
        let data = vec![0.0f32; TestWeights::TOTAL];
        let weights = TestWeights::from_vec(data);

        assert_eq!(weights.as_ref().len(), TestWeights::TOTAL);
    }

    #[test]
    #[should_panic(expected = "expected 745 weights")]
    fn from_vec_rejects_wrong_length() {
        let data = vec![0.0f32; 100];
        let _ = TestWeights::from_vec(data);
    }

    #[test]
    #[should_panic(expected = "residual blocks not yet implemented")]
    fn panics_on_residual_blocks() {
        type WithResidual = ConvWeights<3, 8, 2, 1>;
        let mut rng = fastrand::Rng::with_seed(42);
        let _ = WithResidual::random(&mut rng);
    }

    #[test]
    fn slice_accessors_have_correct_lengths() {
        let mut rng = fastrand::Rng::with_seed(42);
        let weights = TestWeights::random(&mut rng);

        assert_eq!(weights.first_conv_weights().len(), 2 * 8 * 3 * 3);
        assert_eq!(weights.first_conv_bias().len(), 8);
        assert_eq!(weights.hidden_conv_weights(0).len(), 8 * 8 * 3 * 3);
        assert_eq!(weights.hidden_conv_bias(0).len(), 8);
        assert_eq!(weights.final_conv_weights().len(), 8);
    }

    #[test]
    fn slice_accessors_are_non_overlapping() {
        let data: Vec<f32> = (0..TestWeights::TOTAL as u32).map(|i| i as f32).collect();
        let weights = TestWeights::from_vec(data);

        // Check that each accessor returns sequential, non-overlapping data
        let first_w_end = weights.first_conv_weights().last().unwrap();
        let first_b_start = weights.first_conv_bias().first().unwrap();
        assert_eq!(*first_w_end + 1.0, *first_b_start);

        let first_b_end = weights.first_conv_bias().last().unwrap();
        let hidden_w_start = weights.hidden_conv_weights(0).first().unwrap();
        assert_eq!(*first_b_end + 1.0, *hidden_w_start);
    }

    #[test]
    fn he_initialization_has_reasonable_distribution() {
        let mut rng = fastrand::Rng::with_seed(42);
        let weights = TestWeights::random(&mut rng);

        // Check first conv weights have reasonable variance
        let first_weights = weights.first_conv_weights();
        let mean: f32 = first_weights.iter().sum::<f32>() / first_weights.len() as f32;
        let variance: f32 = first_weights
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>()
            / first_weights.len() as f32;
        let std_dev = variance.sqrt();

        // Expected std: sqrt(2 / (2 * 9)) = sqrt(1/9) ≈ 0.333
        let expected_std = he_std(2 * 3 * 3);
        assert!(
            (std_dev - expected_std).abs() < 0.1,
            "std_dev {std_dev} not close to expected {expected_std}"
        );
    }

    #[test]
    fn biases_initialized_to_zero() {
        let mut rng = fastrand::Rng::with_seed(42);
        let weights = TestWeights::random(&mut rng);

        assert!(weights.first_conv_bias().iter().all(|&b| b == 0.0));
        assert!(weights.hidden_conv_bias(0).iter().all(|&b| b == 0.0));
        assert_eq!(weights.final_conv_bias(), 0.0);
    }

    #[test]
    #[should_panic(expected = "hidden layer index out of bounds")]
    fn hidden_weights_panics_on_invalid_index() {
        let mut rng = fastrand::Rng::with_seed(42);
        let weights = TestWeights::random(&mut rng);
        // TestWeights has L=2, so only 1 hidden layer (index 0)
        let _ = weights.hidden_conv_weights(1);
    }

    #[test]
    fn into_vec_returns_data() {
        let original: Vec<f32> = (0..TestWeights::TOTAL).map(|i| i as f32).collect();
        let weights = TestWeights::from_vec(original.clone());
        let recovered: Vec<f32> = weights.into();

        assert_eq!(recovered, original);
    }

    #[test]
    fn as_mut_allows_modification() {
        let mut weights = TestWeights::from_vec(vec![0.0f32; TestWeights::TOTAL]);

        weights.as_mut()[0] = 42.0;

        assert_eq!(weights.as_ref()[0], 42.0);
    }

    #[test]
    fn crossover_with_uniform_produces_valid_child() {
        let mut rng = fastrand::Rng::with_seed(42);
        let parent1 = TestWeights::random(&mut rng);
        let parent2 = TestWeights::random(&mut rng);

        let child = parent1.crossover(&parent2, Crossover::Uniform, &mut rng);

        assert_eq!(child.as_ref().len(), TestWeights::TOTAL);
        for (i, &val) in child.as_ref().iter().enumerate() {
            assert!(val == parent1.as_ref()[i] || val == parent2.as_ref()[i]);
        }
    }

    #[test]
    fn mutate_with_gaussian_changes_values() {
        let weights = TestWeights::from_vec(vec![0.0f32; TestWeights::TOTAL]);
        let mut rng = fastrand::Rng::with_seed(42);

        let result = weights.mutate(Mutation::Gaussian { sigma: 1.0 }, &mut rng);

        assert_eq!(result.as_ref().len(), TestWeights::TOTAL);
        let changed = result.as_ref().iter().filter(|&&val| val != 0.0).count();
        assert!(changed > 0, "some values should change");
    }
}
