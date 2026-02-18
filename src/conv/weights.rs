use serde::{Deserialize, Serialize};

use crate::evolution::crossover::Crossover;
use crate::evolution::genes::EvolvableGenes;
use crate::evolution::mutation::Mutation;

use crate::nn::INPUT_CHANNELS;

use super::params::ConvParams;

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
/// Weights are stored in `[IN_C * K * K][OUT_C]` layout (transposed from the standard
/// `[OUT_C][IN_C * K * K]`). This allows efficient dot products against the workspace buffer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConvWeights<const K: usize, const C: usize, const L: usize, const R: usize> {
    first: ConvParams<{ INPUT_CHANNELS }, C, K>,
    hidden: Vec<ConvParams<C, C, K>>,
    last: ConvParams<C, 1, 1>,
}

impl<const K: usize, const C: usize, const L: usize, const R: usize> ConvWeights<K, C, L, R> {
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

        Self {
            first: ConvParams::random(rng),
            hidden: (0..(L - 1)).map(|_| ConvParams::random(rng)).collect(),
            last: ConvParams::random(rng),
        }
    }

    /// Returns a reference to the first conv layer parameters.
    #[must_use]
    pub fn first(&self) -> &ConvParams<{ INPUT_CHANNELS }, C, K> {
        &self.first
    }

    /// Returns the hidden conv layer parameters.
    #[must_use]
    pub fn hidden(&self) -> &[ConvParams<C, C, K>] {
        &self.hidden
    }

    /// Returns a reference to the final conv layer parameters.
    #[must_use]
    pub fn last(&self) -> &ConvParams<C, 1, 1> {
        &self.last
    }
}

impl<const K: usize, const C: usize, const L: usize, const R: usize> std::fmt::Display
    for ConvWeights<K, C, L, R>
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            formatter,
            "ConvWeights ({K}x{K} kernel, {C} channels, {L} layers)"
        )?;
        writeln!(formatter)?;
        writeln!(formatter, "Layer 1 (first): {}", self.first)?;
        for (idx, layer) in self.hidden.iter().enumerate() {
            writeln!(formatter, "Layer {} (hidden): {layer}", idx + 2)?;
        }
        let last_idx = 2 + self.hidden.len();
        writeln!(formatter, "Layer {last_idx} (last): {}", self.last)?;

        let total: usize = self.first.weights().len()
            + self.first.bias().len()
            + self
                .hidden
                .iter()
                .map(|layer| layer.weights().len() + layer.bias().len())
                .sum::<usize>()
            + self.last.weights().len()
            + self.last.bias().len();
        write!(formatter, "Total parameters: {total}")
    }
}

impl<const K: usize, const C: usize, const L: usize, const R: usize> EvolvableGenes
    for ConvWeights<K, C, L, R>
{
    fn crossover(&self, other: &Self, crossover: Crossover, rng: &mut fastrand::Rng) -> Self {
        match crossover {
            Crossover::Uniform => Self {
                first: self.first.uniform_crossover(&other.first, rng),
                hidden: self
                    .hidden
                    .iter()
                    .zip(&other.hidden)
                    .map(|(a, b)| a.uniform_crossover(b, rng))
                    .collect(),
                last: self.last.uniform_crossover(&other.last, rng),
            },
        }
    }

    fn mutate(&self, mutation: Mutation, rng: &mut fastrand::Rng) -> Self {
        match mutation {
            Mutation::Gaussian { sigma } => Self {
                first: self.first.gaussian_mutate(sigma, rng),
                hidden: self
                    .hidden
                    .iter()
                    .map(|layer| layer.gaussian_mutate(sigma, rng))
                    .collect(),
                last: self.last.gaussian_mutate(sigma, rng),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Type alias for testing: 3×3 kernel, 8 channels, 2 layers, 0 residual
    type TestWeights = ConvWeights<3, 8, 2, 0>;

    #[test]
    fn random_creates_correct_structure() {
        let mut rng = fastrand::Rng::with_seed(42);
        let weights = TestWeights::random(&mut rng);

        // First: IN_C=2, OUT_C=8, K=3
        assert_eq!(weights.first().weights().len(), 2 * 8 * 3 * 3);
        assert_eq!(weights.first().bias().len(), 8);

        // Hidden: L-1 = 1 layer, IN_C=8, OUT_C=8, K=3
        assert_eq!(weights.hidden().len(), 1);
        assert_eq!(weights.hidden()[0].weights().len(), 8 * 8 * 3 * 3);
        assert_eq!(weights.hidden()[0].bias().len(), 8);

        // Last: IN_C=8, OUT_C=1, K=1
        assert_eq!(weights.last().weights().len(), 8);
        assert_eq!(weights.last().bias().len(), 1);
    }

    #[test]
    fn random_single_layer() {
        type SingleLayer = ConvWeights<3, 8, 1, 0>;
        let mut rng = fastrand::Rng::with_seed(42);
        let weights = SingleLayer::random(&mut rng);

        assert!(weights.hidden().is_empty());
    }

    #[test]
    #[should_panic(expected = "residual blocks not yet implemented")]
    fn panics_on_residual_blocks() {
        type WithResidual = ConvWeights<3, 8, 2, 1>;
        let mut rng = fastrand::Rng::with_seed(42);
        let _ = WithResidual::random(&mut rng);
    }

    #[test]
    fn biases_initialized_to_zero() {
        let mut rng = fastrand::Rng::with_seed(42);
        let weights = TestWeights::random(&mut rng);

        assert!(weights.first().bias().iter().all(|&b| b == 0.0));
        assert!(weights.hidden()[0].bias().iter().all(|&b| b == 0.0));
        assert!(weights.last().bias().iter().all(|&b| b == 0.0));
    }

    #[test]
    fn crossover_with_uniform_produces_valid_child() {
        let mut rng = fastrand::Rng::with_seed(42);
        let parent1 = TestWeights::random(&mut rng);
        let parent2 = TestWeights::random(&mut rng);

        let child = parent1.crossover(&parent2, Crossover::Uniform, &mut rng);

        // Child should have same structure
        assert_eq!(child.hidden().len(), parent1.hidden().len());

        // Each weight comes from one parent
        for (i, &val) in child.first().weights().iter().enumerate() {
            assert!(val == parent1.first().weights()[i] || val == parent2.first().weights()[i]);
        }
    }

    #[test]
    fn mutate_with_gaussian_changes_values() {
        let mut rng = fastrand::Rng::with_seed(42);
        let weights = TestWeights::random(&mut rng);

        let mutated = weights.mutate(Mutation::Gaussian { sigma: 1.0 }, &mut rng);

        assert_ne!(weights, mutated);
    }

    #[test]
    fn clone_produces_equal_weights() {
        let mut rng = fastrand::Rng::with_seed(42);
        let weights = TestWeights::random(&mut rng);
        let cloned = weights.clone();

        assert_eq!(weights, cloned);
    }
}
