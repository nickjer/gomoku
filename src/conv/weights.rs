use serde::{Deserialize, Serialize};

use crate::evolution::crossover::Crossover;
use crate::evolution::genes::EvolvableGenes;
use crate::evolution::mutation::Mutation;

use crate::nn::STONE_CHANNELS;

use super::params::ConvParams;

/// Weights for a convolutional neural network with const-generic architecture.
///
/// # Type Parameters
/// - `SIDE`: Kernel size (e.g., 3 for 3×3 kernels)
/// - `CHANNELS`: Number of channels in middle layers
/// - `LAYERS`: Number of convolutional layers (must be >= 1)
/// - `RESIDUAL_BLOCKS`: Number of residual blocks (must be 0 until implemented)
///
/// # Architecture
/// - First conv: 2 input channels → CHANNELS channels, SIDE×SIDE kernel
/// - LAYERS-1 middle convs: CHANNELS → CHANNELS channels, SIDE×SIDE kernel
/// - Final conv: CHANNELS → 1 channel, 1×1 kernel (position scoring)
///
/// # Weight Layout
/// Weights are stored in `[IN_CHANNELS * SIDE * SIDE][OUT_CHANNELS]` layout (transposed from the standard
/// `[OUT_CHANNELS][IN_CHANNELS * SIDE * SIDE]`). This allows efficient dot products against the workspace buffer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConvWeights<
    const SIDE: usize,
    const CHANNELS: usize,
    const LAYERS: usize,
    const RESIDUAL_BLOCKS: usize,
> {
    board_layer: ConvParams<{ STONE_CHANNELS }, CHANNELS, SIDE>,
    middle_layers: Vec<ConvParams<CHANNELS, CHANNELS, SIDE>>,
    scoring_layer: ConvParams<CHANNELS, 1, 1>,
}

impl<const SIDE: usize, const CHANNELS: usize, const LAYERS: usize, const RESIDUAL_BLOCKS: usize>
    ConvWeights<SIDE, CHANNELS, LAYERS, RESIDUAL_BLOCKS>
{
    /// Creates weights initialized using He initialization.
    ///
    /// Uses `N(0, √(2/n_in))` for each layer, which is optimal for `ReLU` networks.
    ///
    /// # Panics
    ///
    /// Panics if `LAYERS < 1` or `RESIDUAL_BLOCKS > 0` (residual blocks not yet implemented).
    #[must_use]
    pub fn random(rng: &mut fastrand::Rng) -> Self {
        assert!(LAYERS >= 1, "must have at least 1 layer");
        assert!(RESIDUAL_BLOCKS == 0, "residual blocks not yet implemented");

        Self {
            board_layer: ConvParams::random(rng),
            middle_layers: (0..(LAYERS - 1)).map(|_| ConvParams::random(rng)).collect(),
            scoring_layer: ConvParams::random(rng),
        }
    }

    /// Returns a reference to the board conv layer parameters.
    #[must_use]
    pub fn board_layer(&self) -> &ConvParams<{ STONE_CHANNELS }, CHANNELS, SIDE> {
        &self.board_layer
    }

    /// Returns the middle conv layer parameters.
    #[must_use]
    pub fn middle_layers(&self) -> &[ConvParams<CHANNELS, CHANNELS, SIDE>] {
        &self.middle_layers
    }

    /// Returns a reference to the final conv layer parameters.
    #[must_use]
    pub fn scoring_layer(&self) -> &ConvParams<CHANNELS, 1, 1> {
        &self.scoring_layer
    }
}

impl<const SIDE: usize, const CHANNELS: usize, const LAYERS: usize, const RESIDUAL_BLOCKS: usize>
    std::fmt::Display for ConvWeights<SIDE, CHANNELS, LAYERS, RESIDUAL_BLOCKS>
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            formatter,
            "ConvWeights ({SIDE}x{SIDE} kernel, {CHANNELS} channels, {LAYERS} layers)"
        )?;
        writeln!(formatter)?;
        writeln!(formatter, "Layer 1 (board): {}", self.board_layer)?;
        for (idx, layer) in self.middle_layers.iter().enumerate() {
            writeln!(formatter, "Layer {} (middle): {layer}", idx + 2)?;
        }
        let last_idx = 2 + self.middle_layers.len();
        writeln!(
            formatter,
            "Layer {last_idx} (scoring): {}",
            self.scoring_layer
        )?;

        let total: usize = self.board_layer.weights().len()
            + self.board_layer.bias().len()
            + self
                .middle_layers
                .iter()
                .map(|layer| layer.weights().len() + layer.bias().len())
                .sum::<usize>()
            + self.scoring_layer.weights().len()
            + self.scoring_layer.bias().len();
        write!(formatter, "Total parameters: {total}")
    }
}

impl<const SIDE: usize, const CHANNELS: usize, const LAYERS: usize, const RESIDUAL_BLOCKS: usize>
    EvolvableGenes for ConvWeights<SIDE, CHANNELS, LAYERS, RESIDUAL_BLOCKS>
{
    fn crossover(&self, other: &Self, crossover: Crossover, rng: &mut fastrand::Rng) -> Self {
        match crossover {
            Crossover::Uniform => Self {
                board_layer: self.board_layer.uniform_crossover(&other.board_layer, rng),
                middle_layers: self
                    .middle_layers
                    .iter()
                    .zip(&other.middle_layers)
                    .map(|(a, b)| a.uniform_crossover(b, rng))
                    .collect(),
                scoring_layer: self
                    .scoring_layer
                    .uniform_crossover(&other.scoring_layer, rng),
            },
        }
    }

    fn mutate(&self, mutation: Mutation, rng: &mut fastrand::Rng) -> Self {
        match mutation {
            Mutation::Gaussian { sigma } => Self {
                board_layer: self.board_layer.gaussian_mutate(sigma, rng),
                middle_layers: self
                    .middle_layers
                    .iter()
                    .map(|layer| layer.gaussian_mutate(sigma, rng))
                    .collect(),
                scoring_layer: self.scoring_layer.gaussian_mutate(sigma, rng),
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

        // First: IN_CHANNELS=2, OUT_CHANNELS=8, SIDE=3
        assert_eq!(weights.board_layer().weights().len(), 2 * 8 * 3 * 3);
        assert_eq!(weights.board_layer().bias().len(), 8);

        // Middle: LAYERS-1 = 1 layer, IN_CHANNELS=8, OUT_CHANNELS=8, SIDE=3
        assert_eq!(weights.middle_layers().len(), 1);
        assert_eq!(weights.middle_layers()[0].weights().len(), 8 * 8 * 3 * 3);
        assert_eq!(weights.middle_layers()[0].bias().len(), 8);

        // Last: IN_CHANNELS=8, OUT_CHANNELS=1, SIDE=1
        assert_eq!(weights.scoring_layer().weights().len(), 8);
        assert_eq!(weights.scoring_layer().bias().len(), 1);
    }

    #[test]
    fn random_single_layer() {
        type SingleLayer = ConvWeights<3, 8, 1, 0>;
        let mut rng = fastrand::Rng::with_seed(42);
        let weights = SingleLayer::random(&mut rng);

        assert!(weights.middle_layers().is_empty());
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

        assert!(weights.board_layer().bias().iter().all(|&b| b == 0.0));
        assert!(weights.middle_layers()[0].bias().iter().all(|&b| b == 0.0));
        assert!(weights.scoring_layer().bias().iter().all(|&b| b == 0.0));
    }

    #[test]
    fn crossover_with_uniform_produces_valid_child() {
        let mut rng = fastrand::Rng::with_seed(42);
        let parent1 = TestWeights::random(&mut rng);
        let parent2 = TestWeights::random(&mut rng);

        let child = parent1.crossover(&parent2, Crossover::Uniform, &mut rng);

        // Child should have same structure
        assert_eq!(child.middle_layers().len(), parent1.middle_layers().len());

        // Each weight comes from one parent
        for (i, &val) in child.board_layer().weights().iter().enumerate() {
            assert!(
                val == parent1.board_layer().weights()[i]
                    || val == parent2.board_layer().weights()[i]
            );
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
