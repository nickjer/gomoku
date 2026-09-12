use serde::{Deserialize, Serialize};

use crate::evolution::crossover::Crossover;
use crate::evolution::genes::EvolvableGenes;
use crate::evolution::mutation::Mutation;
use crate::nn::STONE_CHANNELS;

use super::params::ClusterParams;

/// Weights for a cluster neural network with const-generic architecture.
///
/// # Type Parameters
/// - `CLUSTERS`: Number of clusters per channel for spatial layers (typically `CLUSTER_COUNT`)
/// - `CHANNELS`: Number of channels in middle layers
/// - `LAYERS`: Number of cluster layers (must be >= 1)
///
/// # Architecture
/// - First cluster: 2 input channels → CHANNELS channels, CLUSTERS clusters + `ReLU`
/// - LAYERS-1 middle clusters: CHANNELS → CHANNELS channels, CLUSTERS clusters + `ReLU`
/// - Final cluster: CHANNELS → 1 channel, CLUSTERS=1 (pointwise, no `ReLU`)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClusterWeights<const CLUSTERS: usize, const CHANNELS: usize, const LAYERS: usize> {
    board_layer: ClusterParams<{ STONE_CHANNELS }, CHANNELS, CLUSTERS>,
    middle_layers: Vec<ClusterParams<CHANNELS, CHANNELS, CLUSTERS>>,
    scoring_layer: ClusterParams<CHANNELS, 1, 1>,
}

impl<const CLUSTERS: usize, const CHANNELS: usize, const LAYERS: usize>
    ClusterWeights<CLUSTERS, CHANNELS, LAYERS>
{
    /// Creates weights initialized using He initialization.
    ///
    /// # Panics
    ///
    /// Panics if `LAYERS < 1`.
    #[must_use]
    pub fn random(rng: &mut fastrand::Rng) -> Self {
        assert!(LAYERS >= 1, "must have at least 1 layer");

        Self {
            board_layer: ClusterParams::random(rng),
            middle_layers: (0..(LAYERS - 1))
                .map(|_| ClusterParams::random(rng))
                .collect(),
            scoring_layer: ClusterParams::random(rng),
        }
    }

    /// Returns a reference to the board cluster layer parameters.
    #[must_use]
    pub fn board_layer(&self) -> &ClusterParams<{ STONE_CHANNELS }, CHANNELS, CLUSTERS> {
        &self.board_layer
    }

    /// Returns the middle cluster layer parameters.
    #[must_use]
    pub fn middle_layers(&self) -> &[ClusterParams<CHANNELS, CHANNELS, CLUSTERS>] {
        &self.middle_layers
    }

    /// Returns a reference to the final cluster layer parameters (pointwise, CLUSTERS=1).
    #[must_use]
    pub fn scoring_layer(&self) -> &ClusterParams<CHANNELS, 1, 1> {
        &self.scoring_layer
    }
}

impl<const CLUSTERS: usize, const CHANNELS: usize, const LAYERS: usize> std::fmt::Display
    for ClusterWeights<CLUSTERS, CHANNELS, LAYERS>
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            formatter,
            "ClusterWeights ({CLUSTERS} clusters, {CHANNELS} channels, {LAYERS} layers)"
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

impl<const CLUSTERS: usize, const CHANNELS: usize, const LAYERS: usize> EvolvableGenes
    for ClusterWeights<CLUSTERS, CHANNELS, LAYERS>
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
    use crate::nn::CLUSTER_COUNT;

    // Type alias for testing: 9 clusters, 8 channels, 2 layers
    type TestWeights = ClusterWeights<CLUSTER_COUNT, 8, 2>;

    #[test]
    fn random_creates_correct_structure() {
        let mut rng = fastrand::Rng::with_seed(42);
        let weights = TestWeights::random(&mut rng);

        // First: IN_CHANNELS=2, OUT_CHANNELS=8, CLUSTERS=9
        assert_eq!(weights.board_layer().weights().len(), 2 * 9 * 8);
        assert_eq!(weights.board_layer().bias().len(), 8);

        // Middle: LAYERS-1 = 1 layer, IN_CHANNELS=8, OUT_CHANNELS=8, CLUSTERS=9
        assert_eq!(weights.middle_layers().len(), 1);
        assert_eq!(weights.middle_layers()[0].weights().len(), 8 * 9 * 8);
        assert_eq!(weights.middle_layers()[0].bias().len(), 8);

        // Last: IN_CHANNELS=8, OUT_CHANNELS=1, CLUSTERS=1 (pointwise)
        assert_eq!(weights.scoring_layer().weights().len(), 8);
        assert_eq!(weights.scoring_layer().bias().len(), 1);
    }

    #[test]
    fn random_single_layer() {
        type SingleLayer = ClusterWeights<CLUSTER_COUNT, 8, 1>;
        let mut rng = fastrand::Rng::with_seed(42);
        let weights = SingleLayer::random(&mut rng);

        assert!(weights.middle_layers().is_empty());
    }

    #[test]
    fn biases_initialized_to_zero() {
        let mut rng = fastrand::Rng::with_seed(42);
        let weights = TestWeights::random(&mut rng);

        assert!(weights.board_layer().bias().iter().all(|&bias| bias == 0.0));
        assert!(
            weights.middle_layers()[0]
                .bias()
                .iter()
                .all(|&bias| bias == 0.0)
        );
        assert!(
            weights
                .scoring_layer()
                .bias()
                .iter()
                .all(|&bias| bias == 0.0)
        );
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
        for (idx, &val) in child.board_layer().weights().iter().enumerate() {
            assert!(
                val == parent1.board_layer().weights()[idx]
                    || val == parent2.board_layer().weights()[idx]
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
