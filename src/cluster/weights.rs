use serde::{Deserialize, Serialize};

use crate::evolution::crossover::Crossover;
use crate::evolution::genes::EvolvableGenes;
use crate::evolution::mutation::Mutation;
use crate::nn::INPUT_CHANNELS;

use super::params::ClusterParams;

/// Weights for a cluster neural network with const-generic architecture.
///
/// # Type Parameters
/// - `F`: Number of features per channel for spatial layers (typically `FEATURE_COUNT`)
/// - `C`: Number of channels in hidden layers
/// - `L`: Number of cluster layers (must be >= 1)
///
/// # Architecture
/// - First cluster: 2 input channels → C channels, F features + `ReLU`
/// - L-1 hidden clusters: C → C channels, F features + `ReLU`
/// - Final cluster: C → 1 channel, F=1 (pointwise, no `ReLU`)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClusterWeights<const F: usize, const C: usize, const L: usize> {
    first: ClusterParams<{ INPUT_CHANNELS }, C, F>,
    hidden: Vec<ClusterParams<C, C, F>>,
    last: ClusterParams<C, 1, 1>,
}

impl<const F: usize, const C: usize, const L: usize> ClusterWeights<F, C, L> {
    /// Creates weights initialized using He initialization.
    ///
    /// # Panics
    ///
    /// Panics if `L < 1`.
    #[must_use]
    pub fn random(rng: &mut fastrand::Rng) -> Self {
        assert!(L >= 1, "must have at least 1 layer");

        Self {
            first: ClusterParams::random(rng),
            hidden: (0..(L - 1)).map(|_| ClusterParams::random(rng)).collect(),
            last: ClusterParams::random(rng),
        }
    }

    /// Returns a reference to the first cluster layer parameters.
    #[must_use]
    pub fn first(&self) -> &ClusterParams<{ INPUT_CHANNELS }, C, F> {
        &self.first
    }

    /// Returns the hidden cluster layer parameters.
    #[must_use]
    pub fn hidden(&self) -> &[ClusterParams<C, C, F>] {
        &self.hidden
    }

    /// Returns a reference to the final cluster layer parameters (pointwise, F=1).
    #[must_use]
    pub fn last(&self) -> &ClusterParams<C, 1, 1> {
        &self.last
    }
}

impl<const F: usize, const C: usize, const L: usize> EvolvableGenes for ClusterWeights<F, C, L> {
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
    use crate::cluster::features::FEATURE_COUNT;

    // Type alias for testing: 9 features, 8 channels, 2 layers
    type TestWeights = ClusterWeights<FEATURE_COUNT, 8, 2>;

    #[test]
    fn random_creates_correct_structure() {
        let mut rng = fastrand::Rng::with_seed(42);
        let weights = TestWeights::random(&mut rng);

        // First: IN_C=2, OUT_C=8, F=9
        assert_eq!(weights.first().weights().len(), 2 * 9 * 8);
        assert_eq!(weights.first().bias().len(), 8);

        // Hidden: L-1 = 1 layer, IN_C=8, OUT_C=8, F=9
        assert_eq!(weights.hidden().len(), 1);
        assert_eq!(weights.hidden()[0].weights().len(), 8 * 9 * 8);
        assert_eq!(weights.hidden()[0].bias().len(), 8);

        // Last: IN_C=8, OUT_C=1, F=1 (pointwise)
        assert_eq!(weights.last().weights().len(), 8);
        assert_eq!(weights.last().bias().len(), 1);
    }

    #[test]
    fn random_single_layer() {
        type SingleLayer = ClusterWeights<FEATURE_COUNT, 8, 1>;
        let mut rng = fastrand::Rng::with_seed(42);
        let weights = SingleLayer::random(&mut rng);

        assert!(weights.hidden().is_empty());
    }

    #[test]
    fn biases_initialized_to_zero() {
        let mut rng = fastrand::Rng::with_seed(42);
        let weights = TestWeights::random(&mut rng);

        assert!(weights.first().bias().iter().all(|&bias| bias == 0.0));
        assert!(weights.hidden()[0].bias().iter().all(|&bias| bias == 0.0));
        assert!(weights.last().bias().iter().all(|&bias| bias == 0.0));
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
        for (idx, &val) in child.first().weights().iter().enumerate() {
            assert!(val == parent1.first().weights()[idx] || val == parent2.first().weights()[idx]);
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
