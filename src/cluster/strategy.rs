use tracing::instrument;

use crate::board::Board;
use crate::nn::{INPUT_CHANNELS, encode_board, relu_inplace, select_best_position};
use crate::position_id::PositionId;
use crate::position_map::PositionMap;
use crate::stone::Stone;
use crate::strategy::{EvolvableStrategy, Strategy};

use super::features::FEATURE_COUNT;
use super::weights::ClusterWeights;

/// A cluster strategy for Gomoku using D8-equivariant polynomial features.
///
/// # Type Parameters
/// - `C`: Number of channels in hidden layers
/// - `L`: Number of cluster layers (must be >= 1)
#[derive(Debug, Clone)]
pub struct ClusterStrategy<const C: usize, const L: usize> {
    label: String,
    weights: ClusterWeights<{ FEATURE_COUNT }, C, L>,
}

/// A tiny cluster strategy: 9 features, 32 channels, 2 layers.
/// ~10K parameters.
pub type ClusterTiny = ClusterStrategy<32, 2>;

/// A small cluster strategy: 9 features, 64 channels, 4 layers.
/// ~112K parameters.
pub type ClusterSmall = ClusterStrategy<64, 4>;

impl<const C: usize, const L: usize> ClusterStrategy<C, L> {
    /// Forward pass through the network.
    ///
    /// Input: board encoding with `INPUT_CHANNELS` channels per position.
    /// Output: one policy score per position.
    fn forward(&self, input: &PositionMap<f32, INPUT_CHANNELS>) -> PositionMap<f32, 1> {
        // First cluster: INPUT_CHANNELS -> C channels
        let mut activations = self.weights.first().cluster2d(input);
        relu_inplace(activations.as_flattened_mut());

        // Hidden clusters: C -> C channels
        for hidden in self.weights.hidden() {
            activations = hidden.cluster2d(&activations);
            relu_inplace(activations.as_flattened_mut());
        }

        // Final cluster: C -> 1 channel (F=1 pointwise, no ReLU)
        self.weights.last().pointwise2d(&activations)
    }
}

impl<const C: usize, const L: usize> Strategy for ClusterStrategy<C, L> {
    /// Selects a move using cluster features. No D8 augmentation needed —
    /// the features are inherently D8-equivariant.
    #[instrument(level = "trace", skip_all)]
    fn choose_move(
        &self,
        current_stone: Stone,
        board: &Board,
        rng: &mut fastrand::Rng,
    ) -> PositionId {
        let encoding = encode_board(board, current_stone);
        let policy = self.forward(&encoding);
        let empty = board.empty_position_ids();
        select_best_position(&empty, &policy, rng)
    }

    fn label(&self) -> &str {
        &self.label
    }
}

impl<const C: usize, const L: usize> EvolvableStrategy for ClusterStrategy<C, L> {
    type Genes = ClusterWeights<{ FEATURE_COUNT }, C, L>;

    fn random(label: impl Into<String>, rng: &mut fastrand::Rng) -> Self {
        Self {
            label: label.into(),
            weights: ClusterWeights::random(rng),
        }
    }

    fn genes(&self) -> &Self::Genes {
        &self.weights
    }

    fn from_genes(label: impl Into<String>, genes: Self::Genes) -> Self {
        Self {
            label: label.into(),
            weights: genes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Use smaller config for faster tests: 4 channels, 1 layer
    type TestStrategy = ClusterStrategy<4, 1>;

    #[test]
    fn forward_produces_finite_score_for_every_position() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = TestStrategy::random("test", &mut rng);
        let input = PositionMap::new(0.0);

        let output = strategy.forward(&input);

        for pos in PositionId::iter() {
            let [score] = *output.get(pos);
            assert!(score.is_finite(), "score at {pos:?} is {score}");
        }
    }

    #[test]
    fn choose_move_returns_empty_position() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = TestStrategy::random("test", &mut rng);
        let board = Board::new();

        let chosen = strategy.choose_move(Stone::Black, &board, &mut rng);

        assert!(board.empty_position_ids().contains(&chosen));
    }

    #[test]
    fn choose_move_deterministic_with_same_seed() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = TestStrategy::random("test", &mut rng);
        let board = Board::new();

        let mut rng1 = fastrand::Rng::with_seed(123);
        let mut rng2 = fastrand::Rng::with_seed(123);

        let move1 = strategy.choose_move(Stone::Black, &board, &mut rng1);
        let move2 = strategy.choose_move(Stone::Black, &board, &mut rng2);

        assert_eq!(move1, move2);
    }

    #[test]
    fn random_creates_strategy_with_label() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = TestStrategy::random("my_label", &mut rng);

        assert_eq!(strategy.label(), "my_label");
    }

    #[test]
    fn from_genes_creates_strategy_with_given_weights() {
        let mut rng = fastrand::Rng::with_seed(42);
        let original = TestStrategy::random("original", &mut rng);
        let genes = original.genes().clone();

        let reconstructed = TestStrategy::from_genes("reconstructed", genes);

        assert_eq!(reconstructed.label(), "reconstructed");
        assert_eq!(reconstructed.genes(), original.genes());
    }

    #[test]
    fn forward_with_hidden_layer_produces_finite_score_for_every_position() {
        // L=2 adds one hidden C -> C layer, exercising the hidden-layer loop.
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = ClusterStrategy::<4, 2>::random("two_layers", &mut rng);
        let input = encode_board(&Board::new(), Stone::Black);

        let output = strategy.forward(&input);

        for pos in PositionId::iter() {
            let [score] = *output.get(pos);
            assert!(score.is_finite(), "score at {pos:?} is {score}");
        }
    }
}
