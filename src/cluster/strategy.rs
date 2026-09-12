use tracing::instrument;

use crate::board::Board;
use crate::nn::{
    STONE_CHANNELS, board_to_stone_channels, highest_scored_empty_position, zero_negatives_inplace,
};
use crate::position_id::PositionId;
use crate::position_map::PositionMap;
use crate::stone::Stone;
use crate::strategy::{EvolvableStrategy, Strategy};

use super::weights::ClusterWeights;
use crate::nn::CLUSTER_COUNT;

/// A cluster strategy for Gomoku using sums over D8-equivalent neighbor clusters.
///
/// # Type Parameters
/// - `CHANNELS`: Number of channels in middle layers
/// - `LAYERS`: Number of cluster layers (must be >= 1)
#[derive(Debug, Clone)]
pub struct ClusterStrategy<const CHANNELS: usize, const LAYERS: usize> {
    label: String,
    weights: ClusterWeights<{ CLUSTER_COUNT }, CHANNELS, LAYERS>,
}

/// A tiny cluster strategy: 9 clusters, 32 channels, 2 layers.
/// ~10K parameters.
pub type ClusterTiny = ClusterStrategy<32, 2>;

/// A small cluster strategy: 9 clusters, 64 channels, 4 layers.
/// ~112K parameters.
pub type ClusterSmall = ClusterStrategy<64, 4>;

impl<const CHANNELS: usize, const LAYERS: usize> ClusterStrategy<CHANNELS, LAYERS> {
    /// Forward pass through the network.
    ///
    /// Input: board encoding with `STONE_CHANNELS` channels per position.
    /// Output: one score per position.
    fn score_positions(&self, input: &PositionMap<f32, STONE_CHANNELS>) -> PositionMap<f32, 1> {
        // First cluster: STONE_CHANNELS -> CHANNELS channels
        let mut activations = self.weights.board_layer().cluster2d(input);
        zero_negatives_inplace(activations.as_flattened_mut());

        // Middle clusters: CHANNELS -> CHANNELS channels
        for middle in self.weights.middle_layers() {
            activations = middle.cluster2d(&activations);
            zero_negatives_inplace(activations.as_flattened_mut());
        }

        // Final cluster: CHANNELS -> 1 channel (CLUSTERS=1 pointwise, no ReLU)
        self.weights.scoring_layer().pointwise2d(&activations)
    }
}

impl<const CHANNELS: usize, const LAYERS: usize> Strategy for ClusterStrategy<CHANNELS, LAYERS> {
    /// Selects a move using cluster sums. No D8 augmentation needed —
    /// the cluster sums are inherently D8-equivariant.
    #[instrument(level = "trace", skip_all)]
    fn choose_move(
        &self,
        current_stone: Stone,
        board: &Board,
        rng: &mut fastrand::Rng,
    ) -> PositionId {
        let encoding = board_to_stone_channels(board, current_stone);
        let scores = self.score_positions(&encoding);
        let empty = board.empty_position_ids();
        highest_scored_empty_position(&empty, &scores, rng)
    }

    fn label(&self) -> &str {
        &self.label
    }
}

impl<const CHANNELS: usize, const LAYERS: usize> EvolvableStrategy
    for ClusterStrategy<CHANNELS, LAYERS>
{
    type Genes = ClusterWeights<{ CLUSTER_COUNT }, CHANNELS, LAYERS>;

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

        let output = strategy.score_positions(&input);

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
        // LAYERS=2 adds one middle CHANNELS -> CHANNELS layer, exercising the middle-layer loop.
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = ClusterStrategy::<4, 2>::random("two_layers", &mut rng);
        let input = board_to_stone_channels(&Board::new(), Stone::Black);

        let output = strategy.score_positions(&input);

        for pos in PositionId::iter() {
            let [score] = *output.get(pos);
            assert!(score.is_finite(), "score at {pos:?} is {score}");
        }
    }
}
