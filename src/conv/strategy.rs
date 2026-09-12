use tracing::instrument;

use crate::board::Board;
use crate::position_id::PositionId;
use crate::position_map::PositionMap;
use crate::stone::Stone;
use crate::strategy::{EvolvableStrategy, Strategy};

use crate::nn::{
    STONE_CHANNELS, board_to_stone_channels, highest_scored_empty_position, zero_negatives_inplace,
};

use super::weights::ConvWeights;
use crate::nn::BoardSymmetry;

/// A convolutional neural network strategy for Gomoku.
///
/// # Type Parameters
/// - `SIDE`: Kernel size (e.g., 3 for 3×3 kernels)
/// - `CHANNELS`: Number of channels in middle layers
/// - `LAYERS`: Number of convolutional layers (must be >= 1)
/// - `RESIDUAL_BLOCKS`: Number of residual blocks (must be 0 until implemented)
#[derive(Debug, Clone)]
pub struct ConvStrategy<
    const SIDE: usize,
    const CHANNELS: usize,
    const LAYERS: usize,
    const RESIDUAL_BLOCKS: usize,
> {
    label: String,
    weights: ConvWeights<SIDE, CHANNELS, LAYERS, RESIDUAL_BLOCKS>,
}

/// A tiny convolutional strategy: 3×3 kernels, 32 channels, 2 layers, no residual blocks.
/// ~10K parameters.
pub type ConvTiny = ConvStrategy<3, 32, 2, 0>;

/// A small convolutional strategy: 3×3 kernels, 64 channels, 4 layers, no residual blocks.
/// ~112K parameters.
pub type ConvSmall = ConvStrategy<3, 64, 4, 0>;

impl<const SIDE: usize, const CHANNELS: usize, const LAYERS: usize, const RESIDUAL_BLOCKS: usize>
    ConvStrategy<SIDE, CHANNELS, LAYERS, RESIDUAL_BLOCKS>
{
    /// Forward pass through the network.
    ///
    /// Input: board encoding with `STONE_CHANNELS` channels per position.
    /// Output: one score per position.
    fn score_positions(&self, input: &PositionMap<f32, STONE_CHANNELS>) -> PositionMap<f32, 1> {
        // First conv: STONE_CHANNELS -> CHANNELS channels
        let mut activations = self.weights.board_layer().conv2d(input);
        zero_negatives_inplace(activations.as_flattened_mut());

        // Middle convs: CHANNELS -> CHANNELS channels
        for middle in self.weights.middle_layers() {
            activations = middle.conv2d(&activations);
            zero_negatives_inplace(activations.as_flattened_mut());
        }

        // Final conv: CHANNELS -> 1 channel (1×1 kernel, no ReLU)
        self.weights.scoring_layer().pointwise2d(&activations)
    }
}

impl<const SIDE: usize, const CHANNELS: usize, const LAYERS: usize, const RESIDUAL_BLOCKS: usize>
    Strategy for ConvStrategy<SIDE, CHANNELS, LAYERS, RESIDUAL_BLOCKS>
{
    #[instrument(level = "trace", skip_all)]
    fn choose_move(
        &self,
        current_stone: Stone,
        board: &Board,
        rng: &mut fastrand::Rng,
    ) -> PositionId {
        let encoding = board_to_stone_channels(board, current_stone);

        // Random D8 symmetry for data augmentation (AlphaGo Zero style)
        let symmetry = BoardSymmetry::random(rng);

        // Transform input encoding
        let transformed_encoding = apply_board_symmetry(&encoding, |pos| symmetry.apply(pos));

        // Forward pass
        let scores = self.score_positions(&transformed_encoding);

        // Transform empty positions to transformed space
        let transformed_empty: Vec<PositionId> = board
            .empty_position_ids()
            .iter()
            .map(|&pos| symmetry.apply(pos))
            .collect();

        // Select best position in transformed space
        let transformed_pos = highest_scored_empty_position(&transformed_empty, &scores, rng);

        // Map selected position back to original orientation
        symmetry.apply_inverse(transformed_pos)
    }

    fn label(&self) -> &str {
        &self.label
    }
}

impl<const SIDE: usize, const CHANNELS: usize, const LAYERS: usize, const RESIDUAL_BLOCKS: usize>
    EvolvableStrategy for ConvStrategy<SIDE, CHANNELS, LAYERS, RESIDUAL_BLOCKS>
{
    type Genes = ConvWeights<SIDE, CHANNELS, LAYERS, RESIDUAL_BLOCKS>;

    fn random(label: impl Into<String>, rng: &mut fastrand::Rng) -> Self {
        Self {
            label: label.into(),
            weights: ConvWeights::random(rng),
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

/// Transforms an encoding by applying a position transformation.
///
/// For each position `p`, copies all channels from `encoding[p]` to `result[f(p)]`.
fn apply_board_symmetry<const CHANNELS: usize>(
    encoding: &PositionMap<f32, CHANNELS>,
    f: impl Fn(PositionId) -> PositionId,
) -> PositionMap<f32, CHANNELS> {
    let mut result = PositionMap::new(0.0);
    for pos in PositionId::iter() {
        *result.get_mut(f(pos)) = *encoding.get(pos);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    // Use smaller config for faster tests: 3×3 kernel, 4 channels, 1 layer
    type TestStrategy = ConvStrategy<3, 4, 1, 0>;

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
    fn d8_symmetry_produces_valid_move_on_empty_board() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = TestStrategy::random("test", &mut rng);
        let board = Board::new();

        let mut rng1 = fastrand::Rng::with_seed(999);
        let chosen = strategy.choose_move(Stone::Black, &board, &mut rng1);

        assert!(board.empty_position_ids().contains(&chosen));
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
    fn transform_encoding_with_identity_preserves_values() {
        let encoding = board_to_stone_channels(&Board::new(), Stone::Black);

        let result = apply_board_symmetry(&encoding, |pos| pos);

        for pos in PositionId::iter() {
            assert_eq!(result.get(pos), encoding.get(pos));
        }
    }

    #[test]
    fn transform_encoding_with_invert_moves_values() {
        let mut encoding = PositionMap::<f32, STONE_CHANNELS>::new(0.0);
        let first_pos = PositionId::iter().next().unwrap();
        let last_pos = first_pos.invert();

        *encoding.get_mut(first_pos) = [1.0, 2.0];

        let result = apply_board_symmetry(&encoding, PositionId::invert);

        assert_eq!(result.get(last_pos), &[1.0, 2.0]);
    }

    mod transform_pipeline_tests {
        use super::*;
        use crate::position::Position;

        fn pos(row: usize, col: usize) -> PositionId {
            PositionId::from_position(Position::new(row, col))
        }

        #[test]
        fn transform_then_inverse_returns_original_position() {
            for symmetry in BoardSymmetry::ALL {
                for original_pos in PositionId::iter() {
                    let transformed = symmetry.apply(original_pos);
                    let back = symmetry.apply_inverse(transformed);
                    assert_eq!(
                        back, original_pos,
                        "{symmetry:?}: {original_pos:?} -> {transformed:?} -> {back:?}"
                    );
                }
            }
        }

        #[test]
        fn selected_position_maps_back_to_valid_empty() {
            let mut state = Board::new();
            state.place(pos(7, 7), Stone::Black).unwrap();

            let empty_positions = state.empty_position_ids();

            for symmetry in BoardSymmetry::ALL {
                let transformed_empty: Vec<PositionId> =
                    empty_positions.iter().map(|&p| symmetry.apply(p)).collect();

                let transformed_pos = transformed_empty[0];
                let original_pos = symmetry.apply_inverse(transformed_pos);

                assert!(
                    empty_positions.contains(&original_pos),
                    "{symmetry:?}: inverse of {transformed_pos:?} = {original_pos:?} not in empty"
                );
            }
        }

        #[test]
        fn all_transforms_produce_valid_moves() {
            let mut rng = fastrand::Rng::with_seed(42);
            let strategy = TestStrategy::random("test", &mut rng);
            let state = Board::new();

            for seed in 0..100 {
                let mut rng = fastrand::Rng::with_seed(seed);
                let chosen = strategy.choose_move(Stone::Black, &state, &mut rng);

                assert!(
                    state.empty_position_ids().contains(&chosen),
                    "seed {seed}: chosen {chosen:?} not in empty positions"
                );
            }
        }

        #[test]
        fn transform_pipeline_preserves_best_position_semantics() {
            let mut state = Board::new();
            state.place(pos(7, 7), Stone::Black).unwrap();

            let mut rng = fastrand::Rng::with_seed(42);
            let strategy = TestStrategy::random("test", &mut rng);

            for seed in 0..50 {
                let mut rng = fastrand::Rng::with_seed(seed);
                let chosen = strategy.choose_move(Stone::Black, &state, &mut rng);

                assert!(
                    state.empty_position_ids().contains(&chosen),
                    "seed {seed}: position {chosen:?} is not empty"
                );
            }
        }
    }

    mod relu_tests {
        use super::*;

        #[test]
        fn positive_values_unchanged() {
            let mut data = vec![1.0, 2.5, 0.001];
            zero_negatives_inplace(&mut data);
            assert_eq!(data, vec![1.0, 2.5, 0.001]);
        }

        #[test]
        fn negative_values_become_zero() {
            let mut data = vec![-1.0, -0.001, -100.0];
            zero_negatives_inplace(&mut data);
            assert_eq!(data, vec![0.0, 0.0, 0.0]);
        }

        #[test]
        fn zero_unchanged() {
            let mut data = vec![0.0];
            zero_negatives_inplace(&mut data);
            assert_eq!(data, vec![0.0]);
        }

        #[test]
        fn mixed_values() {
            let mut data = vec![-2.0, 0.0, 3.0, -0.5, 1.0];
            zero_negatives_inplace(&mut data);
            assert_eq!(data, vec![0.0, 0.0, 3.0, 0.0, 1.0]);
        }

        #[test]
        fn empty_slice() {
            let mut data: Vec<f32> = vec![];
            zero_negatives_inplace(&mut data);
            assert!(data.is_empty());
        }
    }

    #[test]
    fn forward_with_hidden_layer_produces_finite_score_for_every_position() {
        // LAYERS=2 adds one middle CHANNELS -> CHANNELS layer, exercising the middle-layer loop.
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = ConvStrategy::<3, 4, 2, 0>::random("two_layers", &mut rng);
        let input = board_to_stone_channels(&Board::new(), Stone::Black);

        let output = strategy.score_positions(&input);

        for pos in PositionId::iter() {
            let [score] = *output.get(pos);
            assert!(score.is_finite(), "score at {pos:?} is {score}");
        }
    }
}
