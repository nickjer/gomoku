use tracing::instrument;

use crate::board::Board;
use crate::position_id::PositionId;
use crate::position_map::PositionMap;
use crate::stone::Stone;
use crate::strategy::{EvolvableStrategy, Strategy};

use super::encoding::encode_board;
use super::select::select_best_position;
use super::symmetry::D8Transform;
use super::weights::ConvWeights;

/// A convolutional neural network strategy for Gomoku.
///
/// # Type Parameters
/// - `K`: Kernel size (e.g., 3 for 3×3 kernels)
/// - `C`: Number of channels in hidden layers
/// - `L`: Number of convolutional layers (must be >= 1)
/// - `R`: Number of residual blocks (must be 0 until implemented)
#[derive(Debug, Clone)]
pub struct ConvStrategy<const K: usize, const C: usize, const L: usize, const R: usize> {
    label: String,
    weights: ConvWeights<K, C, L, R>,
}

/// A tiny convolutional strategy: 3×3 kernels, 32 channels, 2 layers, no residual blocks.
/// ~10K parameters.
pub type ConvTiny = ConvStrategy<3, 32, 2, 0>;

/// A small convolutional strategy: 3×3 kernels, 64 channels, 4 layers, no residual blocks.
/// ~112K parameters.
pub type ConvSmall = ConvStrategy<3, 64, 4, 0>;

impl<const K: usize, const C: usize, const L: usize, const R: usize> ConvStrategy<K, C, L, R> {
    /// Forward pass through the network.
    ///
    /// Input: encoding with `INPUT_CHANNELS` channels per position.
    /// Output: [`PositionMap<f32>`] with policy logits for each position.
    fn forward(&self, input: &PositionMap<f32>) -> PositionMap<f32> {
        // First conv: INPUT_CHANNELS -> C channels
        let mut activations = self.weights.first().conv2d(input);
        relu_inplace(activations.as_mut_slice());

        // Hidden convs: C -> C channels
        for hidden in self.weights.hidden() {
            activations = hidden.conv2d(&activations);
            relu_inplace(activations.as_mut_slice());
        }

        // Final conv: C -> 1 channel (1×1 kernel)
        self.weights.last().conv2d(&activations)
    }
}

impl<const K: usize, const C: usize, const L: usize, const R: usize> Strategy
    for ConvStrategy<K, C, L, R>
{
    #[instrument(level = "trace", skip_all)]
    fn choose_move(
        &self,
        current_stone: Stone,
        board: &Board,
        rng: &mut fastrand::Rng,
    ) -> PositionId {
        let encoding = encode_board(board, current_stone);

        // Random D8 transform for data augmentation (AlphaGo Zero style)
        let transform = D8Transform::random(rng);

        // Transform input encoding
        let transformed_encoding = transform_encoding(&encoding, |pos| transform.apply(pos));

        // Forward pass
        let policy = self.forward(&transformed_encoding);

        // Transform empty positions to transformed space
        let transformed_empty: Vec<PositionId> = board
            .empty_position_ids()
            .iter()
            .map(|&pos| transform.apply(pos))
            .collect();

        // Select best position in transformed space
        let transformed_pos = select_best_position(&transformed_empty, &policy, rng);

        // Map selected position back to original orientation
        transform.apply_inverse(transformed_pos)
    }

    fn label(&self) -> &str {
        &self.label
    }
}

impl<const K: usize, const C: usize, const L: usize, const R: usize> EvolvableStrategy
    for ConvStrategy<K, C, L, R>
{
    type Genes = ConvWeights<K, C, L, R>;

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

/// Applies `ReLU` activation in-place: `x = max(0, x)`.
fn relu_inplace(data: &mut [f32]) {
    for val in data.iter_mut() {
        *val = val.max(0.0);
    }
}

/// Transforms an encoding by applying a position transformation.
///
/// For each position `p`, copies all channels from `encoding[p]` to `result[f(p)]`.
fn transform_encoding(
    encoding: &PositionMap<f32>,
    f: impl Fn(PositionId) -> PositionId,
) -> PositionMap<f32> {
    let mut result = PositionMap::new(0.0, encoding.stride());
    for pos in PositionId::iter() {
        result.get_mut(f(pos)).copy_from_slice(encoding.get(pos));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conv::encoding::INPUT_CHANNELS;

    // Use smaller config for faster tests: 3×3 kernel, 4 channels, 1 layer
    type TestStrategy = ConvStrategy<3, 4, 1, 0>;

    #[test]
    fn forward_produces_correct_output_size() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = TestStrategy::random("test", &mut rng);
        let input = PositionMap::new(0.0f32, INPUT_CHANNELS);

        let output = strategy.forward(&input);

        assert_eq!(output.stride(), 1);
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
        let encoding = encode_board(&Board::new(), Stone::Black);

        let result = transform_encoding(&encoding, |pos| pos);

        for pos in PositionId::iter() {
            assert_eq!(result.get(pos), encoding.get(pos));
        }
    }

    #[test]
    fn transform_encoding_with_invert_moves_values() {
        let mut encoding = PositionMap::new(0.0f32, INPUT_CHANNELS);
        let first_pos = PositionId::iter().next().unwrap();
        let last_pos = first_pos.invert();

        encoding.get_mut(first_pos).copy_from_slice(&[1.0, 2.0]);

        let result = transform_encoding(&encoding, PositionId::invert);

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
            for transform in D8Transform::ALL {
                for original_pos in PositionId::iter() {
                    let transformed = transform.apply(original_pos);
                    let back = transform.apply_inverse(transformed);
                    assert_eq!(
                        back, original_pos,
                        "{transform:?}: {original_pos:?} -> {transformed:?} -> {back:?}"
                    );
                }
            }
        }

        #[test]
        fn selected_position_maps_back_to_valid_empty() {
            let mut state = Board::new();
            state.place(pos(7, 7), Stone::Black).unwrap();

            let empty_positions = state.empty_position_ids();

            for transform in D8Transform::ALL {
                let transformed_empty: Vec<PositionId> = empty_positions
                    .iter()
                    .map(|&p| transform.apply(p))
                    .collect();

                let transformed_pos = transformed_empty[0];
                let original_pos = transform.apply_inverse(transformed_pos);

                assert!(
                    empty_positions.contains(&original_pos),
                    "{transform:?}: inverse of {transformed_pos:?} = {original_pos:?} not in empty"
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
            relu_inplace(&mut data);
            assert_eq!(data, vec![1.0, 2.5, 0.001]);
        }

        #[test]
        fn negative_values_become_zero() {
            let mut data = vec![-1.0, -0.001, -100.0];
            relu_inplace(&mut data);
            assert_eq!(data, vec![0.0, 0.0, 0.0]);
        }

        #[test]
        fn zero_unchanged() {
            let mut data = vec![0.0];
            relu_inplace(&mut data);
            assert_eq!(data, vec![0.0]);
        }

        #[test]
        fn mixed_values() {
            let mut data = vec![-2.0, 0.0, 3.0, -0.5, 1.0];
            relu_inplace(&mut data);
            assert_eq!(data, vec![0.0, 0.0, 3.0, 0.0, 1.0]);
        }

        #[test]
        fn empty_slice() {
            let mut data: Vec<f32> = vec![];
            relu_inplace(&mut data);
            assert!(data.is_empty());
        }
    }
}
