use tracing::instrument;

use crate::board::Board;
use crate::position_id::PositionId;
use crate::position_map::{PositionSlice, PositionSliceMut};
use crate::stone::Stone;
use crate::strategy::{EvolvableStrategy, Strategy};

use super::encoding::{INPUT_CHANNELS, encode_board};
use super::layer::{conv2d_im2col, relu_inplace};
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
    /// Input: `[2 * PositionId::COUNT]` (two-channel board encoding)
    /// Output: `[PositionId::COUNT]` (policy logits for each position)
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        // Allocate patch buffer for im2col (reused across all layers)
        // First layer needs INPUT_CHANNELS * K * K, hidden layers need C * K * K
        const fn max(a: usize, b: usize) -> usize {
            if a > b { a } else { b }
        }
        let patch_buffer_size = PositionId::COUNT * max(INPUT_CHANNELS, C) * K * K;
        let mut patch_buffer = vec![0.0f32; patch_buffer_size];

        // First conv: INPUT_CHANNELS -> C channels
        let mut x = conv2d_im2col::<INPUT_CHANNELS, C, K>(
            input,
            self.weights.first_conv_weights(),
            self.weights.first_conv_bias(),
            &mut patch_buffer,
        );
        relu_inplace(&mut x);

        // Hidden convs: C -> C channels
        for i in 0..(L - 1) {
            x = conv2d_im2col::<C, C, K>(
                &x,
                self.weights.hidden_conv_weights(i),
                self.weights.hidden_conv_bias(i),
                &mut patch_buffer,
            );
            relu_inplace(&mut x);
        }

        // Final conv: C -> 1 channel (1×1 kernel)
        conv2d_im2col::<C, 1, 1>(
            &x,
            self.weights.final_conv_weights(),
            &[self.weights.final_conv_bias()],
            &mut patch_buffer,
        )
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
        let policy_vec = self.forward(&transformed_encoding);

        // Transform empty positions to transformed space
        let transformed_empty: Vec<PositionId> = board
            .empty_position_ids()
            .iter()
            .map(|&pos| transform.apply(pos))
            .collect();

        // Select best position in transformed space
        let transformed_pos =
            select_best_position(&transformed_empty, PositionSlice::new(&policy_vec), rng);

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

/// Transforms a 2-channel encoding by applying a position transformation.
///
/// For each position `p`, sets `output[f(p)] = input[p]`.
fn transform_encoding(encoding: &[f32], f: impl Fn(PositionId) -> PositionId) -> Vec<f32> {
    let mut result = vec![0.0f32; encoding.len()];
    let (own_in, opp_in) = encoding.split_at(PositionId::COUNT);
    let (own_out, opp_out) = result.split_at_mut(PositionId::COUNT);

    let own_in = PositionSlice::new(own_in);
    let opp_in = PositionSlice::new(opp_in);
    let mut own_out = PositionSliceMut::new(own_out);
    let mut opp_out = PositionSliceMut::new(opp_out);

    for pos in PositionId::iter() {
        let dst_pos = f(pos);
        own_out[dst_pos] = own_in[pos];
        opp_out[dst_pos] = opp_in[pos];
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    // Use smaller config for faster tests: 3×3 kernel, 4 channels, 1 layer
    type TestStrategy = ConvStrategy<3, 4, 1, 0>;

    #[test]
    fn forward_produces_correct_output_size() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = TestStrategy::random("test", &mut rng);
        let input = vec![0.0f32; 2 * PositionId::COUNT];

        let output = strategy.forward(&input);

        assert_eq!(output.len(), PositionId::COUNT);
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
        assert_eq!(reconstructed.genes().as_ref(), original.genes().as_ref());
    }

    #[test]
    fn transform_encoding_with_identity_preserves_values() {
        let encoding: Vec<f32> = (0..2 * PositionId::COUNT).map(|i| i as f32).collect();

        let result = transform_encoding(&encoding, |pos| pos);

        assert_eq!(result, encoding);
    }

    #[test]
    fn transform_encoding_with_invert_moves_values() {
        let mut encoding = vec![0.0f32; 2 * PositionId::COUNT];
        let first_pos = PositionId::iter().next().unwrap();
        let last_pos = first_pos.invert();

        // Set distinctive values at first position in both channels
        encoding[PositionId::COUNT * 0 + usize::from(first_pos)] = 1.0;
        encoding[PositionId::COUNT * 1 + usize::from(first_pos)] = 2.0;

        let result = transform_encoding(&encoding, PositionId::invert);

        // After invert, first_pos maps to last_pos
        assert_eq!(result[PositionId::COUNT * 0 + usize::from(last_pos)], 1.0);
        assert_eq!(result[PositionId::COUNT * 1 + usize::from(last_pos)], 2.0);
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
                // Transform empty positions
                let transformed_empty: Vec<PositionId> = empty_positions
                    .iter()
                    .map(|&p| transform.apply(p))
                    .collect();

                // Pick any transformed position
                let transformed_pos = transformed_empty[0];

                // Map back to original
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

            // Test with many different seeds to cover different random transforms
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
            // Create a board with a specific pattern
            let mut state = Board::new();
            state.place(pos(7, 7), Stone::Black).unwrap();

            let mut rng = fastrand::Rng::with_seed(42);
            let strategy = TestStrategy::random("test", &mut rng);

            // Run many times with different transforms
            for seed in 0..50 {
                let mut rng = fastrand::Rng::with_seed(seed);
                let chosen = strategy.choose_move(Stone::Black, &state, &mut rng);

                // The chosen position must be empty
                assert!(
                    state.empty_position_ids().contains(&chosen),
                    "seed {seed}: position {chosen:?} is not empty"
                );
            }
        }
    }
}
