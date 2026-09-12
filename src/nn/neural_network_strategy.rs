use std::fmt;

use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::board::Board;
use crate::position_id::PositionId;
use crate::position_map::PositionMap;
use crate::stone::Stone;
use crate::strategy::{EvolvableStrategy, Strategy};

use super::cluster_expansion::{CLUSTER_COUNT, ClusterExpansion};
use super::neighborhood_encoder::NeighborhoodEncoder;
use super::neural_network::NeuralNetwork;
use super::square3x3::Square3x3;
use super::stone_channels::board_to_stone_channels;

/// Plays the empty position that a neural network scores highest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetworkStrategy<Encoder, const CHANNELS: usize, const LAYERS: usize> {
    label: String,
    network: NeuralNetwork<Encoder, CHANNELS, LAYERS>,
}

/// 3×3 squares, 32 channels, 2 layers. ~10K parameters.
pub type ConvTiny = NeuralNetworkStrategy<Square3x3, 32, 2>;

/// 3×3 squares, 64 channels, 4 layers. ~112K parameters.
pub type ConvSmall = NeuralNetworkStrategy<Square3x3, 64, 4>;

/// Full cluster expansion, 32 channels, 2 layers. ~10K parameters.
pub type ClusterTiny = NeuralNetworkStrategy<ClusterExpansion<CLUSTER_COUNT>, 32, 2>;

/// Full cluster expansion, 64 channels, 4 layers. ~112K parameters.
pub type ClusterSmall = NeuralNetworkStrategy<ClusterExpansion<CLUSTER_COUNT>, 64, 4>;

impl<Encoder: NeighborhoodEncoder, const CHANNELS: usize, const LAYERS: usize> fmt::Display
    for NeuralNetworkStrategy<Encoder, CHANNELS, LAYERS>
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(formatter, "Label: {}", self.label)?;
        write!(formatter, "{}", self.network)
    }
}

impl<Encoder: NeighborhoodEncoder, const CHANNELS: usize, const LAYERS: usize> Strategy
    for NeuralNetworkStrategy<Encoder, CHANNELS, LAYERS>
{
    /// Turns the board as the encoder asks, scores it, and maps the best
    /// empty position back to the real board.
    #[instrument(level = "trace", skip_all)]
    fn choose_move(
        &self,
        current_stone: Stone,
        board: &Board,
        rng: &mut fastrand::Rng,
    ) -> PositionId {
        let symmetry = Encoder::board_symmetry(rng);
        let stone_channels = symmetry.apply_to_map(&board_to_stone_channels(board, current_stone));
        let scores = self.network.score_positions(stone_channels);

        let empty: Vec<PositionId> = board
            .empty_position_ids()
            .iter()
            .map(|&pos| symmetry.apply(pos))
            .collect();
        let chosen = highest_scored_empty_position(&empty, &scores, rng);
        symmetry.apply_inverse(chosen)
    }

    fn label(&self) -> &str {
        &self.label
    }
}

impl<Encoder: NeighborhoodEncoder, const CHANNELS: usize, const LAYERS: usize> EvolvableStrategy
    for NeuralNetworkStrategy<Encoder, CHANNELS, LAYERS>
{
    type Genes = NeuralNetwork<Encoder, CHANNELS, LAYERS>;

    fn random(label: impl Into<String>, rng: &mut fastrand::Rng) -> Self {
        Self {
            label: label.into(),
            network: NeuralNetwork::random(rng),
        }
    }

    fn genes(&self) -> &Self::Genes {
        &self.network
    }

    fn from_genes(label: impl Into<String>, genes: Self::Genes) -> Self {
        Self {
            label: label.into(),
            network: genes,
        }
    }
}

/// The empty position with the highest score. Ties are broken uniformly at
/// random.
///
/// # Panics
///
/// Panics if `empty_positions` is empty.
fn highest_scored_empty_position(
    empty_positions: &[PositionId],
    scores: &PositionMap<f32, 1>,
    rng: &mut fastrand::Rng,
) -> PositionId {
    let (&first, rest) = empty_positions
        .split_first()
        .expect("no empty positions to select from");

    let mut best_pos = first;
    let [mut best_score] = *scores.get(first);
    let mut tie_count = 1;

    for &pos in rest {
        let [score] = *scores.get(pos);

        if score > best_score {
            best_score = score;
            best_pos = pos;
            tie_count = 1;
        } else if (score - best_score).abs() < f32::EPSILON {
            tie_count += 1;
            // Reservoir sampling: replace with probability 1/tie_count
            if rng.usize(..tie_count) == 0 {
                best_pos = pos;
            }
        }
    }

    best_pos
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;

    // Small configurations for fast tests: 4 channels, 1 layer.
    type SquareStrategy = NeuralNetworkStrategy<Square3x3, 4, 1>;
    type ClusterStrategy = NeuralNetworkStrategy<ClusterExpansion<CLUSTER_COUNT>, 4, 1>;

    fn pos(row: usize, col: usize) -> PositionId {
        PositionId::from_position(Position::new(row, col))
    }

    #[test]
    fn square_strategy_chooses_an_empty_position() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = SquareStrategy::random("test", &mut rng);
        let board = Board::new();

        let chosen = strategy.choose_move(Stone::Black, &board, &mut rng);

        assert!(board.empty_position_ids().contains(&chosen));
    }

    #[test]
    fn cluster_strategy_chooses_an_empty_position() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = ClusterStrategy::random("test", &mut rng);
        let board = Board::new();

        let chosen = strategy.choose_move(Stone::Black, &board, &mut rng);

        assert!(board.empty_position_ids().contains(&chosen));
    }

    #[test]
    fn choose_move_is_deterministic_for_a_seed() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = SquareStrategy::random("test", &mut rng);
        let board = Board::new();

        let mut rng1 = fastrand::Rng::with_seed(123);
        let mut rng2 = fastrand::Rng::with_seed(123);

        assert_eq!(
            strategy.choose_move(Stone::Black, &board, &mut rng1),
            strategy.choose_move(Stone::Black, &board, &mut rng2)
        );
    }

    #[test]
    fn every_symmetry_yields_a_valid_move_on_a_played_board() {
        let mut board = Board::new();
        board.place(pos(7, 7), Stone::Black).unwrap();
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = SquareStrategy::random("test", &mut rng);

        for seed in 0..100 {
            let mut rng = fastrand::Rng::with_seed(seed);
            let chosen = strategy.choose_move(Stone::White, &board, &mut rng);

            assert!(
                board.empty_position_ids().contains(&chosen),
                "seed {seed}: {chosen:?} is not empty"
            );
        }
    }

    #[test]
    fn display_shows_label_then_network() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = SquareStrategy::random("shown", &mut rng);

        let text = strategy.to_string();

        assert!(text.starts_with("Label: shown\nNeural network"), "{text}");
    }

    #[test]
    fn random_creates_strategy_with_label() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = SquareStrategy::random("my_label", &mut rng);

        assert_eq!(strategy.label(), "my_label");
    }

    #[test]
    fn from_genes_creates_strategy_with_given_network() {
        let mut rng = fastrand::Rng::with_seed(42);
        let original = SquareStrategy::random("original", &mut rng);

        let reconstructed = SquareStrategy::from_genes("reconstructed", original.genes().clone());

        assert_eq!(reconstructed.label(), "reconstructed");
        assert_eq!(reconstructed.genes(), original.genes());
    }

    mod highest_scored_empty_position_tests {
        use super::*;

        fn scores_with_values(values: &[(PositionId, f32)]) -> PositionMap<f32, 1> {
            let mut map = PositionMap::new(f32::NEG_INFINITY);
            for &(pos, value) in values {
                *map.get_mut(pos) = [value];
            }
            map
        }

        #[test]
        fn selects_highest_score() {
            let positions = vec![pos(0, 0), pos(0, 1), pos(0, 2)];
            let scores =
                scores_with_values(&[(pos(0, 0), 1.0), (pos(0, 1), 5.0), (pos(0, 2), 3.0)]);
            let mut rng = fastrand::Rng::with_seed(42);

            let selected = highest_scored_empty_position(&positions, &scores, &mut rng);

            assert_eq!(selected, pos(0, 1));
        }

        #[test]
        fn handles_negative_scores() {
            let positions = vec![pos(0, 0), pos(0, 1), pos(0, 2)];
            let scores =
                scores_with_values(&[(pos(0, 0), -5.0), (pos(0, 1), -1.0), (pos(0, 2), -3.0)]);
            let mut rng = fastrand::Rng::with_seed(42);

            let selected = highest_scored_empty_position(&positions, &scores, &mut rng);

            assert_eq!(selected, pos(0, 1));
        }

        #[test]
        fn tiebreaking_selects_from_tied_positions() {
            let positions = vec![pos(0, 0), pos(0, 1), pos(0, 2)];
            let scores =
                scores_with_values(&[(pos(0, 0), 5.0), (pos(0, 1), 5.0), (pos(0, 2), 1.0)]);
            let mut rng = fastrand::Rng::with_seed(42);

            let selected = highest_scored_empty_position(&positions, &scores, &mut rng);

            assert!([pos(0, 0), pos(0, 1)].contains(&selected), "{selected:?}");
        }

        #[test]
        fn tiebreaking_is_uniform() {
            let positions = vec![pos(0, 0), pos(0, 1)];
            let scores = scores_with_values(&[(pos(0, 0), 5.0), (pos(0, 1), 5.0)]);

            let mut counts = [0, 0];
            for seed in 0..1000 {
                let mut rng = fastrand::Rng::with_seed(seed);
                let selected = highest_scored_empty_position(&positions, &scores, &mut rng);

                if selected == pos(0, 0) {
                    counts[0] += 1;
                } else {
                    counts[1] += 1;
                }
            }

            // With 1000 trials, each should be ~500. Allow 40% to 60% range.
            assert!(
                counts[0] > 400 && counts[0] < 600,
                "distribution not uniform: {counts:?}"
            );
        }

        #[test]
        fn deterministic_with_same_seed() {
            let positions = vec![pos(0, 0), pos(0, 1), pos(0, 2)];
            let scores =
                scores_with_values(&[(pos(0, 0), 5.0), (pos(0, 1), 5.0), (pos(0, 2), 5.0)]);

            let results: Vec<_> = (0..5)
                .map(|_| {
                    let mut rng = fastrand::Rng::with_seed(12345);
                    highest_scored_empty_position(&positions, &scores, &mut rng)
                })
                .collect();

            assert!(results.windows(2).all(|w| w[0] == w[1]));
        }

        #[test]
        fn only_considers_provided_positions() {
            // pos(0,0) has the highest score but isn't in the list
            let positions = vec![pos(0, 1), pos(0, 2)];
            let scores =
                scores_with_values(&[(pos(0, 0), 100.0), (pos(0, 1), 5.0), (pos(0, 2), 3.0)]);
            let mut rng = fastrand::Rng::with_seed(42);

            let selected = highest_scored_empty_position(&positions, &scores, &mut rng);

            assert_eq!(selected, pos(0, 1));
        }

        #[test]
        fn single_position_returns_that_position() {
            let positions = vec![pos(7, 7)];
            let scores = scores_with_values(&[(pos(7, 7), 0.0)]);
            let mut rng = fastrand::Rng::with_seed(42);

            let selected = highest_scored_empty_position(&positions, &scores, &mut rng);

            assert_eq!(selected, pos(7, 7));
        }

        #[test]
        #[should_panic(expected = "no empty positions")]
        fn panics_on_empty_positions() {
            let positions: Vec<PositionId> = vec![];
            let scores = PositionMap::<f32, 1>::new(0.0);
            let mut rng = fastrand::Rng::with_seed(42);

            highest_scored_empty_position(&positions, &scores, &mut rng);
        }
    }
}
