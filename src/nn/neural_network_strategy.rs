use std::fmt;

use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::board::Board;
use crate::position_id::PositionId;
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
    /// Turns the board as the encoder asks, ranks the empty positions, and
    /// maps the best one back to the real board.
    ///
    /// # Panics
    ///
    /// Panics if the board is full.
    #[instrument(level = "trace", skip_all)]
    fn choose_move(
        &self,
        current_stone: Stone,
        board: &Board,
        rng: &mut fastrand::Rng,
    ) -> PositionId {
        let symmetry = Encoder::board_symmetry(rng);
        let turned_board = symmetry.apply_to_board(board);
        let ranked_positions = self.rank_positions(current_stone, &turned_board, rng);
        let best_turned_position = *ranked_positions
            .first()
            .expect("no empty positions to choose from");
        symmetry.apply_inverse(best_turned_position)
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

    fn rank_positions(
        &self,
        current_stone: Stone,
        board: &Board,
        rng: &mut fastrand::Rng,
    ) -> Vec<PositionId> {
        let scores = self
            .network
            .score_positions(board_to_stone_channels(board, current_stone));
        // Shuffling first puts tied positions in random order, since the sort is stable.
        let mut ranked_positions = board.empty_position_ids();
        rng.shuffle(&mut ranked_positions);
        ranked_positions.sort_by(|&earlier_position, &later_position| {
            scores.get(later_position)[0].total_cmp(&scores.get(earlier_position)[0])
        });
        ranked_positions
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::position::Position;

    // Small configurations for fast tests: 4 channels, 1 layer.
    type SquareStrategy = NeuralNetworkStrategy<Square3x3, 4, 1>;
    type ClusterStrategy = NeuralNetworkStrategy<ClusterExpansion<CLUSTER_COUNT>, 4, 1>;

    fn pos(row: usize, col: usize) -> PositionId {
        PositionId::from_position(Position::new(row, col))
    }

    /// A few stones in the middle so the network has something to score.
    fn played_board() -> Board {
        let mut board = Board::new();
        board.place(pos(7, 7), Stone::Black).unwrap();
        board.place(pos(7, 8), Stone::White).unwrap();
        board.place(pos(6, 6), Stone::Black).unwrap();
        board
    }

    #[test]
    fn square_strategy_chooses_an_empty_position() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = SquareStrategy::random("test", &mut rng);
        let board = Board::new();

        let chosen_position = strategy.choose_move(Stone::Black, &board, &mut rng);

        assert!(board.empty_position_ids().contains(&chosen_position));
    }

    #[test]
    fn cluster_strategy_chooses_an_empty_position() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = ClusterStrategy::random("test", &mut rng);
        let board = Board::new();

        let chosen_position = strategy.choose_move(Stone::Black, &board, &mut rng);

        assert!(board.empty_position_ids().contains(&chosen_position));
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
            let chosen_position = strategy.choose_move(Stone::White, &board, &mut rng);

            assert!(
                board.empty_position_ids().contains(&chosen_position),
                "seed {seed}: {chosen_position:?} is not empty"
            );
        }
    }

    #[test]
    #[should_panic(expected = "no empty positions")]
    fn choose_move_panics_on_a_full_board() {
        let mut board = Board::new();
        for (position, stone) in crate::test_utils::draw_moves() {
            board.place(position, stone).unwrap();
        }
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = SquareStrategy::random("test", &mut rng);

        strategy.choose_move(Stone::Black, &board, &mut rng);
    }

    #[test]
    fn rank_positions_orders_every_empty_position_by_falling_score() {
        let board = played_board();
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = SquareStrategy::random("test", &mut rng);
        let scores = strategy
            .network
            .score_positions(board_to_stone_channels(&board, Stone::White));

        let ranked_positions = strategy.rank_positions(Stone::White, &board, &mut rng);

        assert_eq!(ranked_positions.len(), board.empty_position_ids().len());
        assert_eq!(
            ranked_positions.iter().collect::<HashSet<_>>().len(),
            ranked_positions.len(),
            "a position is ranked twice"
        );
        assert!(
            ranked_positions
                .iter()
                .all(|&position| board.is_empty(position))
        );
        assert!(
            ranked_positions
                .windows(2)
                .all(|pair| scores.get(pair[0])[0] >= scores.get(pair[1])[0]),
            "scores rise along the ranking"
        );
    }

    #[test]
    fn rank_positions_puts_tied_positions_in_random_order() {
        // With zero biases an empty board scores every position the same.
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = SquareStrategy::random("test", &mut rng);
        let board = Board::new();

        let first_ranked_positions: HashSet<PositionId> = (0..200)
            .map(|seed| {
                let mut rng = fastrand::Rng::with_seed(seed);
                strategy.rank_positions(Stone::Black, &board, &mut rng)[0]
            })
            .collect();

        assert!(
            first_ranked_positions.len() > 100,
            "{} distinct",
            first_ranked_positions.len()
        );
    }

    #[test]
    fn choose_move_plays_the_first_ranked_position_when_the_board_is_not_turned() {
        let board = played_board();
        let mut rng = fastrand::Rng::with_seed(42);
        // The cluster encoder never turns the board, so both read it as it lies;
        // the same seed makes both break ties the same way.
        let strategy = ClusterStrategy::random("test", &mut rng);
        let mut ranking_rng = fastrand::Rng::with_seed(7);
        let mut choosing_rng = fastrand::Rng::with_seed(7);

        let first_ranked_position =
            strategy.rank_positions(Stone::White, &board, &mut ranking_rng)[0];

        assert_eq!(
            strategy.choose_move(Stone::White, &board, &mut choosing_rng),
            first_ranked_position
        );
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
}
