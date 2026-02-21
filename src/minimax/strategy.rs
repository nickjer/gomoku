use crate::board::Board;
use crate::position_id::PositionId;
use crate::stone::Stone;
use crate::strategy::Strategy;

use super::search::find_best_move;

/// A minimax strategy with alpha-beta pruning.
pub struct MinimaxStrategy {
    depth: u32,
    label: String,
}

impl MinimaxStrategy {
    #[must_use]
    pub fn new(depth: u32) -> Self {
        Self {
            depth,
            label: format!("minimax-d{depth}"),
        }
    }
}

impl Strategy for MinimaxStrategy {
    fn choose_move(
        &self,
        current_stone: Stone,
        board: &Board,
        rng: &mut fastrand::Rng,
    ) -> PositionId {
        let mut board = *board;
        find_best_move(&mut board, current_stone, self.depth, rng)
    }

    fn label(&self) -> &str {
        &self.label
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;

    /// Depth 2 suffices for tests — these verify behavior (winning, blocking),
    /// not search quality, and depth 4 is too slow in debug builds.
    const TEST_DEPTH: u32 = 2;

    fn pos(row: usize, col: usize) -> PositionId {
        PositionId::from_position(Position::new(row, col))
    }

    fn place_stones(board: &mut Board, stone: Stone, positions: &[(usize, usize)]) {
        for &(row, col) in positions {
            board.place(pos(row, col), stone).unwrap();
        }
    }

    #[test]
    fn choose_move_returns_empty_position() {
        let strategy = MinimaxStrategy::new(TEST_DEPTH);
        let board = Board::new();
        let mut rng = fastrand::Rng::with_seed(42);

        let result = strategy.choose_move(Stone::Black, &board, &mut rng);

        assert!(board.is_empty(result));
    }

    #[test]
    fn choose_move_finds_winning_move() {
        let strategy = MinimaxStrategy::new(TEST_DEPTH);
        let mut board = Board::new();
        place_stones(&mut board, Stone::Black, &[(7, 5), (7, 6), (7, 7), (7, 8)]);
        place_stones(&mut board, Stone::White, &[(8, 5), (8, 6), (8, 7)]);
        let mut rng = fastrand::Rng::with_seed(42);

        let result = strategy.choose_move(Stone::Black, &board, &mut rng);

        assert!(result == pos(7, 4) || result == pos(7, 9));
    }

    #[test]
    fn choose_move_blocks_opponent_threat() {
        let strategy = MinimaxStrategy::new(TEST_DEPTH);
        let mut board = Board::new();
        // Half-open four: Black at (7,4) blocks one end, so (7,9) is the only block
        place_stones(&mut board, Stone::White, &[(7, 5), (7, 6), (7, 7), (7, 8)]);
        place_stones(&mut board, Stone::Black, &[(7, 4), (8, 5), (8, 6)]);
        let mut rng = fastrand::Rng::with_seed(42);

        let result = strategy.choose_move(Stone::Black, &board, &mut rng);

        assert_eq!(result, pos(7, 9));
    }

    #[test]
    fn label_includes_depth() {
        let strategy = MinimaxStrategy::new(6);

        assert_eq!(strategy.label(), "minimax-d6");
    }
}
