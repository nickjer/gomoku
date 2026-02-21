use std::ops;

use derive_more::{Add, From, Neg, Sub};

use crate::bitboard::count_all_patterns;
use crate::board::Board;
use crate::stone::Stone;

/// Evaluation score for minimax search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Add, Sub, Neg, From)]
pub struct Score(i32);

impl Score {
    pub const WIN: Self = Self(1_000_000);
    pub const MIN: Self = Self(-i32::MAX);
    pub const MAX: Self = Self(i32::MAX);

    const OPEN_FOUR: Self = Self(100_000);
    const HALF_OPEN_FOUR: Self = Self(10_000);
    const OPEN_THREE: Self = Self(5_000);
    const HALF_OPEN_THREE: Self = Self(500);
    const OPEN_TWO: Self = Self(200);
    const HALF_OPEN_TWO: Self = Self(20);

    /// Win score adjusted for depth — prefers faster wins (fewer moves).
    #[must_use]
    pub fn win_at_depth(move_count: usize) -> Self {
        Self(
            Self::WIN
                .0
                .checked_sub(i32::try_from(move_count).expect("move count overflow"))
                .expect("win score underflow"),
        )
    }
}

impl ops::Mul<usize> for Score {
    type Output = Self;

    fn mul(self, rhs: usize) -> Self {
        Self(self.0 * i32::try_from(rhs).expect("count overflow"))
    }
}

fn score_patterns(own: crate::bitboard::BitBoard, opponent: crate::bitboard::BitBoard) -> Score {
    let counts = count_all_patterns(own, opponent);
    Score::WIN * counts.fives
        + Score::OPEN_FOUR * counts.open_fours
        + Score::HALF_OPEN_FOUR * counts.half_open_fours
        + Score::OPEN_THREE * counts.open_threes
        + Score::HALF_OPEN_THREE * counts.half_open_threes
        + Score::OPEN_TWO * counts.open_twos
        + Score::HALF_OPEN_TWO * counts.half_open_twos
}

/// Evaluates the board from `stone`'s perspective.
///
/// Returns positive if `stone` has the advantage, negative if the opponent does.
pub fn evaluate(board: &Board, stone: Stone) -> Score {
    let own = *board.bitboard(stone);
    let opponent = *board.bitboard(stone.opponent());
    score_patterns(own, opponent) - score_patterns(opponent, own)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;
    use crate::position_id::PositionId;

    fn pos(row: usize, col: usize) -> PositionId {
        PositionId::from_position(Position::new(row, col))
    }

    fn place_stones(board: &mut Board, stone: Stone, positions: &[(usize, usize)]) {
        for &(row, col) in positions {
            board.place(pos(row, col), stone).unwrap();
        }
    }

    #[test]
    fn empty_board_evaluates_to_zero() {
        let board = Board::new();

        assert_eq!(evaluate(&board, Stone::Black), Score::from(0));
    }

    #[test]
    fn five_in_a_row_returns_win_score() {
        let mut board = Board::new();
        place_stones(
            &mut board,
            Stone::Black,
            &[(7, 5), (7, 6), (7, 7), (7, 8), (7, 9)],
        );

        let score = evaluate(&board, Stone::Black);

        assert!(score >= Score::WIN);
    }

    #[test]
    fn open_four_scores_higher_than_open_three() {
        let mut four_board = Board::new();
        place_stones(
            &mut four_board,
            Stone::Black,
            &[(7, 5), (7, 6), (7, 7), (7, 8)],
        );
        let four_score = evaluate(&four_board, Stone::Black);

        let mut three_board = Board::new();
        place_stones(&mut three_board, Stone::Black, &[(7, 5), (7, 6), (7, 7)]);
        let three_score = evaluate(&three_board, Stone::Black);

        assert!(four_score > three_score);
    }

    #[test]
    fn half_open_four_scores_higher_than_half_open_three() {
        let mut four_board = Board::new();
        place_stones(
            &mut four_board,
            Stone::Black,
            &[(7, 5), (7, 6), (7, 7), (7, 8)],
        );
        place_stones(&mut four_board, Stone::White, &[(7, 4)]);
        let four_score = evaluate(&four_board, Stone::Black);

        let mut three_board = Board::new();
        place_stones(&mut three_board, Stone::Black, &[(7, 5), (7, 6), (7, 7)]);
        place_stones(&mut three_board, Stone::White, &[(7, 4)]);
        let three_score = evaluate(&three_board, Stone::Black);

        assert!(four_score > three_score);
    }

    #[test]
    fn closed_pattern_scores_zero() {
        // Three blocked on both sides — no scoring contribution
        let mut board = Board::new();
        place_stones(&mut board, Stone::Black, &[(7, 5), (7, 6), (7, 7)]);
        place_stones(&mut board, Stone::White, &[(7, 4), (7, 8)]);

        let score = evaluate(&board, Stone::Black);

        assert!(score <= Score::from(0));
    }

    #[test]
    fn evaluation_is_symmetric() {
        let mut board = Board::new();
        place_stones(&mut board, Stone::Black, &[(7, 5), (7, 6), (7, 7)]);
        place_stones(&mut board, Stone::White, &[(8, 5), (8, 6)]);

        let black_score = evaluate(&board, Stone::Black);
        let white_score = evaluate(&board, Stone::White);

        assert_eq!(black_score, -white_score);
    }

    #[test]
    fn opponent_threats_reduce_score() {
        let mut own_only = Board::new();
        place_stones(&mut own_only, Stone::Black, &[(7, 5), (7, 6), (7, 7)]);
        let own_score = evaluate(&own_only, Stone::Black);

        let mut with_threat = Board::new();
        place_stones(&mut with_threat, Stone::Black, &[(7, 5), (7, 6), (7, 7)]);
        place_stones(&mut with_threat, Stone::White, &[(8, 5), (8, 6), (8, 7)]);
        let threat_score = evaluate(&with_threat, Stone::Black);

        assert!(threat_score < own_score);
    }

    #[test]
    fn open_three_in_each_direction_scores_equally() {
        let directions: &[&[(usize, usize)]] = &[
            &[(7, 5), (7, 6), (7, 7)], // horizontal
            &[(5, 7), (6, 7), (7, 7)], // vertical
            &[(5, 5), (6, 6), (7, 7)], // diagonal down-right
            &[(5, 9), (6, 8), (7, 7)], // diagonal down-left
        ];

        let scores: Vec<Score> = directions
            .iter()
            .map(|positions| {
                let mut board = Board::new();
                place_stones(&mut board, Stone::Black, positions);
                evaluate(&board, Stone::Black)
            })
            .collect();

        for score in &scores[1..] {
            assert_eq!(*score, scores[0]);
        }
    }

    #[test]
    fn half_open_at_board_edge() {
        // Three against left edge — edge acts as a block
        let mut board = Board::new();
        place_stones(&mut board, Stone::Black, &[(7, 0), (7, 1), (7, 2)]);
        let edge_score = evaluate(&board, Stone::Black);

        // Compare with an open three in the middle
        let mut open_board = Board::new();
        place_stones(&mut open_board, Stone::Black, &[(7, 5), (7, 6), (7, 7)]);
        let open_score = evaluate(&open_board, Stone::Black);

        assert!(edge_score < open_score);
    }
}
