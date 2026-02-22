use crate::board::Board;
use crate::outcome::Outcome;
use crate::position_id::PositionId;
use crate::stone::Stone;

use super::lines::{LINE_LENGTHS, NUM_LINES, POSITION_LINES, score_line};
use super::score::Score;

/// Board wrapper that maintains incremental line-based evaluation scores.
///
/// Instead of recomputing full-board pattern counts at every leaf node,
/// `SearchState` tracks u16 bit masks for each of the 88 board lines and
/// updates only the 4 affected lines per place/undo. Leaf evaluation is O(1).
pub struct SearchState {
    board: Board,
    line_black: [u16; NUM_LINES],
    line_white: [u16; NUM_LINES],
    line_scores: [Score; NUM_LINES],
    total_score: Score,
}

impl SearchState {
    /// Initializes line masks and scores from an existing board.
    #[must_use]
    pub fn from_board(board: &Board) -> Self {
        let mut line_black = [0u16; NUM_LINES];
        let mut line_white = [0u16; NUM_LINES];

        for position in PositionId::iter() {
            if let Some(stone) = board.stone(position) {
                for &(line_id, bit) in POSITION_LINES.get(position) {
                    match stone {
                        Stone::Black => line_black[usize::from(line_id)] |= 1 << bit,
                        Stone::White => line_white[usize::from(line_id)] |= 1 << bit,
                    }
                }
            }
        }

        let mut line_scores = [Score::DRAW; NUM_LINES];
        let mut total_score = Score::DRAW;
        for i in 0..NUM_LINES {
            line_scores[i] = score_line(line_black[i], line_white[i], LINE_LENGTHS[i]);
            total_score += line_scores[i];
        }

        Self {
            board: *board,
            line_black,
            line_white,
            line_scores,
            total_score,
        }
    }

    /// Places a stone and incrementally updates the 4 affected line scores.
    pub fn place(&mut self, position: PositionId, stone: Stone) {
        self.board
            .place(position, stone)
            .expect("valid search move");

        for &(line_id, bit) in POSITION_LINES.get(position) {
            let idx = usize::from(line_id);
            self.total_score -= self.line_scores[idx];

            match stone {
                Stone::Black => self.line_black[idx] |= 1 << bit,
                Stone::White => self.line_white[idx] |= 1 << bit,
            }

            self.line_scores[idx] = score_line(
                self.line_black[idx],
                self.line_white[idx],
                LINE_LENGTHS[idx],
            );
            self.total_score += self.line_scores[idx];
        }
    }

    /// Undoes a stone placement and incrementally updates the 4 affected line scores.
    pub fn undo(&mut self, position: PositionId, stone: Stone) {
        self.board.undo(position, stone);

        for &(line_id, bit) in POSITION_LINES.get(position) {
            let idx = usize::from(line_id);
            self.total_score -= self.line_scores[idx];

            match stone {
                Stone::Black => self.line_black[idx] &= !(1 << bit),
                Stone::White => self.line_white[idx] &= !(1 << bit),
            }

            self.line_scores[idx] = score_line(
                self.line_black[idx],
                self.line_white[idx],
                LINE_LENGTHS[idx],
            );
            self.total_score += self.line_scores[idx];
        }
    }

    /// O(1) evaluation from `stone`'s perspective.
    ///
    /// Returns the running total score (positive = Black advantage),
    /// negated if `stone` is White.
    #[must_use]
    pub fn evaluate(&self, stone: Stone) -> Score {
        match stone {
            Stone::Black => self.total_score,
            Stone::White => -self.total_score,
        }
    }

    #[must_use]
    pub fn board(&self) -> &Board {
        &self.board
    }

    #[must_use]
    pub fn outcome(&self) -> Option<Outcome> {
        self.board.outcome()
    }

    #[must_use]
    pub fn move_count(&self) -> usize {
        self.board.move_count()
    }

    #[must_use]
    pub fn is_full(&self) -> bool {
        self.board.is_full()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;

    fn pos(row: usize, col: usize) -> PositionId {
        PositionId::from_position(Position::new(row, col))
    }

    #[test]
    fn empty_board_evaluates_to_zero() {
        let board = Board::new();
        let state = SearchState::from_board(&board);

        assert_eq!(state.evaluate(Stone::Black), Score::DRAW);
        assert_eq!(state.evaluate(Stone::White), Score::DRAW);
    }

    #[test]
    fn incremental_place_matches_from_scratch() {
        let mut rng = fastrand::Rng::with_seed(54321);
        for _ in 0..200 {
            let mut board = Board::new();
            let mut state = SearchState::from_board(&board);
            let stone_count = rng.usize(1..20);
            let mut positions: Vec<PositionId> = PositionId::iter().collect();
            rng.shuffle(&mut positions);
            let mut turn = Stone::Black;
            for &position in positions.iter().take(stone_count) {
                if board.is_finished() || state.outcome().is_some() {
                    break;
                }
                board.place(position, turn).unwrap();
                state.place(position, turn);

                let expected = SearchState::from_board(&board);
                assert_eq!(
                    state.total_score,
                    expected.total_score,
                    "Score mismatch after placing at ({}, {}):\n{board}",
                    position.row(),
                    position.col()
                );
                turn = turn.opponent();
            }
        }
    }

    #[test]
    fn undo_restores_previous_score() {
        let mut board = Board::new();
        board.place(pos(7, 7), Stone::Black).unwrap();
        board.place(pos(7, 8), Stone::White).unwrap();
        board.place(pos(6, 6), Stone::Black).unwrap();

        let mut state = SearchState::from_board(&board);
        let score_before = state.total_score;

        state.place(pos(8, 8), Stone::White);
        state.undo(pos(8, 8), Stone::White);

        assert_eq!(state.total_score, score_before);
    }
}
