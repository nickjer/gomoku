use crate::board::Board;
use crate::outcome::Outcome;
use crate::position_id::PositionId;
use crate::stone::Stone;

use super::lines::{LINE_LENGTHS, NUM_LINES, POSITION_LINES, score_line};
use super::score::Score;

/// Saved state for a single place operation so `undo` can restore in O(1).
///
/// Line indices are not stored — they are derived from the position via
/// `POSITION_LINES` when `undo` is called.
struct UndoFrame {
    scores: [Score; 4],
    total: Score,
}

/// Board wrapper that maintains incremental line-based evaluation scores.
///
/// Instead of recomputing full-board pattern counts at every leaf node,
/// `SearchState` tracks u16 bit masks for each of the 88 board lines and
/// updates only the 4 affected lines per place/undo. Leaf evaluation is O(1).
///
/// Undo is O(1): `place` saves the 4 affected line scores onto a stack, and
/// `undo` restores them without recomputing `score_line`.
pub struct SearchState {
    board: Board,
    line_black: [u16; NUM_LINES],
    line_white: [u16; NUM_LINES],
    line_scores: [Score; NUM_LINES],
    total_score: Score,
    outcome: Option<Outcome>,
    undo_stack: Vec<UndoFrame>,
}

/// Returns `true` if a u16 line mask contains 5 or more consecutive set bits.
fn has_five_consecutive(mask: u16) -> bool {
    let c2 = mask & (mask >> 1);
    let c4 = c2 & (c2 >> 2);
    let c5 = c4 & (mask >> 4);
    c5 != 0
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
            outcome: board.outcome(),
            undo_stack: Vec::new(),
        }
    }

    /// Places a stone and incrementally updates the 4 affected line scores.
    ///
    /// Uses `Board::place_unchecked` to skip validation and the expensive
    /// `has_five_in_a_row` bitboard scan. Wins are detected cheaply by
    /// checking 5-consecutive bits on the placed stone's line masks.
    ///
    /// Saves old state onto an internal stack so that [`undo`](Self::undo)
    /// can restore in O(1).
    pub fn place(&mut self, position: PositionId, stone: Stone) {
        self.board.place_unchecked(position, stone);

        let lines = POSITION_LINES.get(position);
        let old_total = self.total_score;
        let mut old_scores = [Score::DRAW; 4];
        let mut won = false;

        for (i, &(line_id, bit)) in lines.iter().enumerate() {
            let idx = usize::from(line_id);
            old_scores[i] = self.line_scores[idx];
            self.total_score -= self.line_scores[idx];

            let own_mask = match stone {
                Stone::Black => {
                    self.line_black[idx] |= 1 << bit;
                    self.line_black[idx]
                }
                Stone::White => {
                    self.line_white[idx] |= 1 << bit;
                    self.line_white[idx]
                }
            };
            won = won || has_five_consecutive(own_mask);

            self.line_scores[idx] = score_line(
                self.line_black[idx],
                self.line_white[idx],
                LINE_LENGTHS[idx],
            );
            self.total_score += self.line_scores[idx];
        }

        if won {
            self.outcome = Some(match stone {
                Stone::Black => Outcome::BlackWins,
                Stone::White => Outcome::WhiteWins,
            });
        } else if self.board.is_full() {
            self.outcome = Some(Outcome::Draw);
        }

        self.undo_stack.push(UndoFrame {
            scores: old_scores,
            total: old_total,
        });
    }

    /// Undoes a stone placement by restoring saved line scores from the stack.
    ///
    /// O(1) — no `score_line` recomputation needed.
    pub fn undo(&mut self, position: PositionId, stone: Stone) {
        self.board.undo(position, stone);

        let frame = self.undo_stack.pop().expect("undo without matching place");

        // Restore bit masks and saved line scores for the 4 affected lines.
        for (i, &(line_id, bit)) in POSITION_LINES.get(position).iter().enumerate() {
            let idx = usize::from(line_id);
            match stone {
                Stone::Black => self.line_black[idx] &= !(1 << bit),
                Stone::White => self.line_white[idx] &= !(1 << bit),
            }
            self.line_scores[idx] = frame.scores[i];
        }
        self.total_score = frame.total;
        self.outcome = None;
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
        self.outcome
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

    #[test]
    fn undo_restores_line_scores_and_masks() {
        let mut rng = fastrand::Rng::with_seed(12345);
        for _ in 0..200 {
            let mut board = Board::new();
            let mut state = SearchState::from_board(&board);
            let setup_count = rng.usize(0..15);
            let mut positions: Vec<PositionId> = PositionId::iter().collect();
            rng.shuffle(&mut positions);
            let mut turn = Stone::Black;

            for &position in positions.iter().take(setup_count) {
                if board.is_finished() {
                    break;
                }
                board.place(position, turn).unwrap();
                state.place(position, turn);
                turn = turn.opponent();
            }

            if board.is_finished() {
                continue;
            }

            let scores_before = state.line_scores;
            let black_before = state.line_black;
            let white_before = state.line_white;
            let total_before = state.total_score;

            let empty: Vec<PositionId> =
                PositionId::iter().filter(|&p| board.is_empty(p)).collect();
            let target = empty[rng.usize(0..empty.len())];

            state.place(target, turn);
            state.undo(target, turn);

            assert_eq!(state.total_score, total_before, "total_score mismatch");
            assert_eq!(state.line_scores, scores_before, "line_scores mismatch");
            assert_eq!(state.line_black, black_before, "line_black mismatch");
            assert_eq!(state.line_white, white_before, "line_white mismatch");
        }
    }

    #[test]
    fn nested_place_undo_restores_correctly() {
        let mut board = Board::new();
        board.place(pos(7, 7), Stone::Black).unwrap();

        let mut state = SearchState::from_board(&board);
        let score_0 = state.total_score;

        state.place(pos(7, 8), Stone::White);
        let score_1 = state.total_score;

        state.place(pos(6, 6), Stone::Black);

        state.undo(pos(6, 6), Stone::Black);
        assert_eq!(state.total_score, score_1, "after undoing second place");

        state.undo(pos(7, 8), Stone::White);
        assert_eq!(state.total_score, score_0, "after undoing first place");
    }

    #[test]
    fn detects_black_win() {
        let mut board = Board::new();
        board.place(pos(7, 3), Stone::Black).unwrap();
        board.place(pos(7, 4), Stone::Black).unwrap();
        board.place(pos(7, 5), Stone::Black).unwrap();
        board.place(pos(7, 6), Stone::Black).unwrap();
        // White stones to keep alternating turns valid for Board
        board.place(pos(0, 0), Stone::White).unwrap();
        board.place(pos(0, 1), Stone::White).unwrap();

        let mut state = SearchState::from_board(&board);
        assert!(state.outcome().is_none());

        state.place(pos(7, 7), Stone::Black);

        assert_eq!(state.outcome(), Some(Outcome::BlackWins));
    }

    #[test]
    fn detects_white_win() {
        let mut board = Board::new();
        board.place(pos(3, 7), Stone::White).unwrap();
        board.place(pos(4, 7), Stone::White).unwrap();
        board.place(pos(5, 7), Stone::White).unwrap();
        board.place(pos(6, 7), Stone::White).unwrap();
        board.place(pos(0, 0), Stone::Black).unwrap();

        let mut state = SearchState::from_board(&board);
        assert!(state.outcome().is_none());

        state.place(pos(7, 7), Stone::White);

        assert_eq!(state.outcome(), Some(Outcome::WhiteWins));
    }

    #[test]
    fn undo_clears_win_outcome() {
        let mut board = Board::new();
        board.place(pos(7, 3), Stone::Black).unwrap();
        board.place(pos(7, 4), Stone::Black).unwrap();
        board.place(pos(7, 5), Stone::Black).unwrap();
        board.place(pos(7, 6), Stone::Black).unwrap();
        board.place(pos(0, 0), Stone::White).unwrap();

        let mut state = SearchState::from_board(&board);

        state.place(pos(7, 7), Stone::Black);
        assert_eq!(state.outcome(), Some(Outcome::BlackWins));

        state.undo(pos(7, 7), Stone::Black);
        assert!(state.outcome().is_none());
    }

    #[test]
    fn outcome_agrees_with_board_across_random_games() {
        let mut rng = fastrand::Rng::with_seed(99999);
        for _ in 0..200 {
            let mut board = Board::new();
            let mut state = SearchState::from_board(&board);
            let mut positions: Vec<PositionId> = PositionId::iter().collect();
            rng.shuffle(&mut positions);
            let mut turn = Stone::Black;

            for &position in &positions {
                if board.is_finished() {
                    break;
                }
                board.place(position, turn).unwrap();
                state.place(position, turn);

                assert_eq!(
                    state.outcome(),
                    board.outcome(),
                    "outcome mismatch after placing at ({}, {})",
                    position.row(),
                    position.col()
                );
                turn = turn.opponent();
            }
        }
    }
}
