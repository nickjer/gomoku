use crate::board::Board;
use crate::outcome::Outcome;
use crate::position_id::PositionId;
use crate::stone::Stone;

use super::lines::{LINE_LENGTHS, NUM_LINES, POSITION_LINES, score_line};
use super::score::Score;
use super::tt::ZOBRIST;

const _: () = assert!(
    NUM_LINES <= 128,
    "the per-line threat flags use one u128 bit per line"
);

/// Saved state for a single place operation so `undo` can restore in O(1).
///
/// Line indices are not stored — they are derived from the position via
/// `POSITION_LINES` when `undo` is called.
struct UndoFrame {
    scores: [Score; 4],
    total: Score,
    lines_with_four: [u128; 2],
    lines_with_open_three: [u128; 2],
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
    hash: u64,
    board: Board,
    line_black: [u16; NUM_LINES],
    line_white: [u16; NUM_LINES],
    line_scores: [Score; NUM_LINES],
    total_score: Score,
    /// One bit per line, set when that line holds a four for Black (index 0)
    /// or White (index 1). Lets the search ask "is there a four on the board"
    /// in O(1).
    lines_with_four: [u128; 2],
    /// Same layout for open threes, so `evaluate` can tell who is about to
    /// make an open four.
    lines_with_open_three: [u128; 2],
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
        let mut lines_with_four = [0u128; 2];
        let mut lines_with_open_three = [0u128; 2];
        for i in 0..NUM_LINES {
            let line = score_line(line_black[i], line_white[i], LINE_LENGTHS[i]);
            line_scores[i] = line.score;
            total_score += line.score;
            for side in [Stone::Black, Stone::White].map(usize::from) {
                if line.has_four[side] {
                    lines_with_four[side] |= 1 << i;
                }
                if line.has_open_three[side] {
                    lines_with_open_three[side] |= 1 << i;
                }
            }
        }

        let zobrist = &*ZOBRIST;
        let mut hash = 0u64;
        for position in PositionId::iter() {
            if let Some(stone) = board.stone(position) {
                hash ^= zobrist[usize::from(position)][usize::from(stone)];
            }
        }

        Self {
            hash,
            board: *board,
            line_black,
            line_white,
            line_scores,
            total_score,
            lines_with_four,
            lines_with_open_three,
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
        self.hash ^= ZOBRIST[usize::from(position)][usize::from(stone)];

        let lines = POSITION_LINES.get(position);
        let old_total = self.total_score;
        let old_lines_with_four = self.lines_with_four;
        let old_lines_with_open_three = self.lines_with_open_three;
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

            let line = score_line(
                self.line_black[idx],
                self.line_white[idx],
                LINE_LENGTHS[idx],
            );
            self.line_scores[idx] = line.score;
            self.total_score += line.score;
            for side in [Stone::Black, Stone::White].map(usize::from) {
                if line.has_four[side] {
                    self.lines_with_four[side] |= 1 << idx;
                } else {
                    self.lines_with_four[side] &= !(1 << idx);
                }
                if line.has_open_three[side] {
                    self.lines_with_open_three[side] |= 1 << idx;
                } else {
                    self.lines_with_open_three[side] &= !(1 << idx);
                }
            }
        }

        if won {
            self.outcome = Some(Outcome::Win(stone));
        } else if self.board.is_full() {
            self.outcome = Some(Outcome::Draw);
        }

        self.undo_stack.push(UndoFrame {
            scores: old_scores,
            total: old_total,
            lines_with_four: old_lines_with_four,
            lines_with_open_three: old_lines_with_open_three,
        });
    }

    /// Undoes a stone placement by restoring saved line scores from the stack.
    ///
    /// O(1) — no `score_line` recomputation needed.
    pub fn undo(&mut self, position: PositionId, stone: Stone) {
        self.board.undo(position, stone);
        self.hash ^= ZOBRIST[usize::from(position)][usize::from(stone)];

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
        self.lines_with_four = frame.lines_with_four;
        self.lines_with_open_three = frame.lines_with_open_three;
        self.outcome = None;
    }

    /// O(1) evaluation from the perspective of `to_move`, the side about to play.
    ///
    /// Starts from the running line total, then credits the side to move for
    /// acting first: its own four completes five next move; its own open three
    /// becomes an open four that no reply stops; two open threes against it
    /// can only be blocked one at a time. A four against it is left to the
    /// search, which always plays the block out rather than scoring here.
    #[must_use]
    pub fn evaluate(&self, to_move: Stone) -> Score {
        let opponent = to_move.opponent();
        let base = match to_move {
            Stone::Black => self.total_score,
            Stone::White => -self.total_score,
        };

        if self.has_four(to_move) {
            return Score::win_at_depth(self.move_count() + 1);
        }
        if self.has_four(opponent) {
            return base;
        }
        if self.lines_with_open_three[usize::from(to_move)] != 0 {
            return base + Score::OPEN_FOUR;
        }
        if self.lines_with_open_three[usize::from(opponent)].count_ones() >= 2 {
            return base - Score::OPEN_FOUR;
        }
        base
    }

    /// Whether `stone` has a four anywhere on the board: a line where one
    /// more stone of that colour completes five.
    #[must_use]
    pub fn has_four(&self, stone: Stone) -> bool {
        self.lines_with_four[usize::from(stone)] != 0
    }

    #[must_use]
    pub fn hash(&self) -> u64 {
        self.hash
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
                assert_eq!(
                    state.lines_with_four,
                    expected.lines_with_four,
                    "Four tracking mismatch after placing at ({}, {}):\n{board}",
                    position.row(),
                    position.col()
                );
                assert_eq!(
                    state.lines_with_open_three,
                    expected.lines_with_open_three,
                    "Open three tracking mismatch after placing at ({}, {}):\n{board}",
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
            let fours_before = state.lines_with_four;
            let open_threes_before = state.lines_with_open_three;

            let empty: Vec<PositionId> =
                PositionId::iter().filter(|&p| board.is_empty(p)).collect();
            let target = empty[rng.usize(0..empty.len())];

            state.place(target, turn);
            state.undo(target, turn);

            assert_eq!(state.total_score, total_before, "total_score mismatch");
            assert_eq!(state.line_scores, scores_before, "line_scores mismatch");
            assert_eq!(state.line_black, black_before, "line_black mismatch");
            assert_eq!(state.line_white, white_before, "line_white mismatch");
            assert_eq!(
                state.lines_with_four, fours_before,
                "lines_with_four mismatch"
            );
            assert_eq!(
                state.lines_with_open_three, open_threes_before,
                "lines_with_open_three mismatch"
            );
        }
    }

    #[test]
    fn evaluate_credits_the_side_to_move_for_its_own_open_three() {
        let mut board = Board::new();
        board.place(pos(7, 6), Stone::Black).unwrap();
        board.place(pos(7, 7), Stone::Black).unwrap();
        board.place(pos(7, 8), Stone::Black).unwrap();
        board.place(pos(0, 0), Stone::White).unwrap();
        board.place(pos(14, 14), Stone::White).unwrap();
        let state = SearchState::from_board(&board);

        // Black to move makes an open four next; White to move can still block.
        assert_eq!(
            state.evaluate(Stone::Black),
            Score::OPEN_THREE + Score::OPEN_FOUR
        );
        assert_eq!(state.evaluate(Stone::White), -Score::OPEN_THREE);
    }

    #[test]
    fn evaluate_treats_two_open_threes_against_the_side_to_move_as_decisive() {
        let mut board = Board::new();
        // Row 7 and column 3, far enough apart to share no line.
        for &(row, col) in &[(7, 6), (7, 7), (7, 8), (3, 3), (4, 3), (5, 3)] {
            board.place(pos(row, col), Stone::Black).unwrap();
        }
        board.place(pos(0, 14), Stone::White).unwrap();
        let state = SearchState::from_board(&board);

        assert_eq!(
            state.evaluate(Stone::White),
            -(Score::OPEN_THREE * 2) - Score::OPEN_FOUR
        );
    }

    #[test]
    fn evaluate_scores_a_four_for_the_side_to_move_as_a_win_next_move() {
        let mut board = Board::new();
        for &(row, col) in &[(7, 6), (7, 7), (7, 8), (7, 9)] {
            board.place(pos(row, col), Stone::Black).unwrap();
        }
        board.place(pos(7, 5), Stone::White).unwrap();
        let state = SearchState::from_board(&board);

        assert_eq!(
            state.evaluate(Stone::Black),
            Score::win_at_depth(board.move_count() + 1)
        );
        // White to move has to block; the plain total stands.
        assert_eq!(state.evaluate(Stone::White), -Score::HALF_OPEN_FOUR);
    }

    #[test]
    fn evaluate_does_not_credit_an_open_three_against_a_four() {
        let mut board = Board::new();
        for &(row, col) in &[(7, 6), (7, 7), (7, 8)] {
            board.place(pos(row, col), Stone::Black).unwrap();
        }
        for &(row, col) in &[(3, 2), (3, 3), (3, 4), (3, 5)] {
            board.place(pos(row, col), Stone::White).unwrap();
        }
        board.place(pos(3, 1), Stone::Black).unwrap();
        let state = SearchState::from_board(&board);

        // Black's open three must wait: White's OOOO_ has to be blocked first.
        assert_eq!(
            state.evaluate(Stone::Black),
            Score::OPEN_THREE - Score::HALF_OPEN_FOUR
        );
    }

    #[test]
    fn has_four_follows_the_four_through_place_block_and_undo() {
        let mut board = Board::new();
        board.place(pos(7, 3), Stone::Black).unwrap();
        board.place(pos(7, 4), Stone::Black).unwrap();
        board.place(pos(7, 5), Stone::Black).unwrap();
        board.place(pos(7, 2), Stone::White).unwrap();

        let mut state = SearchState::from_board(&board);
        assert!(!state.has_four(Stone::Black));
        assert!(!state.has_four(Stone::White));

        // OXXXX_ is a four for Black.
        state.place(pos(7, 6), Stone::Black);
        assert!(state.has_four(Stone::Black));
        assert!(!state.has_four(Stone::White));

        // OXXXXO is dead: no longer a four.
        state.place(pos(7, 7), Stone::White);
        assert!(!state.has_four(Stone::Black));

        state.undo(pos(7, 7), Stone::White);
        assert!(state.has_four(Stone::Black));

        state.undo(pos(7, 6), Stone::Black);
        assert!(!state.has_four(Stone::Black));
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

        assert_eq!(state.outcome(), Some(Outcome::Win(Stone::Black)));
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

        assert_eq!(state.outcome(), Some(Outcome::Win(Stone::White)));
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
        assert_eq!(state.outcome(), Some(Outcome::Win(Stone::Black)));

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
