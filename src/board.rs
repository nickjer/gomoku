use std::fmt;

use anyhow::{Result, bail};

use crate::bitboard::BitBoard;
use crate::outcome::Outcome;
use crate::position_id::PositionId;
use crate::stone::Stone;

/// A Gomoku game board.
#[derive(Debug, Clone)]
pub struct Board {
    black: BitBoard,
    white: BitBoard,
    move_count: usize,
    outcome: Option<Outcome>,
}

impl Board {
    #[must_use]
    pub fn new() -> Self {
        Self {
            black: BitBoard::EMPTY,
            white: BitBoard::EMPTY,
            move_count: 0,
            outcome: None,
        }
    }

    #[must_use]
    pub fn stone(&self, position_id: PositionId) -> Option<Stone> {
        if self.black.is_set(position_id) {
            Some(Stone::Black)
        } else if self.white.is_set(position_id) {
            Some(Stone::White)
        } else {
            None
        }
    }

    #[must_use]
    pub fn is_empty(&self, position_id: PositionId) -> bool {
        !self.black.is_set(position_id) && !self.white.is_set(position_id)
    }

    #[must_use]
    pub fn outcome(&self) -> Option<Outcome> {
        self.outcome
    }

    #[must_use]
    pub fn move_count(&self) -> usize {
        self.move_count
    }

    #[must_use]
    pub fn empty_position_ids(&self) -> Vec<PositionId> {
        PositionId::iter()
            .filter(|&pos| self.is_empty(pos))
            .collect()
    }

    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.outcome.is_some()
    }

    #[must_use]
    pub fn is_full(&self) -> bool {
        self.move_count == PositionId::COUNT
    }

    /// Places a stone at the given position.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The game is already finished
    /// - The position is not empty
    pub fn place(&mut self, position_id: PositionId, stone: Stone) -> Result<()> {
        if self.is_finished() {
            bail!("Game is already finished");
        }
        if !self.is_empty(position_id) {
            bail!("Position is not empty");
        }

        self.bitboard_mut(stone).set(position_id);
        self.move_count += 1;

        let winner = self.bitboard(stone).has_five_in_a_row();
        self.outcome = if winner {
            Some(match stone {
                Stone::Black => Outcome::BlackWins,
                Stone::White => Outcome::WhiteWins,
            })
        } else if self.is_full() {
            Some(Outcome::Draw)
        } else {
            None
        };

        Ok(())
    }

    /// Undoes a previously placed stone at the given position.
    pub fn undo(&mut self, position_id: PositionId, stone: Stone) {
        self.bitboard_mut(stone).clear(position_id);
        self.move_count -= 1;
        self.outcome = None;
    }

    /// Returns `true` if placing a stone at the given position would win.
    ///
    /// Does not modify the board — checks a hypothetical placement using
    /// a stack copy of the bitboard.
    #[must_use]
    pub fn would_win(&self, position_id: PositionId, stone: Stone) -> bool {
        let mut bitboard = *self.bitboard(stone);
        bitboard.set(position_id);
        bitboard.has_five_in_a_row()
    }

    pub(crate) fn bitboard(&self, stone: Stone) -> &BitBoard {
        match stone {
            Stone::Black => &self.black,
            Stone::White => &self.white,
        }
    }

    fn bitboard_mut(&mut self, stone: Stone) -> &mut BitBoard {
        match stone {
            Stone::Black => &mut self.black,
            Stone::White => &mut self.white,
        }
    }

    fn stone_char(stone: Option<Stone>) -> char {
        match stone {
            Some(Stone::Black) => 'X',
            Some(Stone::White) => 'O',
            None => '·',
        }
    }
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (row_idx, row) in PositionId::rows().enumerate() {
            if row_idx > 0 {
                writeln!(f)?;
            }
            for (col_idx, position_id) in row.into_iter().enumerate() {
                if col_idx > 0 {
                    write!(f, " ")?;
                }
                write!(f, "{}", Self::stone_char(self.stone(position_id)))?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::offset::Offset;
    use crate::position::Position;

    fn pos(row: usize, col: usize) -> PositionId {
        PositionId::from_position(Position::new(row, col))
    }

    fn place_line(
        board: &mut Board,
        start: PositionId,
        offset: Offset,
        stone: Stone,
        count: usize,
    ) {
        let mut current = start;
        for _ in 0..count {
            board.place(current, stone).unwrap();
            if let Some(next) = current.offset(offset) {
                current = next;
            }
        }
    }

    #[test]
    fn stone_returns_none_initially() {
        let board = Board::new();

        assert_eq!(board.stone(pos(0, 0)), None);
    }

    #[test]
    fn is_empty_returns_true_initially() {
        let board = Board::new();

        assert!(board.is_empty(pos(0, 0)));
    }

    #[test]
    fn place_sets_stone_at_position() {
        let mut board = Board::new();
        let corner = pos(0, 0);
        board.place(corner, Stone::Black).unwrap();

        assert_eq!(board.stone(corner), Some(Stone::Black));
        assert!(!board.is_empty(corner));
    }

    #[test]
    fn place_returns_error_when_position_not_empty() {
        let mut board = Board::new();
        let corner = pos(0, 0);
        board.place(corner, Stone::Black).unwrap();

        let err = board.place(corner, Stone::White).unwrap_err();
        assert_eq!(err.to_string(), "Position is not empty");
    }

    #[test]
    fn display_for_empty_board() {
        let board = Board::new();
        let expected_row = vec!["·"; 15].join(" ");
        let expected = vec![expected_row; 15].join("\n");

        assert_eq!(board.to_string(), expected);
    }

    #[test]
    fn display_shows_placed_stones() {
        let mut board = Board::new();
        let corner = pos(0, 0);
        let right = Offset::new(0, 1);
        board.place(corner, Stone::Black).unwrap();
        board
            .place(corner.offset(right).unwrap(), Stone::White)
            .unwrap();
        let display = board.to_string();

        assert!(display.starts_with("X O"));
    }

    #[test]
    fn outcome_is_none_initially() {
        let board = Board::new();

        assert_eq!(board.outcome(), None);
    }

    #[test]
    fn is_finished_is_false_initially() {
        let board = Board::new();

        assert!(!board.is_finished());
    }

    #[test]
    fn horizontal_win_detection() {
        let mut board = Board::new();
        let right = Offset::new(0, 1);
        place_line(&mut board, pos(0, 0), right, Stone::Black, 5);

        assert!(board.is_finished());
        assert_eq!(board.outcome(), Some(Outcome::BlackWins));
    }

    #[test]
    fn vertical_win_detection() {
        let mut board = Board::new();
        let down = Offset::new(1, 0);
        place_line(&mut board, pos(0, 0), down, Stone::White, 5);

        assert!(board.is_finished());
        assert_eq!(board.outcome(), Some(Outcome::WhiteWins));
    }

    #[test]
    fn diagonal_down_right_win_detection() {
        let mut board = Board::new();
        let down_right = Offset::new(1, 1);
        place_line(&mut board, pos(0, 0), down_right, Stone::Black, 5);

        assert!(board.is_finished());
        assert_eq!(board.outcome(), Some(Outcome::BlackWins));
    }

    #[test]
    fn diagonal_down_left_win_detection() {
        let mut board = Board::new();
        let down_left = Offset::new(1, -1);
        place_line(&mut board, pos(0, 4), down_left, Stone::Black, 5);

        assert!(board.is_finished());
        assert_eq!(board.outcome(), Some(Outcome::BlackWins));
    }

    #[test]
    fn no_win_with_four_in_a_row() {
        let mut board = Board::new();
        let right = Offset::new(0, 1);
        place_line(&mut board, pos(0, 0), right, Stone::Black, 4);

        assert!(!board.is_finished());
        assert_eq!(board.outcome(), None);
    }

    #[test]
    fn returns_error_when_placing_after_win() {
        let mut board = Board::new();
        let right = Offset::new(0, 1);
        let corner = pos(0, 0);
        place_line(&mut board, corner, right, Stone::Black, 5);

        let after_win = pos(0, 5);
        let err = board.place(after_win, Stone::White).unwrap_err();
        assert_eq!(err.to_string(), "Game is already finished");
    }

    #[test]
    fn outcome_is_draw_when_board_full_without_winner() {
        use crate::test_utils::draw_moves;

        let mut board = Board::new();
        for (position, stone) in draw_moves() {
            board.place(position, stone).unwrap();
        }

        assert_eq!(board.outcome(), Some(Outcome::Draw));
    }
}
