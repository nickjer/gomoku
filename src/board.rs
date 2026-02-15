use std::fmt;

use anyhow::{Result, bail};

use crate::offset::Offset;
use crate::outcome::Outcome;
use crate::position_id::PositionId;
use crate::position_map::PositionMap;
use crate::stone::Stone;

const WIN_LENGTH: usize = 5;
const DIRECTIONS: [Offset; 4] = [
    Offset::new(0, 1),
    Offset::new(1, 0),
    Offset::new(1, 1),
    Offset::new(1, -1),
];

/// A Gomoku game board.
#[derive(Debug, Clone)]
pub struct Board {
    stones: PositionMap<Stone>,
    empty_position_ids: Vec<PositionId>,
    outcome: Option<Outcome>,
}

impl Board {
    #[must_use]
    pub fn new() -> Self {
        Self {
            stones: PositionMap::new(Stone::Empty, 1),
            empty_position_ids: PositionId::iter().collect(),
            outcome: None,
        }
    }

    #[must_use]
    pub fn stone(&self, position_id: PositionId) -> Stone {
        self.stones.get(position_id)[0]
    }

    #[must_use]
    pub fn outcome(&self) -> Option<Outcome> {
        self.outcome
    }

    #[must_use]
    pub fn empty_position_ids(&self) -> &[PositionId] {
        &self.empty_position_ids
    }

    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.outcome.is_some()
    }

    #[must_use]
    pub fn is_full(&self) -> bool {
        self.empty_position_ids.is_empty()
    }

    /// Places a stone at the given position.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The stone is `Empty`
    /// - The game is already finished
    /// - The position is not empty
    pub fn place(&mut self, position_id: PositionId, stone: Stone) -> Result<()> {
        if stone == Stone::Empty {
            bail!("Cannot place empty stone");
        }
        if self.is_finished() {
            bail!("Game is already finished");
        }
        if self.stone(position_id) != Stone::Empty {
            bail!("Position is not empty");
        }

        self.stones.get_mut(position_id)[0] = stone;
        self.empty_position_ids.retain(|&id| id != position_id);

        self.outcome = if self.check_winner(position_id, stone) {
            Some(match stone {
                Stone::Black => Outcome::BlackWins,
                Stone::White => Outcome::WhiteWins,
                Stone::Empty => unreachable!(),
            })
        } else if self.is_full() {
            Some(Outcome::Draw)
        } else {
            None
        };

        Ok(())
    }

    fn check_winner(&self, position_id: PositionId, stone: Stone) -> bool {
        DIRECTIONS.iter().any(|&offset| {
            let count = 1
                + self.count_direction(position_id, offset, stone)
                + self.count_direction(position_id, -offset, stone);
            count >= WIN_LENGTH
        })
    }

    fn count_direction(&self, start_id: PositionId, offset: Offset, stone: Stone) -> usize {
        let mut count = 0;
        let mut current_id = start_id;

        while let Some(next_id) = current_id.offset(offset) {
            if self.stone(next_id) != stone {
                break;
            }
            count += 1;
            current_id = next_id;
        }

        count
    }

    fn stone_char(stone: Stone) -> char {
        match stone {
            Stone::Black => 'X',
            Stone::White => 'O',
            Stone::Empty => '·',
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
    fn stone_returns_empty_initially() {
        let board = Board::new();

        assert_eq!(board.stone(pos(0, 0)), Stone::Empty);
    }

    #[test]
    fn place_sets_stone_at_position() {
        let mut board = Board::new();
        let corner = pos(0, 0);
        board.place(corner, Stone::Black).unwrap();

        assert_eq!(board.stone(corner), Stone::Black);
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
    fn returns_error_when_placing_empty_stone() {
        let mut board = Board::new();

        let err = board.place(pos(0, 0), Stone::Empty).unwrap_err();
        assert_eq!(err.to_string(), "Cannot place empty stone");
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
