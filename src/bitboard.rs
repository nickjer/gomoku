use std::ops;

use bitvec::BitArr;
use bitvec::array::BitArray;
use bitvec::order::Lsb0;

use crate::position_id::PositionId;

/// Raw backing storage, sized to fit `PositionId::COUNT` bits.
type Storage = [u64; PositionId::COUNT.div_ceil(64)];

/// Bit array type used internally.
type Bits = BitArr!(for PositionId::COUNT, in u64, Lsb0);

const WIDTH: usize = PositionId::WIDTH;

// ── Column masks for wraparound prevention ──────────────────────────────

/// Builds raw storage with all board positions set EXCEPT those in columns
/// whose bit is set in `exclude`. E.g., `exclude = 1 << 14` clears column 14.
const fn build_exclude_columns_mask(exclude: u16) -> Storage {
    let mut result = [0u64; PositionId::COUNT.div_ceil(64)];
    let mut index = 0;
    while index < PositionId::COUNT {
        let col = index % WIDTH;
        if (exclude >> col) & 1 == 0 {
            result[index / 64] |= 1 << (index % 64);
        }
        index += 1;
    }
    result
}

/// All positions set — no column masking (for vertical direction).
const ALL_COLS: BitBoard = BitBoard::from_raw(build_exclude_columns_mask(0));

// Masks for col_delta = +1 (horizontal, diagonal down-right):
// Shift by 1: exclude last column to prevent right-edge wraparound.
const NOT_LAST_COL: BitBoard = BitBoard::from_raw(build_exclude_columns_mask(1 << (WIDTH - 1)));
// Shift by 2: exclude last 2 columns.
const NOT_LAST_2_COLS: BitBoard = BitBoard::from_raw(build_exclude_columns_mask(
    (1 << (WIDTH - 1)) | (1 << (WIDTH - 2)),
));
// Shift by 4: exclude last 4 columns.
const NOT_LAST_4_COLS: BitBoard = BitBoard::from_raw(build_exclude_columns_mask(
    (1 << (WIDTH - 1)) | (1 << (WIDTH - 2)) | (1 << (WIDTH - 3)) | (1 << (WIDTH - 4)),
));

// Masks for col_delta = -1 (diagonal down-left):
// Shift by 1: exclude first column to prevent left-edge wraparound.
const NOT_FIRST_COL: BitBoard = BitBoard::from_raw(build_exclude_columns_mask(1));
// Shift by 2: exclude first 2 columns.
const NOT_FIRST_2_COLS: BitBoard = BitBoard::from_raw(build_exclude_columns_mask(1 | (1 << 1)));
// Shift by 4: exclude first 4 columns.
const NOT_FIRST_4_COLS: BitBoard = BitBoard::from_raw(build_exclude_columns_mask(
    1 | (1 << 1) | (1 << 2) | (1 << 3),
));

// ── BitBoard ────────────────────────────────────────────────────────────

/// A fixed-size bitboard for a Gomoku board.
///
/// Wraps a bit array sized to `PositionId::COUNT` positions.
/// Stack-allocated and `Copy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitBoard {
    bits: Bits,
}

impl BitBoard {
    /// An empty bitboard with all bits unset.
    pub const EMPTY: Self = Self {
        bits: BitArray::ZERO,
    };

    /// Creates a bitboard from raw storage. Uses the same public-field
    /// construction pattern as `BitArray::ZERO`.
    const fn from_raw(data: Storage) -> Self {
        Self {
            bits: BitArray {
                _ord: std::marker::PhantomData,
                data,
            },
        }
    }

    /// Returns `true` if the bit at the given position is set.
    #[must_use]
    pub fn is_set(&self, position_id: PositionId) -> bool {
        self.bits[usize::from(position_id)]
    }

    /// Sets the bit at the given position.
    pub fn set(&mut self, position_id: PositionId) {
        self.bits.set(usize::from(position_id), true);
    }

    /// Returns `true` if any bit is set.
    #[must_use]
    pub fn any(&self) -> bool {
        self.bits.any()
    }

    /// Returns `true` if the bitboard contains five or more consecutive stones
    /// in any direction (horizontal, vertical, diagonal).
    #[must_use]
    pub fn has_five_in_a_row(self) -> bool {
        // Horizontal: stride 1, col delta +1
        check_direction(self, 1, NOT_LAST_COL, NOT_LAST_2_COLS, NOT_LAST_4_COLS)
        // Vertical: stride WIDTH, col delta 0 (no wraparound possible)
        || check_direction(self, WIDTH, ALL_COLS, ALL_COLS, ALL_COLS)
        // Diagonal down-right: stride WIDTH+1, col delta +1
        || check_direction(self, WIDTH + 1, NOT_LAST_COL, NOT_LAST_2_COLS, NOT_LAST_4_COLS)
        // Diagonal down-left: stride WIDTH-1, col delta -1
        || check_direction(self, WIDTH - 1, NOT_FIRST_COL, NOT_FIRST_2_COLS, NOT_FIRST_4_COLS)
    }
}

/// Detects five or more consecutive stones in one direction using shift-AND.
///
/// For stride `s`, the masks prevent row-wraparound after shifts:
/// - `mask1` applied after shifting by `s`   (single step)
/// - `mask2` applied after shifting by `2s`  (double step)
/// - `mask4` applied after shifting by `4s`  (quadruple step)
fn check_direction(
    board: BitBoard,
    stride: usize,
    mask1: BitBoard,
    mask2: BitBoard,
    mask4: BitBoard,
) -> bool {
    let pairs = board & ((board >> stride) & mask1);
    let quads = pairs & ((pairs >> (2 * stride)) & mask2);
    let fives = quads & ((board >> (4 * stride)) & mask4);
    fives.any()
}

impl Default for BitBoard {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl ops::Shr<usize> for BitBoard {
    type Output = Self;

    fn shr(mut self, amount: usize) -> Self {
        // bitvec uses positional naming: `shift_left` moves bits toward
        // lower indices, which matches arithmetic right-shift semantics.
        self.bits.shift_left(amount);
        self
    }
}

impl ops::BitAnd for BitBoard {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self {
        Self {
            bits: self.bits & rhs.bits,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;

    fn pos(row: usize, col: usize) -> PositionId {
        PositionId::from_position(Position::new(row, col))
    }

    fn bitboard_from_positions(positions: &[(usize, usize)]) -> BitBoard {
        let mut board = BitBoard::EMPTY;
        for &(row, col) in positions {
            board.set(pos(row, col));
        }
        board
    }

    // ── Basic operations ────────────────────────────────────────────

    #[test]
    fn empty_bitboard_has_no_bits_set() {
        let board = BitBoard::EMPTY;

        for position in PositionId::iter() {
            assert!(!board.is_set(position));
        }
    }

    #[test]
    fn set_and_query_single_bit() {
        let mut board = BitBoard::EMPTY;
        let center = PositionId::center();
        board.set(center);

        assert!(board.is_set(center));
    }

    #[test]
    fn unset_position_returns_false() {
        let board = BitBoard::EMPTY;

        assert!(!board.is_set(pos(0, 0)));
    }

    #[test]
    fn set_multiple_bits() {
        let mut board = BitBoard::EMPTY;
        board.set(pos(0, 0));
        board.set(pos(7, 7));
        board.set(pos(14, 14));

        assert!(board.is_set(pos(0, 0)));
        assert!(board.is_set(pos(7, 7)));
        assert!(board.is_set(pos(14, 14)));
        assert!(!board.is_set(pos(0, 1)));
    }

    #[test]
    fn default_is_empty() {
        let board = BitBoard::default();

        assert_eq!(board, BitBoard::EMPTY);
    }

    #[test]
    fn any_returns_false_for_empty() {
        assert!(!BitBoard::EMPTY.any());
    }

    #[test]
    fn any_returns_true_when_bit_set() {
        let mut board = BitBoard::EMPTY;
        board.set(pos(3, 7));

        assert!(board.any());
    }

    #[test]
    fn shr_shifts_bit_to_lower_index() {
        let mut board = BitBoard::EMPTY;
        board.set(pos(0, 5));

        let shifted = board >> 1;

        assert!(shifted.is_set(pos(0, 4)));
        assert!(!shifted.is_set(pos(0, 5)));
    }

    #[test]
    fn bitand_keeps_common_bits() {
        let mut left = BitBoard::EMPTY;
        left.set(pos(0, 0));
        left.set(pos(0, 1));

        let mut right = BitBoard::EMPTY;
        right.set(pos(0, 1));
        right.set(pos(0, 2));

        let result = left & right;

        assert!(!result.is_set(pos(0, 0)));
        assert!(result.is_set(pos(0, 1)));
        assert!(!result.is_set(pos(0, 2)));
    }

    // ── has_five_in_a_row tests ─────────────────────────────────────

    #[test]
    fn horizontal_five_detected() {
        let board = bitboard_from_positions(&[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]);

        assert!(board.has_five_in_a_row());
    }

    #[test]
    fn vertical_five_detected() {
        let board = bitboard_from_positions(&[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]);

        assert!(board.has_five_in_a_row());
    }

    #[test]
    fn diagonal_down_right_five_detected() {
        let board = bitboard_from_positions(&[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]);

        assert!(board.has_five_in_a_row());
    }

    #[test]
    fn diagonal_down_left_five_detected() {
        let board = bitboard_from_positions(&[(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)]);

        assert!(board.has_five_in_a_row());
    }

    #[test]
    fn four_in_a_row_not_detected() {
        let board = bitboard_from_positions(&[(0, 0), (0, 1), (0, 2), (0, 3)]);

        assert!(!board.has_five_in_a_row());
    }

    #[test]
    fn six_in_a_row_detected() {
        let board = bitboard_from_positions(&[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]);

        assert!(board.has_five_in_a_row());
    }

    #[test]
    fn horizontal_five_at_right_edge() {
        let board = bitboard_from_positions(&[(0, 10), (0, 11), (0, 12), (0, 13), (0, 14)]);

        assert!(board.has_five_in_a_row());
    }

    #[test]
    fn vertical_five_at_bottom_edge() {
        let board = bitboard_from_positions(&[(10, 0), (11, 0), (12, 0), (13, 0), (14, 0)]);

        assert!(board.has_five_in_a_row());
    }

    #[test]
    fn diagonal_five_at_bottom_right_corner() {
        let board = bitboard_from_positions(&[(10, 10), (11, 11), (12, 12), (13, 13), (14, 14)]);

        assert!(board.has_five_in_a_row());
    }

    #[test]
    fn no_false_horizontal_across_rows() {
        let board = bitboard_from_positions(&[(0, 13), (0, 14), (1, 0), (1, 1), (1, 2)]);

        assert!(!board.has_five_in_a_row());
    }

    #[test]
    fn no_false_diagonal_across_rows() {
        let board = bitboard_from_positions(&[(0, 12), (0, 13), (0, 14), (1, 0), (1, 1)]);

        assert!(!board.has_five_in_a_row());
    }

    #[test]
    fn empty_board_has_no_five() {
        assert!(!BitBoard::EMPTY.has_five_in_a_row());
    }
}
