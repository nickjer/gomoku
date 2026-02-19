use bitvec::BitArr;
use bitvec::array::BitArray;
use bitvec::order::Lsb0;

use crate::position_id::PositionId;

/// Storage type for bitboard bits, sized to fit `PositionId::COUNT` positions.
type Bits = BitArr!(for PositionId::COUNT, in u64, Lsb0);

/// A fixed-size bitboard for a Gomoku board.
///
/// Wraps a bit array sized to `PositionId::COUNT` positions.
/// Stack-allocated and `Copy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitBoard {
    bits: Bits,
}

impl BitBoard {
    /// Creates an empty bitboard with all bits unset.
    pub const EMPTY: Self = Self {
        bits: BitArray::ZERO,
    };

    /// Returns `true` if the bit at the given position is set.
    #[must_use]
    pub fn is_set(&self, position_id: PositionId) -> bool {
        self.bits[usize::from(position_id)]
    }

    /// Sets the bit at the given position.
    pub fn set(&mut self, position_id: PositionId) {
        self.bits.set(usize::from(position_id), true);
    }
}

impl Default for BitBoard {
    fn default() -> Self {
        Self::EMPTY
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
}
