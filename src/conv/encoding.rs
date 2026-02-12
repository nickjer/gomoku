use crate::board::Board;
use crate::position_id::PositionId;
use crate::position_map::PositionMap;
use crate::stone::Stone;

/// Number of input channels for board encoding (own stones, opponent stones).
pub const INPUT_CHANNELS: usize = 2;

/// Encodes the board from the perspective of the current player.
///
/// Returns a [`PositionMap`] with `INPUT_CHANNELS` channels per position:
/// channel 0 is the current player's stones, channel 1 is the opponent's stones.
/// Values are 1.0 (stone present) or 0.0.
pub fn encode_board(board: &Board, current_stone: Stone) -> PositionMap<f32> {
    debug_assert_ne!(current_stone, Stone::Empty, "current_stone cannot be Empty");

    let mut encoding = PositionMap::new(0.0, INPUT_CHANNELS);

    for pos in PositionId::iter() {
        let stone = board.stone(pos);

        if stone == current_stone {
            encoding.get_mut(pos)[0] = 1.0;
        } else if stone != Stone::Empty {
            encoding.get_mut(pos)[1] = 1.0;
        }
    }

    encoding
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::offset::Offset;
    use crate::position::Position;

    fn pos(row: usize, col: usize) -> PositionId {
        PositionId::from_position(Position::new(row, col))
    }

    fn corner() -> PositionId {
        pos(0, 0)
    }

    fn far_corner() -> PositionId {
        let last = PositionId::WIDTH - 1;
        pos(last, last)
    }

    #[test]
    fn empty_board_encodes_to_all_zeros() {
        let board = Board::new();

        let encoding = encode_board(&board, Stone::Black);

        assert_eq!(
            encoding.as_slice().len(),
            INPUT_CHANNELS * PositionId::COUNT
        );
        assert!(encoding.as_slice().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn own_stone_appears_in_channel_zero() {
        let mut board = Board::new();
        let center = PositionId::center();
        board.place(center, Stone::Black).unwrap();

        let encoding = encode_board(&board, Stone::Black);

        assert_eq!(encoding.get(center), &[1.0, 0.0]);
    }

    #[test]
    fn opponent_stone_appears_in_channel_one() {
        let mut board = Board::new();
        let center = PositionId::center();
        board.place(center, Stone::White).unwrap();

        let encoding = encode_board(&board, Stone::Black);

        assert_eq!(encoding.get(center), &[0.0, 1.0]);
    }

    #[test]
    fn perspective_reversal_swaps_channels() {
        let mut board = Board::new();
        let center = PositionId::center();
        let adjacent = center.from_offset(Offset::new(0, 1)).unwrap();
        board.place(center, Stone::Black).unwrap();
        board.place(adjacent, Stone::White).unwrap();

        let black_view = encode_board(&board, Stone::Black);
        let white_view = encode_board(&board, Stone::White);

        // From Black's perspective: center is own, adjacent is opponent
        assert_eq!(black_view.get(center)[0], 1.0);
        assert_eq!(black_view.get(adjacent)[1], 1.0);

        // From White's perspective: adjacent is own, center is opponent
        assert_eq!(white_view.get(adjacent)[0], 1.0);
        assert_eq!(white_view.get(center)[1], 1.0);
    }

    #[test]
    fn multiple_stones_encoded_correctly() {
        let mut board = Board::new();
        let positions = [corner(), PositionId::center(), far_corner()];
        for &p in &positions {
            board.place(p, Stone::Black).unwrap();
        }

        let encoding = encode_board(&board, Stone::Black);

        // All three positions should be 1.0 in own channel
        for &p in &positions {
            assert_eq!(encoding.get(p)[0], 1.0);
        }

        // Count total positions with own=1.0 should be exactly 3
        let own_count = PositionId::iter()
            .filter(|&p| encoding.get(p)[0] == 1.0)
            .count();
        assert_eq!(own_count, 3);

        // No opponent stones
        let opp_count = PositionId::iter()
            .filter(|&p| encoding.get(p)[1] == 1.0)
            .count();
        assert_eq!(opp_count, 0);
    }
}
