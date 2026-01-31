use crate::board::Board;
use crate::position_id::PositionId;
use crate::position_map::PositionSliceMut;
use crate::stone::Stone;

/// Number of input channels for board encoding (own stones, opponent stones).
pub const INPUT_CHANNELS: usize = 2;

/// Encodes the board from the perspective of the current player.
///
/// Returns a flat array of length `2 * PositionId::COUNT` containing:
/// - Channel 0: Current player's stones (1.0 where present, 0.0 elsewhere)
/// - Channel 1: Opponent's stones (1.0 where present, 0.0 elsewhere)
///
/// Layout: `[own_channel..., opponent_channel...]`
pub fn encode_board(board: &Board, current_stone: Stone) -> Vec<f32> {
    debug_assert_ne!(current_stone, Stone::Empty, "current_stone cannot be Empty");

    let mut encoding = vec![0.0f32; INPUT_CHANNELS * PositionId::COUNT];
    let (own_data, opponent_data) = encoding.split_at_mut(PositionId::COUNT);
    let mut own_channel = PositionSliceMut::new(own_data);
    let mut opponent_channel = PositionSliceMut::new(opponent_data);

    for pos in PositionId::iter() {
        let stone = board.stone(pos);

        if stone == current_stone {
            own_channel[pos] = 1.0;
        } else if stone != Stone::Empty {
            opponent_channel[pos] = 1.0;
        }
    }

    encoding
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::offset::Offset;
    use crate::position::Position;
    use crate::position_map::PositionSlice;

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

        assert_eq!(encoding.len(), INPUT_CHANNELS * PositionId::COUNT);
        assert!(encoding.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn own_stone_appears_in_channel_zero() {
        let mut board = Board::new();
        let center = PositionId::center();
        board.place(center, Stone::Black).unwrap();

        let encoding = encode_board(&board, Stone::Black);
        let (own_data, opponent_data) = encoding.split_at(PositionId::COUNT);
        let own_channel = PositionSlice::new(own_data);
        let opponent_channel = PositionSlice::new(opponent_data);

        assert_eq!(own_channel[center], 1.0);
        assert_eq!(opponent_channel[center], 0.0);
    }

    #[test]
    fn opponent_stone_appears_in_channel_one() {
        let mut board = Board::new();
        let center = PositionId::center();
        board.place(center, Stone::White).unwrap();

        let encoding = encode_board(&board, Stone::Black);
        let (own_data, opponent_data) = encoding.split_at(PositionId::COUNT);
        let own_channel = PositionSlice::new(own_data);
        let opponent_channel = PositionSlice::new(opponent_data);

        assert_eq!(own_channel[center], 0.0);
        assert_eq!(opponent_channel[center], 1.0);
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

        let (black_own, black_opp) = black_view.split_at(PositionId::COUNT);
        let black_own = PositionSlice::new(black_own);
        let black_opp = PositionSlice::new(black_opp);

        let (white_own, white_opp) = white_view.split_at(PositionId::COUNT);
        let white_own = PositionSlice::new(white_own);
        let white_opp = PositionSlice::new(white_opp);

        // From Black's perspective: center is own, adjacent is opponent
        assert_eq!(black_own[center], 1.0);
        assert_eq!(black_opp[adjacent], 1.0);

        // From White's perspective: adjacent is own, center is opponent
        assert_eq!(white_own[adjacent], 1.0);
        assert_eq!(white_opp[center], 1.0);
    }

    #[test]
    fn multiple_stones_encoded_correctly() {
        let mut board = Board::new();
        let positions = [corner(), PositionId::center(), far_corner()];
        for &p in &positions {
            board.place(p, Stone::Black).unwrap();
        }

        let encoding = encode_board(&board, Stone::Black);
        let (own_data, opponent_data) = encoding.split_at(PositionId::COUNT);
        let own_channel = PositionSlice::new(own_data);

        // All three positions should be 1.0 in own channel
        for &p in &positions {
            assert_eq!(own_channel[p], 1.0);
        }

        // Count total 1.0s in own channel should be exactly 3
        let own_count = own_data.iter().filter(|&&v| v == 1.0).count();
        assert_eq!(own_count, 3);

        // Opponent channel should be all zeros
        assert!(opponent_data.iter().all(|&v| v == 0.0));
    }
}
