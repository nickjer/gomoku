use crate::board::Board;
use crate::offset::Offset;
use crate::position_id::PositionId;
use crate::position_map::PositionArray;
use crate::stone::Stone;

use super::evaluate::{Score, evaluate};

pub const DEFAULT_DEPTH: u32 = 4;
const PROXIMITY_RADIUS: isize = 2;

type CandidateBuf = [PositionId; PositionId::COUNT];

/// Generates candidate moves ordered by priority: winning, blocking, then proximity.
///
/// Returns the number of candidates written to `buf`.
fn generate_candidates(board: &Board, stone: Stone, buf: &mut CandidateBuf) -> usize {
    if board.move_count() == 0 {
        buf[0] = PositionId::center();
        return 1;
    }

    let mut nearby = PositionArray::new(false);
    for position in PositionId::iter() {
        if !board.is_empty(position) {
            for row_delta in -PROXIMITY_RADIUS..=PROXIMITY_RADIUS {
                for col_delta in -PROXIMITY_RADIUS..=PROXIMITY_RADIUS {
                    if row_delta == 0 && col_delta == 0 {
                        continue;
                    }
                    if let Some(neighbor) = position.offset(Offset::new(row_delta, col_delta))
                        && board.is_empty(neighbor)
                    {
                        nearby[neighbor] = true;
                    }
                }
            }
        }
    }

    let mut winning_end = 0;
    let mut blocking_end = 0;
    let mut count = 0;

    for (position, &is_nearby) in nearby.iter() {
        if !is_nearby {
            continue;
        }
        if board.would_win(position, stone) {
            buf.copy_within(winning_end..count, winning_end + 1);
            buf[winning_end] = position;
            winning_end += 1;
            blocking_end += 1;
            count += 1;
        } else if board.would_win(position, stone.opponent()) {
            buf.copy_within(blocking_end..count, blocking_end + 1);
            buf[blocking_end] = position;
            blocking_end += 1;
            count += 1;
        } else {
            buf[count] = position;
            count += 1;
        }
    }

    count
}

/// Finds the best move for `stone` using negamax with alpha-beta pruning.
pub fn find_best_move(board: &mut Board, stone: Stone, depth: u32) -> PositionId {
    assert!(depth > 0, "find_best_move called with depth 0");

    let mut buf = [PositionId::default(); PositionId::COUNT];
    let count = generate_candidates(board, stone, &mut buf);
    let candidates = &buf[..count];
    assert!(
        !candidates.is_empty(),
        "find_best_move called with no candidates"
    );

    let mut best_move = candidates[0];
    let mut alpha = Score::MIN;
    let beta = Score::MAX;

    for &candidate in candidates {
        board.place(candidate, stone).expect("valid search move");

        let score = if board.is_finished() {
            Score::win_at_depth(board.move_count())
        } else {
            -negamax(board, depth - 1, -beta, -alpha, stone.opponent())
        };

        board.undo(candidate, stone);

        if score > alpha {
            alpha = score;
            best_move = candidate;
        }
    }

    best_move
}

fn negamax(board: &mut Board, depth: u32, mut alpha: Score, beta: Score, stone: Stone) -> Score {
    if depth == 0 || board.is_full() {
        return evaluate(board, stone);
    }

    let mut buf = [PositionId::default(); PositionId::COUNT];
    let count = generate_candidates(board, stone, &mut buf);
    let candidates = &buf[..count];
    if candidates.is_empty() {
        return evaluate(board, stone);
    }

    for &candidate in candidates {
        board.place(candidate, stone).expect("valid search move");

        let score = if board.is_finished() {
            Score::win_at_depth(board.move_count())
        } else {
            -negamax(board, depth - 1, -beta, -alpha, stone.opponent())
        };

        board.undo(candidate, stone);

        if score > alpha {
            alpha = score;
        }
        if alpha >= beta {
            break;
        }
    }

    alpha
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;

    fn pos(row: usize, col: usize) -> PositionId {
        PositionId::from_position(Position::new(row, col))
    }

    fn place_stones(board: &mut Board, stone: Stone, positions: &[(usize, usize)]) {
        for &(row, col) in positions {
            board.place(pos(row, col), stone).unwrap();
        }
    }

    #[test]
    fn returns_center_on_empty_board() {
        let mut board = Board::new();

        let result = find_best_move(&mut board, Stone::Black, 4);

        assert_eq!(result, PositionId::center());
    }

    #[test]
    fn finds_immediate_winning_move() {
        let mut board = Board::new();
        place_stones(&mut board, Stone::Black, &[(7, 5), (7, 6), (7, 7), (7, 8)]);
        place_stones(&mut board, Stone::White, &[(8, 5), (8, 6), (8, 7)]);

        let result = find_best_move(&mut board, Stone::Black, 4);

        assert!(
            result == pos(7, 4) || result == pos(7, 9),
            "Expected winning move at (7,4) or (7,9), got ({}, {})",
            result.row(),
            result.col()
        );
    }

    #[test]
    fn blocks_opponent_winning_move() {
        let mut board = Board::new();
        place_stones(&mut board, Stone::White, &[(7, 5), (7, 6), (7, 7), (7, 8)]);
        place_stones(&mut board, Stone::Black, &[(8, 5), (8, 6), (8, 7)]);

        let result = find_best_move(&mut board, Stone::Black, 4);

        assert!(
            result == pos(7, 4) || result == pos(7, 9),
            "Expected blocking move at (7,4) or (7,9), got ({}, {})",
            result.row(),
            result.col()
        );
    }

    #[test]
    fn prefers_faster_win() {
        let mut board = Board::new();
        place_stones(&mut board, Stone::Black, &[(7, 5), (7, 6), (7, 7), (7, 8)]);
        place_stones(&mut board, Stone::White, &[(8, 5), (8, 6), (8, 7)]);

        let result = find_best_move(&mut board, Stone::Black, 4);

        assert!(
            result == pos(7, 4) || result == pos(7, 9),
            "Expected immediate win, got ({}, {})",
            result.row(),
            result.col()
        );
    }

    #[test]
    fn generates_candidates_near_stones() {
        let mut board = Board::new();
        board.place(PositionId::center(), Stone::Black).unwrap();

        let mut buf = [PositionId::default(); PositionId::COUNT];
        let count = generate_candidates(&board, Stone::White, &mut buf);

        assert!(count > 0);
        for &candidate in &buf[..count] {
            let row_dist = candidate.row().abs_diff(PositionId::center().row());
            let col_dist = candidate.col().abs_diff(PositionId::center().col());
            assert!(
                row_dist <= PROXIMITY_RADIUS as usize && col_dist <= PROXIMITY_RADIUS as usize,
                "Candidate ({}, {}) is too far from center",
                candidate.row(),
                candidate.col()
            );
        }
    }

    #[test]
    fn winning_candidates_appear_first() {
        let mut board = Board::new();
        place_stones(&mut board, Stone::Black, &[(7, 5), (7, 6), (7, 7), (7, 8)]);
        place_stones(&mut board, Stone::White, &[(8, 5), (8, 6)]);

        let mut buf = [PositionId::default(); PositionId::COUNT];
        let count = generate_candidates(&board, Stone::Black, &mut buf);

        assert!(
            board.would_win(buf[0], Stone::Black),
            "First candidate should be a winning move"
        );

        let winning_count = buf[..count]
            .iter()
            .filter(|&&position| board.would_win(position, Stone::Black))
            .count();
        for &candidate in &buf[..winning_count] {
            assert!(board.would_win(candidate, Stone::Black));
        }
    }

    #[test]
    fn blocking_candidates_appear_before_regular() {
        let mut board = Board::new();
        place_stones(&mut board, Stone::White, &[(7, 5), (7, 6), (7, 7), (7, 8)]);
        place_stones(&mut board, Stone::Black, &[(8, 5), (8, 6)]);

        let mut buf = [PositionId::default(); PositionId::COUNT];
        let count = generate_candidates(&board, Stone::Black, &mut buf);
        let candidates = &buf[..count];

        let first_regular = candidates.iter().position(|&candidate| {
            !board.would_win(candidate, Stone::White) && !board.would_win(candidate, Stone::Black)
        });
        let last_blocking = candidates
            .iter()
            .rposition(|&candidate| board.would_win(candidate, Stone::White));

        if let (Some(first_reg), Some(last_blk)) = (first_regular, last_blocking) {
            assert!(
                last_blk < first_reg,
                "Blocking candidates should appear before regular candidates"
            );
        }
    }
}
