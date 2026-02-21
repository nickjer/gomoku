use crate::bitboard::winning_threats;
use crate::board::Board;
use crate::offset::Offset;
use crate::outcome::Outcome;
use crate::position_id::PositionId;
use crate::position_map::PositionArray;
use crate::stone::Stone;

use super::evaluate::{Score, evaluate};

pub const DEFAULT_DEPTH: u32 = 4;
const PROXIMITY_RADIUS: isize = 2;

type CandidateBuf = [PositionId; PositionId::COUNT];

/// Generates candidate moves ordered by priority: winning, blocking, then proximity.
///
/// Returns `(blocking_count, total_count)` — the number of blocking moves at the
/// front of `buf` and the total number of candidates written.
fn generate_candidates(board: &Board, stone: Stone, buf: &mut CandidateBuf) -> (usize, usize) {
    if board.move_count() == 0 {
        buf[0] = PositionId::center();
        return (0, 1);
    }

    // Batch-compute winning threat masks for both players.
    let own_threats = winning_threats(*board.bitboard(stone));
    let opp_threats = winning_threats(*board.bitboard(stone.opponent()));

    let occupied = *board.bitboard(Stone::Black) | *board.bitboard(Stone::White);
    let mut nearby = PositionArray::new(false);
    for position in occupied.iter_set() {
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

    let mut count = 0;
    for (position, &is_nearby) in nearby.iter() {
        if !is_nearby {
            continue;
        }
        if own_threats.is_set(position) {
            buf[0] = position;
            return (0, 1);
        }
        buf[count] = position;
        count += 1;
    }

    // Partition blocking moves to the front with swaps.
    let mut blocking_end = 0;
    for i in 0..count {
        if opp_threats.is_set(buf[i]) {
            buf.swap(i, blocking_end);
            blocking_end += 1;
        }
    }

    (blocking_end, count)
}

/// Finds the best move for `stone` using negamax with alpha-beta pruning.
///
/// Candidates are shuffled before searching so that among equally-scored moves,
/// whichever appears first after the shuffle is chosen — providing variety
/// without the fail-soft false-tie bug that reservoir sampling would introduce.
pub fn find_best_move(
    board: &mut Board,
    stone: Stone,
    depth: u32,
    rng: &mut fastrand::Rng,
) -> PositionId {
    assert!(depth > 0, "find_best_move called with depth 0");

    let mut buf = [PositionId::default(); PositionId::COUNT];
    let (blocking_count, count) = generate_candidates(board, stone, &mut buf);
    let candidates = &mut buf[..count];
    assert!(
        !candidates.is_empty(),
        "find_best_move called with no candidates"
    );

    // Shuffle within priority tiers to preserve blocking-first move ordering
    // while randomizing which equally-scored move is encountered first.
    rng.shuffle(&mut candidates[..blocking_count]);
    rng.shuffle(&mut candidates[blocking_count..]);

    let mut best_move = candidates[0];
    let mut best_score = Score::MIN;

    for &candidate in candidates.iter() {
        board.place(candidate, stone).expect("valid search move");

        let score = match board.outcome() {
            Some(Outcome::BlackWins | Outcome::WhiteWins) => {
                Score::win_at_depth(board.move_count())
            }
            Some(Outcome::Draw) => Score::DRAW,
            None => -negamax(board, depth - 1, Score::MIN, -best_score, stone.opponent()),
        };

        board.undo(candidate, stone);

        if score > best_score {
            best_score = score;
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
    let (_, count) = generate_candidates(board, stone, &mut buf);
    let candidates = &buf[..count];
    if candidates.is_empty() {
        return evaluate(board, stone);
    }

    for &candidate in candidates {
        board.place(candidate, stone).expect("valid search move");

        let score = match board.outcome() {
            Some(Outcome::BlackWins | Outcome::WhiteWins) => {
                Score::win_at_depth(board.move_count())
            }
            Some(Outcome::Draw) => Score::DRAW,
            None => -negamax(board, depth - 1, -beta, -alpha, stone.opponent()),
        };

        board.undo(candidate, stone);

        if score >= beta {
            return beta;
        }
        if score > alpha {
            alpha = score;
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
        let mut rng = fastrand::Rng::with_seed(42);

        let result = find_best_move(&mut board, Stone::Black, 4, &mut rng);

        assert_eq!(result, PositionId::center());
    }

    #[test]
    fn finds_immediate_winning_move() {
        let mut board = Board::new();
        place_stones(&mut board, Stone::Black, &[(7, 5), (7, 6), (7, 7), (7, 8)]);
        place_stones(&mut board, Stone::White, &[(8, 5), (8, 6), (8, 7)]);
        let mut rng = fastrand::Rng::with_seed(42);

        let result = find_best_move(&mut board, Stone::Black, 4, &mut rng);

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
        // Half-open four: Black at (7,4) blocks one end, so (7,9) is the only block
        place_stones(&mut board, Stone::White, &[(7, 5), (7, 6), (7, 7), (7, 8)]);
        place_stones(&mut board, Stone::Black, &[(7, 4), (8, 5), (8, 6)]);
        let mut rng = fastrand::Rng::with_seed(42);

        let result = find_best_move(&mut board, Stone::Black, 4, &mut rng);

        assert_eq!(
            result,
            pos(7, 9),
            "Expected blocking move at (7,9), got ({}, {})",
            result.row(),
            result.col()
        );
    }

    #[test]
    fn prefers_faster_win() {
        let mut board = Board::new();
        place_stones(&mut board, Stone::Black, &[(7, 5), (7, 6), (7, 7), (7, 8)]);
        place_stones(&mut board, Stone::White, &[(8, 5), (8, 6), (8, 7)]);
        let mut rng = fastrand::Rng::with_seed(42);

        let result = find_best_move(&mut board, Stone::Black, 4, &mut rng);

        assert!(
            result == pos(7, 4) || result == pos(7, 9),
            "Expected immediate win, got ({}, {})",
            result.row(),
            result.col()
        );
    }

    #[test]
    fn equal_moves_are_selected_uniformly() {
        // Open three: White has (7,6), (7,7), (7,8) with both ends open.
        // Black stones are far away and symmetric, so blocking at (7,5) or
        // (7,9) is equally good — both should appear.
        let mut board = Board::new();
        place_stones(&mut board, Stone::White, &[(7, 6), (7, 7), (7, 8)]);
        place_stones(&mut board, Stone::Black, &[(2, 6), (2, 8)]);

        let mut seen = std::collections::HashSet::new();
        for seed in 0..100 {
            let mut rng = fastrand::Rng::with_seed(seed);
            let result = find_best_move(&mut board, Stone::Black, 2, &mut rng);
            seen.insert(result);
        }

        assert!(
            seen.contains(&pos(7, 5)),
            "Expected (7,5) to appear at least once in 100 samples"
        );
        assert!(
            seen.contains(&pos(7, 9)),
            "Expected (7,9) to appear at least once in 100 samples"
        );
    }

    #[test]
    fn generates_candidates_near_stones() {
        let mut board = Board::new();
        board.place(PositionId::center(), Stone::Black).unwrap();

        let mut buf = [PositionId::default(); PositionId::COUNT];
        let (_, count) = generate_candidates(&board, Stone::White, &mut buf);

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
    fn winning_move_short_circuits_to_single_candidate() {
        let mut board = Board::new();
        place_stones(&mut board, Stone::Black, &[(7, 5), (7, 6), (7, 7), (7, 8)]);
        place_stones(&mut board, Stone::White, &[(8, 5), (8, 6)]);

        let mut buf = [PositionId::default(); PositionId::COUNT];
        let (_, count) = generate_candidates(&board, Stone::Black, &mut buf);

        assert_eq!(count, 1, "Should short-circuit to a single winning move");
        assert!(
            board.would_win(buf[0], Stone::Black),
            "The single candidate should be a winning move"
        );
    }

    #[test]
    fn blocking_candidates_appear_before_regular() {
        let mut board = Board::new();
        place_stones(&mut board, Stone::White, &[(7, 5), (7, 6), (7, 7), (7, 8)]);
        place_stones(&mut board, Stone::Black, &[(8, 5), (8, 6)]);

        let mut buf = [PositionId::default(); PositionId::COUNT];
        let (_, count) = generate_candidates(&board, Stone::Black, &mut buf);
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
