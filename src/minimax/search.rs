use crate::bitboard::winning_threats;
use crate::board::Board;
use crate::outcome::Outcome;
use crate::position_id::PositionId;
use crate::stone::Stone;

use super::score::Score;
use super::search_state::SearchState;

pub const DEFAULT_DEPTH: u32 = 4;
type CandidateBuf = [PositionId; PositionId::COUNT];

struct KillerTable {
    slots: Vec<[Option<PositionId>; 2]>,
}

impl KillerTable {
    fn new(depth: u32) -> Self {
        let len: usize = depth.try_into().expect("depth fits in usize");
        Self {
            slots: vec![[None; 2]; len],
        }
    }

    fn get(&self, depth: u32) -> &[Option<PositionId>; 2] {
        let idx: usize = depth.try_into().expect("depth fits in usize");
        &self.slots[idx]
    }

    fn put(&mut self, depth: u32, position: PositionId) {
        let idx: usize = depth.try_into().expect("depth fits in usize");
        let entry = &mut self.slots[idx];
        if entry[0] == Some(position) {
            return;
        }
        entry[1] = entry[0];
        entry[0] = Some(position);
    }
}

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
    let nearby = occupied.expand_nearby() & !occupied;

    let mut count = 0;
    for position in nearby.iter_set() {
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

    let mut state = SearchState::from_board(board);

    let mut buf = [PositionId::default(); PositionId::COUNT];
    let (blocking_count, count) = generate_candidates(state.board(), stone, &mut buf);
    let candidates = &mut buf[..count];
    assert!(
        !candidates.is_empty(),
        "find_best_move called with no candidates"
    );

    // Shuffle within priority tiers to preserve blocking-first move ordering
    // while randomizing which equally-scored move is encountered first.
    rng.shuffle(&mut candidates[..blocking_count]);
    rng.shuffle(&mut candidates[blocking_count..]);

    let mut killers = KillerTable::new(depth);

    let mut best_move = candidates[0];
    let mut best_score = Score::MIN;

    for &candidate in candidates.iter() {
        state.place(candidate, stone);

        let score = match state.outcome() {
            Some(Outcome::BlackWins | Outcome::WhiteWins) => {
                Score::win_at_depth(state.move_count())
            }
            Some(Outcome::Draw) => Score::DRAW,
            None => -negamax(
                &mut state,
                depth - 1,
                Score::MIN,
                -best_score,
                stone.opponent(),
                &mut killers,
            ),
        };

        state.undo(candidate, stone);

        if score > best_score {
            best_score = score;
            best_move = candidate;
        }
    }

    best_move
}

fn negamax(
    state: &mut SearchState,
    depth: u32,
    mut alpha: Score,
    beta: Score,
    stone: Stone,
    killers: &mut KillerTable,
) -> Score {
    if depth == 0 || state.is_full() {
        return state.evaluate(stone);
    }

    let mut buf = [PositionId::default(); PositionId::COUNT];
    let (blocking_count, count) = generate_candidates(state.board(), stone, &mut buf);
    if count == 0 {
        return state.evaluate(stone);
    }

    // Promote killer moves to right after blocking moves.
    let mut priority_end = blocking_count;
    for killer in killers.get(depth - 1).iter().flatten() {
        if let Some(idx) = buf[priority_end..count]
            .iter()
            .position(|&pos| pos == *killer)
        {
            buf.swap(priority_end, priority_end + idx);
            priority_end += 1;
        }
    }

    let candidates = &buf[..count];

    for &candidate in candidates {
        state.place(candidate, stone);

        let score = match state.outcome() {
            Some(Outcome::BlackWins | Outcome::WhiteWins) => {
                Score::win_at_depth(state.move_count())
            }
            Some(Outcome::Draw) => Score::DRAW,
            None => -negamax(state, depth - 1, -beta, -alpha, stone.opponent(), killers),
        };

        state.undo(candidate, stone);

        if score >= beta {
            killers.put(depth - 1, candidate);
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
    use crate::bitboard::PROXIMITY_RADIUS;
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
                row_dist <= PROXIMITY_RADIUS && col_dist <= PROXIMITY_RADIUS,
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
        let threats = winning_threats(*board.bitboard(Stone::Black));
        assert!(
            threats.is_set(buf[0]),
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

        let opp_threats = winning_threats(*board.bitboard(Stone::White));
        let own_threats = winning_threats(*board.bitboard(Stone::Black));

        let first_regular = candidates.iter().position(|&candidate| {
            !opp_threats.is_set(candidate) && !own_threats.is_set(candidate)
        });
        let last_blocking = candidates
            .iter()
            .rposition(|&candidate| opp_threats.is_set(candidate));

        if let (Some(first_reg), Some(last_blk)) = (first_regular, last_blocking) {
            assert!(
                last_blk < first_reg,
                "Blocking candidates should appear before regular candidates"
            );
        }
    }

    #[test]
    fn killer_table_starts_empty() {
        let table = KillerTable::new(4);

        for depth in 0..4 {
            assert_eq!(*table.get(depth), [None, None]);
        }
    }

    #[test]
    fn killer_table_stores_in_first_slot() {
        let mut table = KillerTable::new(4);
        let position = pos(7, 7);

        table.put(2, position);

        assert_eq!(table.get(2), &[Some(position), None]);
    }

    #[test]
    fn killer_table_shifts_first_to_second_on_new_entry() {
        let mut table = KillerTable::new(4);
        let first = pos(7, 7);
        let second = pos(3, 3);

        table.put(1, first);
        table.put(1, second);

        assert_eq!(table.get(1), &[Some(second), Some(first)]);
    }

    #[test]
    fn killer_table_skips_duplicate_in_first_slot() {
        let mut table = KillerTable::new(4);
        let first = pos(7, 7);
        let second = pos(3, 3);

        table.put(0, first);
        table.put(0, second);
        table.put(0, second);

        assert_eq!(table.get(0), &[Some(second), Some(first)]);
    }

    #[test]
    fn killer_table_depths_are_independent() {
        let mut table = KillerTable::new(4);
        let position_a = pos(7, 7);
        let position_b = pos(3, 3);

        table.put(0, position_a);
        table.put(3, position_b);

        assert_eq!(table.get(0), &[Some(position_a), None]);
        assert_eq!(table.get(1), &[None, None]);
        assert_eq!(table.get(3), &[Some(position_b), None]);
    }

    #[test]
    fn killer_table_third_entry_evicts_oldest() {
        let mut table = KillerTable::new(4);
        let first = pos(7, 7);
        let second = pos(3, 3);
        let third = pos(5, 5);

        table.put(0, first);
        table.put(0, second);
        table.put(0, third);

        assert_eq!(table.get(0), &[Some(third), Some(second)]);
    }

    #[test]
    fn search_is_deterministic_with_killers() {
        let mut board = Board::new();
        place_stones(&mut board, Stone::Black, &[(7, 7), (7, 8), (8, 6)]);
        place_stones(&mut board, Stone::White, &[(6, 7), (8, 8), (9, 5)]);

        let result_a = find_best_move(
            &mut board,
            Stone::Black,
            4,
            &mut fastrand::Rng::with_seed(42),
        );
        let result_b = find_best_move(
            &mut board,
            Stone::Black,
            4,
            &mut fastrand::Rng::with_seed(42),
        );

        assert_eq!(result_a, result_b);
    }

    #[test]
    fn search_at_depth_one_works_with_killers() {
        let mut board = Board::new();
        place_stones(&mut board, Stone::Black, &[(7, 7)]);
        let mut rng = fastrand::Rng::with_seed(42);

        let result = find_best_move(&mut board, Stone::White, 1, &mut rng);

        let row_dist = result.row().abs_diff(7);
        let col_dist = result.col().abs_diff(7);
        assert!(
            row_dist <= PROXIMITY_RADIUS && col_dist <= PROXIMITY_RADIUS,
            "Move ({}, {}) should be near the existing stone",
            result.row(),
            result.col()
        );
    }
}
