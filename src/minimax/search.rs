use crate::board::Board;
use crate::outcome::Outcome;
use crate::position_id::PositionId;
use crate::stone::Stone;
use tracing::debug;

use super::lines::NEIGHBOURHOODS;
use super::patterns::Threat;
use super::score::Score;
use super::search_state::{SearchState, Situation};
use super::tt::{Bound, TranspositionTable};

pub const DEFAULT_DEPTH: u32 = 4;

/// How many stones a victory by continuous fours may run at the horizon
/// before it is given up (Rapfi's `MAX_VCF_PLY`).
const VCF_PLIES: u32 = 36;
type CandidateBuf = [PositionId; PositionId::COUNT];

/// Search depth left, in thirds of a move.
///
/// A move with few possible replies costs less depth than one with many, as
/// in Rapfi, where a move costs the logarithm of its branching factor: the
/// one block of a four costs nothing, the handful of defences against an
/// unstoppable four in the making a third of a move, any other move a whole
/// one. Thirds are the coarsest unit that keeps those three apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Depth(u32);

impl Depth {
    const ZERO: Self = Self(0);
    const THIRD: Self = Self(1);
    const MOVE: Self = Self(3);

    #[must_use]
    const fn moves(count: u32) -> Self {
        Self(count * Self::MOVE.0)
    }

    /// What is left after spending `cost`, never below zero.
    #[must_use]
    const fn minus(self, cost: Self) -> Self {
        Self(self.0.saturating_sub(cost.0))
    }

    /// The count of thirds, for tables indexed by depth.
    #[must_use]
    const fn thirds(self) -> u32 {
        self.0
    }
}

struct KillerTable {
    slots: Vec<[Option<PositionId>; 2]>,
}

impl KillerTable {
    fn new(depth: Depth) -> Self {
        let len: usize = depth.thirds().try_into().expect("depth fits in usize");
        Self {
            slots: vec![[None; 2]; len],
        }
    }

    fn get(&self, depth: Depth) -> &[Option<PositionId>; 2] {
        let idx: usize = depth.thirds().try_into().expect("depth fits in usize");
        &self.slots[idx]
    }

    fn put(&mut self, depth: Depth, position: PositionId) {
        let idx: usize = depth.thirds().try_into().expect("depth fits in usize");
        let entry = &mut self.slots[idx];
        if entry[0] == Some(position) {
            return;
        }
        entry[1] = entry[0];
        entry[0] = Some(position);
    }
}

/// Finds the best move for `stone` using negamax with alpha-beta pruning.
///
/// `depth` is the number of developing moves searched along each line of
/// play; near-forced replies cost less (see [`Depth`]), and a forced reply is
/// searched even when the depth has run out. Candidates are shuffled before
/// searching so that among equally-scored moves, whichever appears first
/// after the shuffle is chosen — providing variety without the fail-soft
/// false-tie bug that reservoir sampling would introduce.
pub fn find_best_move(
    board: &mut Board,
    stone: Stone,
    depth: u32,
    rng: &mut fastrand::Rng,
) -> PositionId {
    let mut tt = TranspositionTable::new();
    let mut search = Search::new(Depth::moves(depth), &mut tt);
    let best_move = search.best_move(board, stone, Depth::moves(depth), rng);
    debug!(
        move_count = board.move_count(),
        depth,
        nodes = search.nodes,
        "minimax move chosen"
    );
    best_move
}

/// Scores `position` for `stone` on `board` using negamax to `depth`.
/// Returns the score from `stone`'s perspective.
pub fn score_move(
    board: &Board,
    stone: Stone,
    position: PositionId,
    depth: u32,
    tt: &mut TranspositionTable,
) -> Score {
    assert!(depth > 0, "score_move called with depth 0");
    let mut state = SearchState::from_board(board);
    let raw = state.evaluate(stone);
    state.place(position, stone);
    Search::new(Depth::moves(depth), tt).score_after_place(
        &mut state,
        stone,
        Depth::moves(depth - 1),
        raw,
        Score::MIN,
        Score::MAX,
    )
}

/// One search from one root position: the move-ordering tables and a node
/// count for diagnostics.
struct Search<'a> {
    killers: KillerTable,
    tt: &'a mut TranspositionTable,
    nodes: u64,
}

impl<'a> Search<'a> {
    fn new(depth: Depth, tt: &'a mut TranspositionTable) -> Self {
        Self {
            killers: KillerTable::new(depth),
            tt,
            nodes: 0,
        }
    }

    /// Searches every candidate for `stone` to `depth` and returns the best.
    fn best_move(
        &mut self,
        board: &Board,
        stone: Stone,
        depth: Depth,
        rng: &mut fastrand::Rng,
    ) -> PositionId {
        assert!(depth > Depth::ZERO, "find_best_move called with depth 0");

        let mut state = SearchState::from_board(board);

        let mut buf = [PositionId::default(); PositionId::COUNT];
        let situation = state.situation(stone);
        let count = state.candidates(stone, situation, &mut buf).count;
        let candidates = &mut buf[..count];
        assert!(
            !candidates.is_empty(),
            "find_best_move called with no candidates"
        );

        // Randomize which equally-scored move is encountered first.
        rng.shuffle(candidates);

        let mut best_move = candidates[0];
        let mut best_score = Score::MIN;
        let raw = state.evaluate(stone);

        for &candidate in candidates.iter() {
            state.place(candidate, stone);
            let score = self.score_after_place(
                &mut state,
                stone,
                depth.minus(Depth::MOVE),
                raw,
                best_score,
                Score::MAX,
            );
            state.undo(candidate, stone);

            if score > best_score {
                best_score = score;
                best_move = candidate;
            }
        }

        best_move
    }

    /// Scores the current `state`, where `stone` just played, from `stone`'s
    /// perspective: a finished game by its outcome, anything else by searching
    /// the opponent's reply to `child_depth` inside the window `alpha..beta`.
    /// `raw` is the static evaluation for `stone` before it played, which a
    /// leaf averages with its own.
    fn score_after_place(
        &mut self,
        state: &mut SearchState,
        stone: Stone,
        child_depth: Depth,
        raw: Score,
        alpha: Score,
        beta: Score,
    ) -> Score {
        match state.outcome() {
            Some(Outcome::Win(_)) => Score::win_at_depth(state.move_count()),
            Some(Outcome::Draw) => Score::DRAW,
            None => -self.negamax(state, child_depth, raw, -beta, -alpha, stone.opponent()),
        }
    }

    /// The static value of a quiet leaf for `stone`: its own evaluation
    /// averaged with the parent's, `parent_raw`, seen from `stone`'s side.
    /// Whoever placed the last stone always looks ahead, since that stone's
    /// threats are unanswered; averaging over the last move takes half of
    /// that swing out, as Rapfi does with `rawStaticEval[ply - 1]`.
    fn leaf_value(state: &SearchState, stone: Stone, parent_raw: Score) -> Score {
        (state.evaluate(stone) - parent_raw) / 2
    }

    /// A win in five stones for `stone`, about to move with no four on the
    /// board on either side, that needs no search to see (Rapfi's
    /// `quickWinCheck` beyond fives and unstoppable fours).
    ///
    /// A stone making a four and a free three at once wins: the four must be
    /// blocked, then the free three becomes an open four. The block can only
    /// break the sequence by making a four of its own, so it is played out
    /// unless the opponent has no cell that makes a four at all. A stone
    /// making two free threes wins the same way as long as the opponent has
    /// no cell that makes a four, since any of its defences could then be a
    /// counter-four instead.
    fn win_by_threats(state: &mut SearchState, stone: Stone) -> Option<Score> {
        let opponent = stone.opponent();
        let opponent_can_make_a_four = (state.four_making_cells(opponent)
            | state.cells_with(opponent, Threat::UnstoppableFour))
        .any();
        let win = Score::win_at_depth(state.move_count() + 5);

        let four_threes = state.cells_with(stone, Threat::FourAndFreeThree);
        if four_threes.any() && !opponent_can_make_a_four {
            return Some(win);
        }
        for cell in four_threes.iter_set() {
            state.place(cell, stone);
            let block = state.cells_with(stone, Threat::Five).iter_set().next();
            let block_makes_no_four =
                block.is_some_and(|block| state.threat_at(block, opponent) < Threat::Four);
            state.undo(cell, stone);
            if block_makes_no_four {
                return Some(win);
            }
        }

        if !opponent_can_make_a_four && state.cells_with(stone, Threat::TwoFreeThrees).any() {
            return Some(win);
        }
        None
    }

    /// Rapfi's `quickVCFSearch`: whether `attacker`, about to move, wins by
    /// fours alone within `plies` stones. Every attacking stone must make a
    /// four; the defender's block is forced; a block that makes a four back
    /// must itself be answered with a four. After the first stone only
    /// fours within reach of the previous attacking stone are tried.
    fn continuous_fours(
        &mut self,
        state: &mut SearchState,
        attacker: Stone,
        previous: Option<PositionId>,
        plies: u32,
    ) -> Option<Score> {
        self.nodes += 1;
        let defender = attacker.opponent();
        match state.situation(attacker) {
            Situation::CompleteFive => return Some(Score::win_at_depth(state.move_count() + 1)),
            Situation::MakeUnstoppableFour => {
                return Some(Score::win_at_depth(state.move_count() + 3));
            }
            Situation::BlockFour => {
                // The last block made a four back: only a four of our own
                // that blocks it keeps the sequence going.
                let mut blocks = state.cells_with(defender, Threat::Five).iter_set();
                let block = blocks.next().expect("a four has a completing cell");
                if blocks.next().is_some() || state.threat_at(block, attacker) < Threat::Four {
                    return None;
                }
                return self.continuous_fours_after(state, attacker, block, plies);
            }
            Situation::PreventUnstoppableFour | Situation::Develop => {}
        }
        if plies == 0 {
            return None;
        }
        let mut fours = state.four_making_cells(attacker);
        if let Some(previous) = previous {
            let mut nearby = crate::bitboard::BitBoard::EMPTY;
            for &(cell, _) in NEIGHBOURHOODS.get(previous).cells() {
                nearby.set(cell);
            }
            fours = fours & nearby;
        }
        for cell in fours.iter_set() {
            if let Some(win) = self.continuous_fours_after(state, attacker, cell, plies) {
                return Some(win);
            }
        }
        None
    }

    /// Plays the attacker's four at `cell` and the defender's forced block,
    /// then carries the search on.
    fn continuous_fours_after(
        &mut self,
        state: &mut SearchState,
        attacker: Stone,
        cell: PositionId,
        plies: u32,
    ) -> Option<Score> {
        let defender = attacker.opponent();
        state.place(cell, attacker);
        let result = match state.cells_with(attacker, Threat::Five).iter_set().next() {
            Some(block) if plies >= 2 => {
                state.place(block, defender);
                let result = self.continuous_fours(state, attacker, Some(cell), plies - 2);
                state.undo(block, defender);
                result
            }
            _ => None,
        };
        state.undo(cell, attacker);
        result
    }

    /// Scores the position for `stone`, who is about to move, with `depth`
    /// left to search.
    ///
    /// A four on the board, or a cell that makes an unstoppable four,
    /// decides what happens next, so such positions are never scored
    /// statically: a side that can complete five or make an unstoppable
    /// four has already won, and a side that must block or prevent one gets
    /// its forced replies searched even when the depth has run out. Only a
    /// position with nothing forced is a leaf then.
    #[expect(
        clippy::too_many_lines,
        reason = "one linear pass over the node reads better than helpers with one caller"
    )]
    fn negamax(
        &mut self,
        state: &mut SearchState,
        depth: Depth,
        parent_raw: Score,
        mut alpha: Score,
        beta: Score,
        stone: Stone,
    ) -> Score {
        self.nodes += 1;
        let situation = state.situation(stone);
        match situation {
            Situation::CompleteFive => return Score::win_at_depth(state.move_count() + 1),
            // Unstoppable four now, any reply, five: nothing the opponent
            // does matters because it has no four to complete first.
            Situation::MakeUnstoppableFour => {
                return Score::win_at_depth(state.move_count() + 3);
            }
            Situation::Develop | Situation::PreventUnstoppableFour
                if let Some(win) = Self::win_by_threats(state, stone) =>
            {
                return win;
            }
            Situation::Develop if depth == Depth::ZERO => {
                // Rapfi's leaf: below beta the side to move first gets to try
                // a victory by continuous fours, which no static value sees.
                let value = Self::leaf_value(state, stone, parent_raw);
                if value < beta
                    && let Some(win) = self.continuous_fours(state, stone, None, VCF_PLIES)
                {
                    return win;
                }
                return value;
            }
            // At the horizon a threatened side may still win by fours of its
            // own: each forces a block, so the threat never gets carried out.
            Situation::PreventUnstoppableFour if depth == Depth::ZERO => {
                if let Some(win) = self.continuous_fours(state, stone, None, VCF_PLIES) {
                    return win;
                }
            }
            Situation::BlockFour | Situation::PreventUnstoppableFour | Situation::Develop => {}
        }

        let alpha_orig = alpha;
        let mut beta = beta;

        // A leaf's value depends on its parent, so only nodes with depth to
        // search below them, whose value is the position's own, use the table.
        // A shallower entry's best move is still the first move to try.
        let mut hash_move = None;
        if depth > Depth::ZERO
            && let Some(probe) = self.tt.probe(state.hash(), depth.thirds())
        {
            hash_move = probe.best_move;
            if let Some((tt_score, bound)) = probe.value {
                match bound {
                    Bound::Exact => return tt_score,
                    Bound::Lower => alpha = alpha.max(tt_score),
                    Bound::Upper => beta = beta.min(tt_score),
                }
                if alpha >= beta {
                    return beta;
                }
            }
        }

        let mut buf: CandidateBuf = [PositionId::default(); PositionId::COUNT];
        let candidates = state.candidates(stone, situation, &mut buf);

        // The horizon never cuts off a forced reply, so a position with a
        // four or an unstoppable four in the making is never scored
        // statically.
        let searched = if depth == Depth::ZERO {
            candidates.forced
        } else {
            candidates.count
        };
        if searched == 0 {
            return Self::leaf_value(state, stone, parent_raw);
        }
        let raw = state.evaluate(stone);

        // Among the chosen moves, try the table's best move first, then the
        // killer moves.
        let mut priority_end = candidates.forced;
        for preferred in hash_move
            .iter()
            .chain(self.killers.get(depth).iter().flatten())
        {
            if let Some(idx) = buf[priority_end..searched]
                .iter()
                .position(|&pos| pos == *preferred)
            {
                buf.swap(priority_end, priority_end + idx);
                priority_end += 1;
            }
        }

        let mut best_move = None;
        for (i, &candidate) in buf[..searched].iter().enumerate() {
            let is_forced = i < candidates.forced;
            // A counter-four against an unstoppable four in the making is
            // charged like any other move, or a side with many fours
            // available could keep the tree growing without bound.
            let cost = match situation {
                Situation::BlockFour => Depth::ZERO,
                Situation::PreventUnstoppableFour if is_forced => Depth::THIRD,
                _ => Depth::MOVE,
            };

            state.place(candidate, stone);
            let score = self.score_after_place(state, stone, depth.minus(cost), raw, alpha, beta);
            state.undo(candidate, stone);

            if score >= beta {
                if !is_forced {
                    self.killers.put(depth, candidate);
                }
                if depth > Depth::ZERO {
                    self.tt.store(
                        state.hash(),
                        depth.thirds(),
                        beta,
                        Bound::Lower,
                        Some(candidate),
                    );
                }
                return beta;
            }
            if score > alpha {
                alpha = score;
                best_move = Some(candidate);
            }
        }

        // Every move failing low leaves no move worth trying first.
        let bound = if alpha > alpha_orig {
            Bound::Exact
        } else {
            Bound::Upper
        };
        if depth > Depth::ZERO {
            self.tt
                .store(state.hash(), depth.thirds(), alpha, bound, best_move);
        }
        alpha
    }
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

    fn score_move_fresh(board: &Board, stone: Stone, position: PositionId, depth: u32) -> Score {
        score_move(
            board,
            stone,
            position,
            depth,
            &mut TranspositionTable::new(),
        )
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
    fn makes_the_open_four_from_an_open_three() {
        let mut board = Board::new();
        // O_XXX__: only (7,9) makes an open four.
        place_stones(&mut board, Stone::Black, &[(7, 6), (7, 7), (7, 8)]);
        place_stones(&mut board, Stone::White, &[(7, 4), (0, 0), (0, 14)]);
        let mut rng = fastrand::Rng::with_seed(42);

        let result = find_best_move(&mut board, Stone::Black, 1, &mut rng);

        assert_eq!(result, pos(7, 9));
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
    fn killer_table_starts_empty() {
        let table = KillerTable::new(Depth(4));

        for depth in 0..4 {
            assert_eq!(*table.get(Depth(depth)), [None, None]);
        }
    }

    #[test]
    fn killer_table_stores_in_first_slot() {
        let mut table = KillerTable::new(Depth(4));
        let position = pos(7, 7);

        table.put(Depth(2), position);

        assert_eq!(table.get(Depth(2)), &[Some(position), None]);
    }

    #[test]
    fn killer_table_shifts_first_to_second_on_new_entry() {
        let mut table = KillerTable::new(Depth(4));
        let first = pos(7, 7);
        let second = pos(3, 3);

        table.put(Depth(1), first);
        table.put(Depth(1), second);

        assert_eq!(table.get(Depth(1)), &[Some(second), Some(first)]);
    }

    #[test]
    fn killer_table_skips_duplicate_in_first_slot() {
        let mut table = KillerTable::new(Depth(4));
        let first = pos(7, 7);
        let second = pos(3, 3);

        table.put(Depth(0), first);
        table.put(Depth(0), second);
        table.put(Depth(0), second);

        assert_eq!(table.get(Depth(0)), &[Some(second), Some(first)]);
    }

    #[test]
    fn killer_table_depths_are_independent() {
        let mut table = KillerTable::new(Depth(4));
        let position_a = pos(7, 7);
        let position_b = pos(3, 3);

        table.put(Depth(0), position_a);
        table.put(Depth(3), position_b);

        assert_eq!(table.get(Depth(0)), &[Some(position_a), None]);
        assert_eq!(table.get(Depth(1)), &[None, None]);
        assert_eq!(table.get(Depth(3)), &[Some(position_b), None]);
    }

    #[test]
    fn killer_table_third_entry_evicts_oldest() {
        let mut table = KillerTable::new(Depth(4));
        let first = pos(7, 7);
        let second = pos(3, 3);
        let third = pos(5, 5);

        table.put(Depth(0), first);
        table.put(Depth(0), second);
        table.put(Depth(0), third);

        assert_eq!(table.get(Depth(0)), &[Some(third), Some(second)]);
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

    #[test]
    fn score_move_completing_five_returns_win() {
        let mut board = Board::new();
        place_stones(&mut board, Stone::Black, &[(7, 5), (7, 6), (7, 7), (7, 8)]);
        place_stones(&mut board, Stone::White, &[(8, 5), (8, 6), (8, 7)]);

        let score = score_move_fresh(&board, Stone::Black, pos(7, 4), 4);

        // 4 Black + 3 White + 1 placed = 8 stones total
        assert_eq!(score, Score::win_at_depth(8));
    }

    #[test]
    fn score_move_winning_scores_higher_than_non_winning() {
        let mut board = Board::new();
        place_stones(&mut board, Stone::Black, &[(7, 5), (7, 6), (7, 7), (7, 8)]);
        place_stones(&mut board, Stone::White, &[(8, 5), (8, 6), (8, 7)]);

        let winning = score_move_fresh(&board, Stone::Black, pos(7, 4), 4);
        let other = score_move_fresh(&board, Stone::Black, pos(6, 5), 4);

        assert!(winning > other);
    }

    #[test]
    fn score_move_open_four_is_won_at_depth_one() {
        let mut board = Board::new();
        place_stones(&mut board, Stone::Black, &[(7, 6), (7, 7), (7, 8)]);
        place_stones(&mut board, Stone::White, &[(2, 2), (2, 3)]);

        // Place at (7,5) creates _XXXX_. White's block is forced and costs no
        // depth; Black then completes five: 6 stones + block + five = 8.
        for depth in 1..=4 {
            let score = score_move_fresh(&board, Stone::Black, pos(7, 5), depth);
            assert_eq!(score, Score::win_at_depth(8), "at depth {depth}");
        }
    }

    #[test]
    fn score_move_half_open_four_is_worth_nothing_once_its_block_is_searched() {
        let mut board = Board::new();
        place_stones(&mut board, Stone::Black, &[(7, 6), (7, 7), (7, 8)]);
        place_stones(&mut board, Stone::White, &[(7, 5)]);

        // Place at (7,9) creates OXXXX_. White's block at (7,10) is forced,
        // leaving the dead four OXXXXO: the row is worth nothing, and what
        // remains is the small change the stones make on other lines.
        let score = score_move_fresh(&board, Stone::Black, pos(7, 9), 1);

        // What remains is the small change the stones make on other lines,
        // well under a single cell's worth of a four in Rapfi's table.
        assert!(!score.is_decided(), "{score:?}");
        assert!(score < Score::new(300), "{score:?}");
    }

    #[test]
    fn score_move_open_three_is_answered_before_the_leaf() {
        let mut board = Board::new();
        place_stones(&mut board, Stone::Black, &[(7, 7), (7, 8)]);

        // Place at (7,6) creates _XXX_. White's answer is forced and costs no
        // depth: either neighbour leaves Black a half-open three.
        let score = score_move_fresh(&board, Stone::Black, pos(7, 6), 1);

        assert!(score > Score::DRAW && !score.is_decided(), "{score:?}");
    }

    #[test]
    fn score_move_finds_win_by_four_then_open_three_at_depth_one() {
        let mut board = Board::new();
        place_stones(
            &mut board,
            Stone::Black,
            &[(7, 4), (7, 5), (7, 6), (5, 7), (6, 7)],
        );
        place_stones(&mut board, Stone::White, &[(7, 3)]);

        // (7,7) makes the four OXXXX_ and the open three in column 7. White
        // must block at (7,8); Black then holds an open three with no White
        // four in sight, which is a win in three more stones: 8 + 3 = 11.
        let score = score_move_fresh(&board, Stone::Black, pos(7, 7), 1);

        assert_eq!(score, Score::win_at_depth(11));
    }

    #[test]
    fn score_move_counter_four_saves_a_lost_open_three() {
        let mut board = Board::new();
        // Black _XXX_ on row 7. White has OOO on row 3 hemmed at (3,4), and a
        // second three XOO_O_ on row 11, hemmed too so it is no win in itself.
        place_stones(
            &mut board,
            Stone::Black,
            &[(7, 6), (7, 7), (7, 8), (3, 4), (11, 4)],
        );
        place_stones(
            &mut board,
            Stone::White,
            &[(3, 5), (3, 6), (3, 7), (11, 5), (11, 6), (11, 8)],
        );

        // White plays (3,8): a four Black must block at (3,9). White then
        // still faces the open three and defuses it, say at (7,5). Black
        // develops one free move at depth 2 and White is not lost.
        let score = score_move_fresh(&board, Stone::White, pos(3, 8), 2);

        assert!(
            !score.is_decided(),
            "the counter-four should keep White in the game: {score:?}"
        );
    }

    #[test]
    fn leaf_finds_a_win_by_continuous_fours() {
        // Black: OXXX_ on row 7, a blocked pair on column 6, and a stone at
        // (9,7) on the diagonal through (7,5) and (8,6). (7,6) is a four;
        // after the forced block at (7,7), (8,6) makes a four on column 6
        // and a free three on the diagonal at once. No single cell wins
        // outright, so only the continuous-fours search sees it.
        let mut board = Board::new();
        place_stones(
            &mut board,
            Stone::Black,
            &[(7, 3), (7, 4), (7, 5), (5, 6), (6, 6), (9, 7)],
        );
        place_stones(&mut board, Stone::White, &[(7, 2), (4, 6), (0, 0)]);

        // A quiet White move elsewhere leaves Black to move at the leaf.
        let lost = score_move_fresh(&board, Stone::White, pos(0, 14), 1);
        assert!(lost.is_decided() && lost < Score::DRAW, "{lost:?}");

        // Without the diagonal stone there is no follow-up.
        let mut calm = Board::new();
        place_stones(
            &mut calm,
            Stone::Black,
            &[(7, 3), (7, 4), (7, 5), (5, 6), (6, 6)],
        );
        place_stones(&mut calm, Stone::White, &[(7, 2), (4, 6), (0, 0)]);
        let fine = score_move_fresh(&calm, Stone::White, pos(0, 14), 1);
        assert!(!fine.is_decided(), "{fine:?}");
    }

    #[test]
    fn score_move_negative_when_position_favors_opponent() {
        let mut board = Board::new();
        place_stones(&mut board, Stone::Black, &[(7, 6), (7, 7), (7, 8)]);

        // White blocks Black's left side — board still favors Black
        let score = score_move_fresh(&board, Stone::White, pos(7, 5), 1);

        assert!(score < Score::DRAW, "{score:?}");
    }
    /// A middle-game position where White, to move, faces threats on several
    /// lines and has many cells that would make a four. Counter-fours must
    /// cost depth: the depth-4 search below takes about 120 thousand nodes,
    /// and ten times that when counter-fours are free replies.
    fn dense_middle_game() -> Board {
        let mut board = Board::new();
        place_stones(
            &mut board,
            Stone::Black,
            &[
                (5, 8),
                (6, 8),
                (7, 7),
                (7, 8),
                (7, 9),
                (8, 6),
                (9, 5),
                (10, 9),
            ],
        );
        place_stones(
            &mut board,
            Stone::White,
            &[(4, 8), (5, 9), (6, 7), (7, 6), (8, 7), (9, 8), (10, 4)],
        );
        board
    }

    #[test]
    fn dense_middle_game_search_stays_within_a_node_budget() {
        let board = dense_middle_game();
        let mut tt = TranspositionTable::new();
        let mut search = Search::new(Depth::moves(4), &mut tt);

        search.best_move(
            &board,
            Stone::White,
            Depth::moves(4),
            &mut fastrand::Rng::with_seed(7),
        );

        assert!(
            search.nodes < 300_000,
            "depth-4 search used {} nodes",
            search.nodes
        );
    }

    #[test]
    fn two_free_threes_win_when_the_opponent_has_no_four_in_reach() {
        // (7,8) joins _XX_ on row 7 and _XX_ on column 8 into two free
        // threes. White's stones are far away and make no four.
        let mut board = Board::new();
        place_stones(&mut board, Stone::Black, &[(7, 6), (7, 7), (5, 8), (6, 8)]);
        place_stones(&mut board, Stone::White, &[(0, 0), (0, 14), (14, 0)]);
        let mut state = SearchState::from_board(&board);

        assert_eq!(
            Search::win_by_threats(&mut state, Stone::Black),
            Some(Score::win_at_depth(7 + 5))
        );
        assert_eq!(Search::win_by_threats(&mut state, Stone::White), None);
    }

    #[test]
    fn two_free_threes_are_no_sure_win_against_a_side_that_can_make_a_four() {
        // As above, but White holds OOO hemmed at (3,4): (3,8) makes a four,
        // so a defence of either three could be a counter-four instead.
        let mut board = Board::new();
        place_stones(
            &mut board,
            Stone::Black,
            &[(7, 6), (7, 7), (5, 8), (6, 8), (3, 4)],
        );
        place_stones(&mut board, Stone::White, &[(3, 5), (3, 6), (3, 7)]);
        let mut state = SearchState::from_board(&board);

        assert_eq!(Search::win_by_threats(&mut state, Stone::Black), None);
    }

    #[test]
    fn four_and_free_three_win_unless_the_block_makes_a_four() {
        // (7,7) makes OXXXX_ on row 7 and _XXX_ on column 7 at once; White
        // must block at (7,8). White's OOO on row 3 gives it a four to make,
        // so the block is played out: at (7,8) it makes nothing.
        let mut board = Board::new();
        place_stones(
            &mut board,
            Stone::Black,
            &[(7, 4), (7, 5), (7, 6), (5, 7), (6, 7), (3, 4)],
        );
        place_stones(&mut board, Stone::White, &[(7, 3), (3, 5), (3, 6), (3, 7)]);
        let mut state = SearchState::from_board(&board);
        assert_eq!(
            Search::win_by_threats(&mut state, Stone::Black),
            Some(Score::win_at_depth(10 + 5))
        );

        // With OO_O on column 8 the block at (7,8) makes a White four.
        place_stones(&mut board, Stone::White, &[(4, 8), (5, 8), (6, 8)]);
        place_stones(&mut board, Stone::Black, &[(8, 8)]);
        let mut state = SearchState::from_board(&board);
        assert_eq!(Search::win_by_threats(&mut state, Stone::Black), None);
    }
}
