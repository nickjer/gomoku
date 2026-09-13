use std::cmp::Reverse;

use crate::bitboard::BitBoard;
use crate::board::Board;
use crate::outcome::Outcome;
use crate::position_id::PositionId;
use crate::position_map::PositionArray;
use crate::stone::Stone;

use super::lines::{
    LINE_CELLS, LINE_LENGTHS, NEIGHBOURHOODS, NUM_LINES, POSITION_LINES, REACH, defusing_cells,
    four_breaking_cells,
};
use super::patterns::{
    CellPatterns, LinePattern, LinePatternTable, Threat, ThreatTable, line_pattern_from,
    line_pattern_table, threat_table,
};
use super::score::Score;
use super::tt::ZOBRIST;

/// What the side to move has to deal with, most urgent first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Situation {
    /// It has a four: one stone completes five.
    CompleteFive,
    /// The opponent has a four: only the cells that complete it stop the loss.
    BlockFour,
    /// It can make an open four or two fours at once, and the opponent has
    /// no four: whatever is blocked, five follows two moves later.
    MakeUnstoppableFour,
    /// The opponent could make an unstoppable four next move: take that cell
    /// or break the shape now, or answer with a four the opponent must block
    /// first.
    PreventUnstoppableFour,
    /// Nothing is forced; any cell near the stones is worth a look.
    Develop,
}

/// The moves written by [`SearchState::candidates`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Candidates {
    /// How many moves at the front are forced: the only ways not to lose at
    /// once. They are searched even when the depth has run out.
    pub forced: usize,
    /// Total number of moves written, forced ones included.
    pub count: usize,
}

/// The cells whose patterns one stone can change: itself and its
/// neighbourhood.
const CHANGED_CELLS: usize = 8 * REACH + 1;

/// The patterns of the changed cells before a stone was placed, in the order
/// `place` visits them, so `undo` can put them back without recomputing.
struct UndoFrame {
    patterns: [[CellPatterns; 2]; CHANGED_CELLS],
}

/// A board with, for every empty cell, what one more stone of each colour
/// would make there, kept up to date as stones are placed and taken back.
///
/// Stones live in one `u16` mask per colour for each of the 88 board lines.
/// A stone changes only the cells within reach on its four lines, so `place`
/// rereads those cells' patterns and `undo` restores them from a saved frame.
pub struct SearchState {
    hash: u64,
    board: Board,
    line_black: [u16; NUM_LINES],
    line_white: [u16; NUM_LINES],
    /// What a stone of each colour would make at each empty cell; `NONE`
    /// for occupied cells.
    patterns: PositionArray<[CellPatterns; 2]>,
    /// The threat of each cell's patterns, kept so a change needs only one
    /// table lookup.
    threats: PositionArray<[Threat; 2]>,
    /// The empty cells grouped by the threat a stone there would create, per
    /// colour. `Threat::Nothing` is not tracked.
    threat_cells: [[BitBoard; Threat::COUNT]; 2],
    /// The threat values of all empty cells added up, per colour.
    total: [Score; 2],
    outcome: Option<Outcome>,
    undo_stack: Vec<UndoFrame>,
}

/// Returns `true` if a u16 line mask contains 5 or more consecutive set bits.
fn has_five_consecutive(mask: u16) -> bool {
    let c2 = mask & (mask >> 1);
    let c4 = c2 & (c2 >> 2);
    let c5 = c4 & (mask >> 4);
    c5 != 0
}

/// Marks the board positions behind the `cells` bits of `line`.
fn mark_line_cells(line: usize, mut cells: u16, into: &mut BitBoard) {
    while cells != 0 {
        let bit = usize::try_from(cells.trailing_zeros()).expect("bit index fits in usize");
        cells &= cells - 1;
        into.set(LINE_CELLS[line][bit]);
    }
}

impl SearchState {
    /// Reads every cell's patterns from an existing board.
    #[must_use]
    pub fn from_board(board: &Board) -> Self {
        let mut state = Self {
            hash: 0,
            board: *board,
            line_black: [0; NUM_LINES],
            line_white: [0; NUM_LINES],
            patterns: PositionArray::new([CellPatterns::NONE; 2]),
            threats: PositionArray::new([Threat::Nothing; 2]),
            threat_cells: [[BitBoard::EMPTY; Threat::COUNT]; 2],
            total: [Score::DRAW; 2],
            outcome: board.outcome(),
            undo_stack: Vec::new(),
        };

        for position in PositionId::iter() {
            if let Some(stone) = board.stone(position) {
                state.hash ^= ZOBRIST[usize::from(position)][usize::from(stone)];
                for &(line_id, bit) in POSITION_LINES.get(position) {
                    state.line_masks_mut(stone)[usize::from(line_id)] |= 1 << bit;
                }
            }
        }

        let (table, lines) = (threat_table(), line_pattern_table());
        for position in PositionId::iter() {
            if board.is_empty(position) {
                for colour in [Stone::Black, Stone::White] {
                    let mut patterns = CellPatterns::NONE;
                    for direction in 0..4 {
                        let pattern = state.line_pattern_at(lines, position, direction, colour);
                        patterns = patterns.with(direction, pattern);
                    }
                    state.set_patterns(position, colour, patterns, table);
                }
            }
        }
        state
    }

    /// Places a stone and rereads the patterns of the cells within its reach.
    ///
    /// Uses `Board::place_unchecked` to skip validation and the expensive
    /// `has_five_in_a_row` bitboard scan. Wins are detected cheaply by
    /// checking 5-consecutive bits on the placed stone's line masks.
    ///
    /// Saves the old patterns onto an internal stack so that
    /// [`undo`](Self::undo) can restore them without recomputing.
    pub fn place(&mut self, position: PositionId, stone: Stone) {
        self.board.place_unchecked(position, stone);
        self.hash ^= ZOBRIST[usize::from(position)][usize::from(stone)];

        let mut won = false;
        for &(line_id, bit) in POSITION_LINES.get(position) {
            let mask = &mut self.line_masks_mut(stone)[usize::from(line_id)];
            *mask |= 1 << bit;
            won = won || has_five_consecutive(*mask);
        }

        let mut frame = UndoFrame {
            patterns: [[CellPatterns::NONE; 2]; CHANGED_CELLS],
        };
        let (table, lines) = (threat_table(), line_pattern_table());
        // The cell itself is taken: it no longer offers anything to anyone.
        frame.patterns[0] = *self.patterns.get(position);
        for colour in [Stone::Black, Stone::White] {
            self.set_patterns(position, colour, CellPatterns::NONE, table);
        }
        // Each neighbour shares exactly one line with the stone, so only
        // that line's pattern can have changed.
        for (i, &(cell, direction)) in NEIGHBOURHOODS.get(position).cells().iter().enumerate() {
            frame.patterns[i + 1] = *self.patterns.get(cell);
            if !self.board.is_empty(cell) {
                continue;
            }
            let direction = usize::from(direction);
            for colour in [Stone::Black, Stone::White] {
                let pattern = self.line_pattern_at(lines, cell, direction, colour);
                let patterns =
                    self.patterns.get(cell)[usize::from(colour)].with(direction, pattern);
                self.set_patterns(cell, colour, patterns, table);
            }
        }

        if won {
            self.outcome = Some(Outcome::Win(stone));
        } else if self.board.is_full() {
            self.outcome = Some(Outcome::Draw);
        }

        self.undo_stack.push(frame);
    }

    /// Takes a stone back, restoring the saved patterns of the cells it had
    /// changed.
    pub fn undo(&mut self, position: PositionId, stone: Stone) {
        self.board.undo(position, stone);
        self.hash ^= ZOBRIST[usize::from(position)][usize::from(stone)];

        for &(line_id, bit) in POSITION_LINES.get(position) {
            self.line_masks_mut(stone)[usize::from(line_id)] &= !(1 << bit);
        }

        let frame = self.undo_stack.pop().expect("undo without matching place");
        let table = threat_table();
        for colour in [Stone::Black, Stone::White] {
            self.set_patterns(
                position,
                colour,
                frame.patterns[0][usize::from(colour)],
                table,
            );
        }
        for (i, &(cell, _)) in NEIGHBOURHOODS.get(position).cells().iter().enumerate() {
            for colour in [Stone::Black, Stone::White] {
                self.set_patterns(
                    cell,
                    colour,
                    frame.patterns[i + 1][usize::from(colour)],
                    table,
                );
            }
        }
        self.outcome = None;
    }

    /// O(1) evaluation from the perspective of `to_move`: what its empty
    /// cells offer it, less what they offer the opponent, plus its best cell
    /// once more because it gets to play that one now. Without that tempo
    /// term the side that just moved always looks ahead, and lines that end
    /// on a forced reply would be judged by who happened to move last. Fours
    /// and unstoppable fours are never scored here; the search settles them
    /// by playing them out (see [`Situation`]).
    #[must_use]
    pub fn evaluate(&self, to_move: Stone) -> Score {
        self.total[usize::from(to_move)] - self.total[usize::from(to_move.opponent())]
            + self.best_threat(to_move).value()
    }

    /// The strongest threat any empty cell offers `colour`.
    fn best_threat(&self, colour: Stone) -> Threat {
        let cells = &self.threat_cells[usize::from(colour)];
        (1..Threat::COUNT)
            .rev()
            .find(|&index| cells[index].any())
            .map_or(Threat::Nothing, Threat::from_index)
    }

    /// Classifies the position for the side about to play.
    #[must_use]
    pub fn situation(&self, to_move: Stone) -> Situation {
        let own = &self.threat_cells[usize::from(to_move)];
        let opponent = &self.threat_cells[usize::from(to_move.opponent())];
        if own[Threat::Five.index()].any() {
            Situation::CompleteFive
        } else if opponent[Threat::Five.index()].any() {
            Situation::BlockFour
        } else if own[Threat::UnstoppableFour.index()].any() {
            Situation::MakeUnstoppableFour
        } else if opponent[Threat::UnstoppableFour.index()].any() {
            Situation::PreventUnstoppableFour
        } else {
            Situation::Develop
        }
    }

    /// Writes the moves worth searching for `to_move` in `situation` into
    /// `buf`: the forced replies first, then the chosen moves, the most
    /// promising first. Every returned position is empty.
    pub fn candidates(
        &self,
        to_move: Stone,
        situation: Situation,
        buf: &mut [PositionId; PositionId::COUNT],
    ) -> Candidates {
        let own = &self.threat_cells[usize::from(to_move)];
        let opponent = &self.threat_cells[usize::from(to_move.opponent())];

        let mut forced = BitBoard::EMPTY;
        let mut chosen = BitBoard::EMPTY;
        match situation {
            Situation::CompleteFive => chosen = own[Threat::Five.index()],
            Situation::BlockFour => forced = opponent[Threat::Five.index()],
            Situation::MakeUnstoppableFour => chosen = own[Threat::UnstoppableFour.index()],
            Situation::PreventUnstoppableFour => {
                // Taking the cell always works. An open four in the making
                // is also stopped anywhere along its line; two fours in the
                // making are stopped by breaking either one.
                forced = opponent[Threat::UnstoppableFour.index()];
                let their_masks = self.line_masks(to_move.opponent());
                for cell in forced.iter_set() {
                    let patterns = self.patterns.get(cell)[usize::from(to_move.opponent())];
                    let grows_open_four =
                        (0..4).any(|direction| patterns.get(direction) == LinePattern::OpenFour);
                    for (direction, &(line_id, bit)) in POSITION_LINES.get(cell).iter().enumerate()
                    {
                        let line = usize::from(line_id);
                        let (own, empty) = (their_masks[line], self.empty_cells(line));
                        let cells = match patterns.get(direction) {
                            LinePattern::OpenFour => defusing_cells(own, empty),
                            LinePattern::Four if !grows_open_four => {
                                four_breaking_cells(own, empty, 1 << bit)
                            }
                            _ => 0,
                        };
                        mark_line_cells(line, cells, &mut forced);
                    }
                }
                // A counter-four has to be blocked before the threat can be
                // carried out, so it is a reply too, but a chosen one: it
                // costs depth like any attack.
                chosen = (own[Threat::Four.index()]
                    | own[Threat::FourAndMore.index()]
                    | own[Threat::FourAndOpenThree.index()])
                    & !forced;
            }
            Situation::Develop => return self.developing_moves(buf),
        }

        let mut count = 0;
        for position in forced.iter_set() {
            buf[count] = position;
            count += 1;
        }
        let forced = count;
        for position in chosen.iter_set() {
            buf[count] = position;
            count += 1;
        }
        Candidates { forced, count }
    }

    /// Every empty cell within reach of a stone, the ones offering most to
    /// either side first; the centre on an empty board.
    fn developing_moves(&self, buf: &mut [PositionId; PositionId::COUNT]) -> Candidates {
        if self.board.move_count() == 0 {
            buf[0] = PositionId::center();
            return Candidates {
                forced: 0,
                count: 1,
            };
        }
        let occupied = *self.board.bitboard(Stone::Black) | *self.board.bitboard(Stone::White);
        let nearby = occupied.expand_nearby() & !occupied;

        let mut scored = [(Score::DRAW, PositionId::default()); PositionId::COUNT];
        let mut count = 0;
        for position in nearby.iter_set() {
            let [black, white] = *self.threats.get(position);
            scored[count] = (black.value() + white.value(), position);
            count += 1;
        }
        scored[..count].sort_unstable_by_key(|&(value, _)| Reverse(value));
        for (slot, &(_, position)) in buf.iter_mut().zip(&scored[..count]) {
            *slot = position;
        }
        Candidates { forced: 0, count }
    }

    /// What one more `colour` stone at `cell` would make on the line in
    /// `direction`.
    #[inline]
    fn line_pattern_at(
        &self,
        table: &LinePatternTable,
        cell: PositionId,
        direction: usize,
        colour: Stone,
    ) -> LinePattern {
        let (line_id, bit) = POSITION_LINES.get(cell)[direction];
        let line = usize::from(line_id);
        line_pattern_from(
            table,
            self.line_masks(colour)[line],
            self.line_masks(colour.opponent())[line],
            LINE_LENGTHS[line],
            bit,
        )
    }

    /// Records new patterns for `cell`, moving its threat value and its
    /// threat-class membership along.
    #[inline]
    fn set_patterns(
        &mut self,
        cell: PositionId,
        colour: Stone,
        patterns: CellPatterns,
        table: &ThreatTable,
    ) {
        let side = usize::from(colour);
        let old = std::mem::replace(&mut self.patterns.get_mut(cell)[side], patterns);
        if old == patterns {
            return;
        }
        let new_threat = patterns.threat_from(table);
        let old_threat = std::mem::replace(&mut self.threats.get_mut(cell)[side], new_threat);
        self.total[side] += new_threat.value() - old_threat.value();
        if old_threat != new_threat {
            if old_threat != Threat::Nothing {
                self.threat_cells[side][old_threat.index()].clear(cell);
            }
            if new_threat != Threat::Nothing {
                self.threat_cells[side][new_threat.index()].set(cell);
            }
        }
    }

    fn line_masks(&self, stone: Stone) -> &[u16; NUM_LINES] {
        match stone {
            Stone::Black => &self.line_black,
            Stone::White => &self.line_white,
        }
    }

    fn line_masks_mut(&mut self, stone: Stone) -> &mut [u16; NUM_LINES] {
        match stone {
            Stone::Black => &mut self.line_black,
            Stone::White => &mut self.line_white,
        }
    }

    fn empty_cells(&self, line: usize) -> u16 {
        let valid = (1u16 << LINE_LENGTHS[line]) - 1;
        !(self.line_black[line] | self.line_white[line]) & valid
    }

    #[must_use]
    pub fn hash(&self) -> u64 {
        self.hash
    }

    #[must_use]
    pub fn outcome(&self) -> Option<Outcome> {
        self.outcome
    }

    #[must_use]
    pub fn move_count(&self) -> usize {
        self.board.move_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;

    fn pos(row: usize, col: usize) -> PositionId {
        PositionId::from_position(Position::new(row, col))
    }

    fn board_with(black: &[(usize, usize)], white: &[(usize, usize)]) -> Board {
        let mut board = Board::new();
        for &(row, col) in black {
            board.place(pos(row, col), Stone::Black).unwrap();
        }
        for &(row, col) in white {
            board.place(pos(row, col), Stone::White).unwrap();
        }
        board
    }

    /// The candidates for `to_move`, sorted, and how many of them are forced.
    fn candidates_of(state: &SearchState, to_move: Stone) -> (Vec<PositionId>, usize) {
        let mut buf = [PositionId::default(); PositionId::COUNT];
        let candidates = state.candidates(to_move, state.situation(to_move), &mut buf);
        let mut moves = buf[..candidates.count].to_vec();
        moves.sort_by_key(|position| position.to_index());
        (moves, candidates.forced)
    }

    fn positions(cells: &[(usize, usize)]) -> Vec<PositionId> {
        let mut result: Vec<PositionId> = cells.iter().map(|&(r, c)| pos(r, c)).collect();
        result.sort_by_key(|position| position.to_index());
        result
    }

    /// Everything derived from the stones, for comparing two states.
    fn snapshot(state: &SearchState) -> (Vec<[CellPatterns; 2]>, [Score; 2], Vec<Vec<PositionId>>) {
        let patterns = PositionId::iter().map(|p| *state.patterns.get(p)).collect();
        let cells = state
            .threat_cells
            .iter()
            .flatten()
            .map(|board| board.iter_set().collect())
            .collect();
        (patterns, state.total, cells)
    }

    #[test]
    fn empty_board_evaluates_to_zero() {
        let state = SearchState::from_board(&Board::new());

        assert_eq!(state.evaluate(Stone::Black), Score::DRAW);
        assert_eq!(state.evaluate(Stone::White), Score::DRAW);
    }

    #[test]
    fn evaluate_favours_the_side_whose_cells_offer_more() {
        let board = board_with(&[(7, 6), (7, 7)], &[(0, 0)]);
        let state = SearchState::from_board(&board);

        // Black's open two gives several cells an open three; White's corner
        // stone gives almost nothing. Black to move also gets its best cell
        // counted once more.
        assert!(state.evaluate(Stone::Black) > Score::DRAW);
        assert_eq!(
            state.evaluate(Stone::Black) + state.evaluate(Stone::White),
            Threat::OpenThree.value() + state.best_threat(Stone::White).value()
        );
    }

    #[test]
    fn incremental_place_matches_from_scratch() {
        let mut rng = fastrand::Rng::with_seed(54321);
        for _ in 0..200 {
            let mut board = Board::new();
            let mut state = SearchState::from_board(&board);
            let stone_count = rng.usize(1..25);
            let mut positions: Vec<PositionId> = PositionId::iter().collect();
            rng.shuffle(&mut positions);
            let mut turn = Stone::Black;
            for &position in positions.iter().take(stone_count) {
                if board.is_finished() || state.outcome().is_some() {
                    break;
                }
                board.place(position, turn).unwrap();
                state.place(position, turn);

                let expected = SearchState::from_board(&board);
                assert_eq!(
                    snapshot(&state),
                    snapshot(&expected),
                    "mismatch after placing at ({}, {}):\n{board}",
                    position.row(),
                    position.col()
                );
                turn = turn.opponent();
            }
        }
    }

    #[test]
    fn undo_restores_everything() {
        let mut rng = fastrand::Rng::with_seed(12345);
        for _ in 0..200 {
            let mut board = Board::new();
            let mut state = SearchState::from_board(&board);
            let setup_count = rng.usize(0..20);
            let mut positions: Vec<PositionId> = PositionId::iter().collect();
            rng.shuffle(&mut positions);
            let mut turn = Stone::Black;

            for &position in positions.iter().take(setup_count) {
                if board.is_finished() {
                    break;
                }
                board.place(position, turn).unwrap();
                state.place(position, turn);
                turn = turn.opponent();
            }
            if board.is_finished() {
                continue;
            }

            let before = snapshot(&state);
            let masks_before = (state.line_black, state.line_white, state.hash);
            let empty: Vec<PositionId> =
                PositionId::iter().filter(|&p| board.is_empty(p)).collect();
            let target = empty[rng.usize(0..empty.len())];

            state.place(target, turn);
            state.undo(target, turn);

            assert_eq!(snapshot(&state), before);
            assert_eq!(
                (state.line_black, state.line_white, state.hash),
                masks_before
            );
        }
    }

    #[test]
    fn nested_place_undo_restores_correctly() {
        let board = board_with(&[(7, 7)], &[]);
        let mut state = SearchState::from_board(&board);
        let snapshot_0 = snapshot(&state);

        state.place(pos(7, 8), Stone::White);
        let snapshot_1 = snapshot(&state);

        state.place(pos(6, 6), Stone::Black);

        state.undo(pos(6, 6), Stone::Black);
        assert_eq!(snapshot(&state), snapshot_1, "after undoing second place");

        state.undo(pos(7, 8), Stone::White);
        assert_eq!(snapshot(&state), snapshot_0, "after undoing first place");
    }

    #[test]
    fn situation_ranks_own_four_above_everything() {
        // Black OXXXX_ on row 7; White _OOO_ open three on row 3.
        let board = board_with(
            &[(7, 6), (7, 7), (7, 8), (7, 9)],
            &[(7, 5), (3, 5), (3, 6), (3, 7)],
        );
        let state = SearchState::from_board(&board);

        assert_eq!(state.situation(Stone::Black), Situation::CompleteFive);
        assert_eq!(state.situation(Stone::White), Situation::BlockFour);
    }

    #[test]
    fn situation_ranks_blocking_a_four_above_making_an_open_four() {
        // Black _XXX_ on row 7; White OOOO_ on row 3, hemmed by Black at (3,1).
        let board = board_with(
            &[(7, 6), (7, 7), (7, 8), (3, 1)],
            &[(3, 2), (3, 3), (3, 4), (3, 5)],
        );
        let state = SearchState::from_board(&board);

        assert_eq!(state.situation(Stone::Black), Situation::BlockFour);
    }

    #[test]
    fn situation_with_open_threes_on_both_sides_favours_the_side_to_move() {
        let board = board_with(&[(7, 6), (7, 7), (7, 8)], &[(3, 5), (3, 6), (3, 7)]);
        let state = SearchState::from_board(&board);

        assert_eq!(
            state.situation(Stone::Black),
            Situation::MakeUnstoppableFour
        );
        assert_eq!(
            state.situation(Stone::White),
            Situation::MakeUnstoppableFour
        );
    }

    #[test]
    fn situation_sees_two_crossing_threes_as_an_unstoppable_four() {
        // OXXX_ on row 7 and OXXX_ on column 9 meet at the empty (7,9): one
        // stone there makes two fours, and only one can be blocked.
        let board = board_with(
            &[(7, 6), (7, 7), (7, 8), (4, 9), (5, 9), (6, 9)],
            &[(7, 5), (3, 9)],
        );
        let state = SearchState::from_board(&board);

        assert_eq!(
            state.situation(Stone::Black),
            Situation::MakeUnstoppableFour
        );
        assert_eq!(
            candidates_of(&state, Stone::Black),
            (positions(&[(7, 9)]), 0)
        );
        assert_eq!(
            state.situation(Stone::White),
            Situation::PreventUnstoppableFour
        );
        // White can take the cell or the cell that would complete either four.
        let (replies, forced) = candidates_of(&state, Stone::White);
        assert_eq!(forced, 3);
        for cell in [pos(7, 9), pos(7, 10), pos(8, 9)] {
            assert!(replies.contains(&cell), "{cell:?}");
        }
    }

    #[test]
    fn situation_answers_an_open_three_and_develops_otherwise() {
        let board = board_with(&[(7, 6), (7, 7), (7, 8)], &[(0, 0), (0, 14)]);
        let state = SearchState::from_board(&board);

        assert_eq!(
            state.situation(Stone::White),
            Situation::PreventUnstoppableFour
        );

        let quiet = board_with(&[(7, 7)], &[(7, 8)]);
        let quiet_state = SearchState::from_board(&quiet);
        assert_eq!(quiet_state.situation(Stone::Black), Situation::Develop);
        assert_eq!(quiet_state.situation(Stone::White), Situation::Develop);
    }

    #[test]
    fn candidates_for_a_four_are_the_cells_that_complete_it() {
        // XX_XX on row 7 and _XXXX_ on column 3; White has nothing.
        let board = board_with(
            &[
                (7, 4),
                (7, 5),
                (7, 7),
                (7, 8),
                (3, 3),
                (4, 3),
                (5, 3),
                (6, 3),
            ],
            &[(0, 0), (0, 14)],
        );
        let state = SearchState::from_board(&board);

        let completing = positions(&[(7, 6), (2, 3), (7, 3)]);
        // Completing five is a choice; blocking is forced.
        assert_eq!(candidates_of(&state, Stone::Black), (completing.clone(), 0));
        assert_eq!(candidates_of(&state, Stone::White), (completing, 3));
    }

    #[test]
    fn candidates_for_making_an_open_four_are_the_cells_that_make_one() {
        // O_XXX__ on row 7: only (7,9) makes _XXXX_.
        let board = board_with(&[(7, 6), (7, 7), (7, 8)], &[(7, 4), (0, 0)]);
        let state = SearchState::from_board(&board);

        assert_eq!(
            state.situation(Stone::Black),
            Situation::MakeUnstoppableFour
        );
        assert_eq!(
            candidates_of(&state, Stone::Black),
            (positions(&[(7, 9)]), 0)
        );
    }

    #[test]
    fn candidates_against_an_open_three_are_its_defences_and_own_counter_fours() {
        // Black _XXX_ on row 7 with room on both sides. White has OOO on
        // row 3 hemmed at (3,4): (3,8) and (3,9) each make a four.
        let board = board_with(
            &[(7, 6), (7, 7), (7, 8), (3, 4)],
            &[(3, 5), (3, 6), (3, 7), (0, 0)],
        );
        let state = SearchState::from_board(&board);

        assert_eq!(
            state.situation(Stone::White),
            Situation::PreventUnstoppableFour
        );
        // The two defences are forced; the two counter-fours are choices.
        assert_eq!(
            candidates_of(&state, Stone::White),
            (positions(&[(7, 5), (7, 9), (3, 8), (3, 9)]), 2)
        );
    }

    #[test]
    fn candidates_against_a_one_sided_open_three_include_the_far_cell() {
        // O_XXX__ on row 7: the open four can only grow to the right, so
        // (7,10) stops it as well as the neighbours (7,5) and (7,9).
        let board = board_with(&[(7, 6), (7, 7), (7, 8)], &[(7, 4)]);
        let state = SearchState::from_board(&board);

        assert_eq!(
            candidates_of(&state, Stone::White),
            (positions(&[(7, 5), (7, 9), (7, 10)]), 3)
        );
    }

    #[test]
    fn candidates_against_two_separate_open_threes_take_their_cells() {
        // Two open threes on row 7 with nothing in common: _XXX___XXX_. No
        // single stone defuses both, so the replies are the four cells that
        // would make an open four; the search then finds the loss.
        let board = board_with(
            &[(7, 2), (7, 3), (7, 4), (7, 8), (7, 9), (7, 10)],
            &[(0, 0), (0, 14)],
        );
        let state = SearchState::from_board(&board);

        assert_eq!(
            state.situation(Stone::White),
            Situation::PreventUnstoppableFour
        );
        assert_eq!(
            candidates_of(&state, Stone::White),
            (positions(&[(7, 1), (7, 5), (7, 7), (7, 11)]), 4)
        );
    }

    #[test]
    fn forced_replies_include_every_move_that_removes_the_threat() {
        // Over random positions where the side to move must prevent an
        // unstoppable four or block a four, every empty cell whose stone
        // leaves the opponent without such a cell must be offered as forced.
        let mut rng = fastrand::Rng::with_seed(777);
        let mut checked = 0;
        for _ in 0..600 {
            let mut board = Board::new();
            let mut turn = Stone::Black;
            for _ in 0..rng.usize(4..30) {
                let occupied = *board.bitboard(Stone::Black) | *board.bitboard(Stone::White);
                let nearby: Vec<PositionId> = if board.move_count() == 0 {
                    vec![PositionId::center()]
                } else {
                    (occupied.expand_nearby() & !occupied).iter_set().collect()
                };
                let position = nearby[rng.usize(..nearby.len())];
                board.place(position, turn).unwrap();
                if board.outcome().is_some() {
                    break;
                }
                turn = turn.opponent();
            }
            if board.outcome().is_some() {
                continue;
            }
            let mut state = SearchState::from_board(&board);
            let situation = state.situation(turn);
            let threat = match situation {
                Situation::BlockFour => Threat::Five,
                Situation::PreventUnstoppableFour => Threat::UnstoppableFour,
                _ => continue,
            };
            let (moves, forced_count) = candidates_of(&state, turn);
            let mut buf = [PositionId::default(); PositionId::COUNT];
            let candidates = state.candidates(turn, situation, &mut buf);
            let forced: Vec<PositionId> = buf[..candidates.forced].to_vec();
            assert_eq!(forced_count, candidates.forced);
            assert!(moves.len() >= forced.len());
            let opponent = usize::from(turn.opponent());
            for cell in PositionId::iter().filter(|&p| board.is_empty(p)) {
                state.place(cell, turn);
                let removed = !state.threat_cells[opponent][threat.index()].any();
                state.undo(cell, turn);
                if removed {
                    assert!(
                        forced.contains(&cell),
                        "({}, {}) removes the threat but is not offered in {situation:?}:\n{board}",
                        cell.row(),
                        cell.col()
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 50, "too few positions exercised: {checked}");
    }

    #[test]
    fn candidates_when_developing_stay_near_the_stones_and_lead_with_threats() {
        // Black _XX_ on row 7: the cells extending it come first.
        let board = board_with(&[(7, 6), (7, 7)], &[(3, 3)]);
        let state = SearchState::from_board(&board);

        let mut buf = [PositionId::default(); PositionId::COUNT];
        let candidates = state.candidates(Stone::White, Situation::Develop, &mut buf);
        assert_eq!(candidates.forced, 0);
        assert!(candidates.count > 20);
        // The four cells that give Black an open three (two straight, two
        // split) outrank everything else, in some order.
        let mut leaders: Vec<PositionId> = buf[..4].to_vec();
        leaders.sort_by_key(|position| position.to_index());
        assert_eq!(leaders, positions(&[(7, 4), (7, 5), (7, 8), (7, 9)]));
        for &candidate in &buf[..candidates.count] {
            assert!(board.is_empty(candidate));
        }
    }

    #[test]
    fn candidates_on_an_empty_board_are_the_centre() {
        let state = SearchState::from_board(&Board::new());

        assert_eq!(
            candidates_of(&state, Stone::Black),
            (vec![PositionId::center()], 0)
        );
    }

    #[test]
    fn situation_follows_a_four_through_place_block_and_undo() {
        let board = board_with(&[(7, 3), (7, 4), (7, 5)], &[(7, 2)]);
        let mut state = SearchState::from_board(&board);
        assert_eq!(state.situation(Stone::Black), Situation::Develop);

        // OXXXX_ is a four for Black.
        state.place(pos(7, 6), Stone::Black);
        assert_eq!(state.situation(Stone::Black), Situation::CompleteFive);
        assert_eq!(state.situation(Stone::White), Situation::BlockFour);

        // OXXXXO is dead: no longer a four.
        state.place(pos(7, 7), Stone::White);
        assert_eq!(state.situation(Stone::Black), Situation::Develop);

        state.undo(pos(7, 7), Stone::White);
        assert_eq!(state.situation(Stone::White), Situation::BlockFour);

        state.undo(pos(7, 6), Stone::Black);
        assert_eq!(state.situation(Stone::Black), Situation::Develop);
    }

    #[test]
    fn detects_black_win() {
        let board = board_with(&[(7, 3), (7, 4), (7, 5), (7, 6)], &[(0, 0), (0, 1)]);

        let mut state = SearchState::from_board(&board);
        assert!(state.outcome().is_none());

        state.place(pos(7, 7), Stone::Black);

        assert_eq!(state.outcome(), Some(Outcome::Win(Stone::Black)));
    }

    #[test]
    fn detects_white_win() {
        let board = board_with(&[(0, 0)], &[(3, 7), (4, 7), (5, 7), (6, 7)]);

        let mut state = SearchState::from_board(&board);
        assert!(state.outcome().is_none());

        state.place(pos(7, 7), Stone::White);

        assert_eq!(state.outcome(), Some(Outcome::Win(Stone::White)));
    }

    #[test]
    fn undo_clears_win_outcome() {
        let board = board_with(&[(7, 3), (7, 4), (7, 5), (7, 6)], &[(0, 0)]);

        let mut state = SearchState::from_board(&board);

        state.place(pos(7, 7), Stone::Black);
        assert_eq!(state.outcome(), Some(Outcome::Win(Stone::Black)));

        state.undo(pos(7, 7), Stone::Black);
        assert!(state.outcome().is_none());
    }

    #[test]
    fn outcome_agrees_with_board_across_random_games() {
        let mut rng = fastrand::Rng::with_seed(99999);
        for _ in 0..200 {
            let mut board = Board::new();
            let mut state = SearchState::from_board(&board);
            let mut positions: Vec<PositionId> = PositionId::iter().collect();
            rng.shuffle(&mut positions);
            let mut turn = Stone::Black;

            for &position in &positions {
                if board.is_finished() {
                    break;
                }
                board.place(position, turn).unwrap();
                state.place(position, turn);

                assert_eq!(
                    state.outcome(),
                    board.outcome(),
                    "outcome mismatch after placing at ({}, {})",
                    position.row(),
                    position.col()
                );
                turn = turn.opponent();
            }
        }
    }
}
