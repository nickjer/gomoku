//! What one more stone at an empty cell would make: first on each of the
//! cell's four lines, then combined into one code for the cell.
//!
//! This is the classical evaluation of Rapfi (Gomocup 2018), ported line for
//! line: sixteen line patterns read from the eight cells within reach, the
//! four patterns of a cell folded into one of 3876 unordered combinations,
//! and Rapfi's tuned tables giving each combination its value and its
//! move-ordering score (see [`super::rapfi_tables`]).

use std::sync::LazyLock;

use super::lines::REACH;
use super::rapfi_tables::{PATTERN_CODES, SCORE, VALUE};
use super::score::Score;

/// What one more own stone at an empty cell makes on one line through it,
/// weakest first, in Rapfi's sixteen kinds. "Blocked" shapes cannot grow
/// into the free shape one size up; "split" shapes have a gap inside. The
/// numbering is Rapfi's and indexes its tables, so it must not change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum LinePattern {
    /// No room left for five (Rapfi `DEAD`).
    Dead = 0,
    /// One stone with room for five but not for a free shape (`B1`).
    BlockedOne,
    /// One stone with room to grow freely (`F1`).
    FreeOne,
    /// Two stones with at most one gap, blocked (`B2J0`).
    BlockedTwo,
    /// Two stones two or three apart, blocked (`B2J2`).
    BlockedTwoWide,
    /// Two adjacent stones, free (`F2J0`).
    FreeTwo,
    /// Two stones with one gap, free (`F2J1`).
    FreeTwoSplit,
    /// Two stones with two gaps, free (`F2J2`).
    FreeTwoWideSplit,
    /// Three adjacent stones, blocked: one step from a four (`B3J0`).
    BlockedThree,
    /// Three stones with one gap, blocked (`B3J1`).
    BlockedThreeSplit,
    /// Three stones with two gaps, blocked (`B3J2`).
    BlockedThreeWideSplit,
    /// Three adjacent stones, free: one step from an open four (`F3J0`).
    FreeThree,
    /// Three stones with one gap, free (`F3J1`).
    FreeThreeSplit,
    /// Four stones with one cell completing five (`B4`).
    BlockedFour,
    /// Four stones with two cells completing five: unstoppable (`F4`).
    FreeFour,
    /// Five in a row (`F5`).
    Five,
}

impl LinePattern {
    pub const COUNT: usize = 16;

    const ALL: [Self; Self::COUNT] = [
        Self::Dead,
        Self::BlockedOne,
        Self::FreeOne,
        Self::BlockedTwo,
        Self::BlockedTwoWide,
        Self::FreeTwo,
        Self::FreeTwoSplit,
        Self::FreeTwoWideSplit,
        Self::BlockedThree,
        Self::BlockedThreeSplit,
        Self::BlockedThreeWideSplit,
        Self::FreeThree,
        Self::FreeThreeSplit,
        Self::BlockedFour,
        Self::FreeFour,
        Self::Five,
    ];

    // A nibble holds 0..16 and a discriminant is its own index; `From` is
    // not usable in a const fn.
    #[allow(clippy::as_conversions)]
    const fn from_nibble(nibble: u16) -> Self {
        Self::ALL[nibble as usize]
    }

    #[allow(clippy::as_conversions)]
    const fn nibble(self) -> u16 {
        self as u16
    }

    /// Position in the weakest-first order, for comparing in a const fn.
    #[allow(clippy::as_conversions)]
    const fn rank(self) -> u8 {
        self as u8
    }

    const fn is_blocked_three(self) -> bool {
        matches!(
            self,
            Self::BlockedThree | Self::BlockedThreeSplit | Self::BlockedThreeWideSplit
        )
    }

    const fn is_free_three(self) -> bool {
        matches!(self, Self::FreeThree | Self::FreeThreeSplit)
    }

    const fn is_free_two(self) -> bool {
        matches!(
            self,
            Self::FreeTwo | Self::FreeTwoSplit | Self::FreeTwoWideSplit
        )
    }
}

// ── The line classifier, ported from Rapfi's Evaluator.cpp ──────────────────

/// One cell of the nine-cell line the classifier reads, the centre being the
/// stone about to be placed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cell {
    Own,
    Other,
    Empty,
}

/// The nine cells along one line, the centre at index `REACH`.
type Line = [Cell; 2 * REACH + 1];

/// Rapfi's `getType`: the pattern for a shape of `count` stones (the centre
/// included) spanning `full_length` cells with `jump` gaps inside, inside
/// `length` cells of room, `blocked` when a stone of the shape touches the
/// opponent or the edge.
const fn shape_pattern(
    length: usize,
    full_length: usize,
    count: usize,
    blocked: bool,
    jump: usize,
) -> LinePattern {
    use LinePattern as P;
    if length < 5 {
        return P::Dead;
    }
    if count >= 5 {
        return P::Five;
    }
    if length > 5 && full_length < 5 && !blocked {
        match (count, jump) {
            (1, _) => P::FreeOne,
            (2, 0) => P::FreeTwo,
            (2, 1) => P::FreeTwoSplit,
            (2, 2) => P::FreeTwoWideSplit,
            (3, 0) => P::FreeThree,
            (3, 1) => P::FreeThreeSplit,
            (4, _) => P::FreeFour,
            _ => P::Dead,
        }
    } else {
        match (count, jump) {
            (1, _) => P::BlockedOne,
            (2, 0 | 1) => P::BlockedTwo,
            (2, 2 | 3) => P::BlockedTwoWide,
            (3, 0) => P::BlockedThree,
            (3, 1) => P::BlockedThreeSplit,
            (3, 2) => P::BlockedThreeWideSplit,
            (4, _) => P::BlockedFour,
            _ => P::Dead,
        }
    }
}

/// Rapfi's `shortLinePattern`: reads the shape around the centre, first to
/// the right and then to the left, and classifies it.
const fn short_line_pattern(line: &Line) -> LinePattern {
    let mut empty = 0;
    let mut blocked = false;
    let mut length = 1;
    let mut full_length = 1;
    let mut count = 1;

    let mut i = REACH + 1;
    while i < line.len() {
        match line[i] {
            Cell::Own => {
                count += 1;
                length += 1;
                full_length = empty + count;
            }
            Cell::Empty => {
                length += 1;
                empty += 1;
            }
            Cell::Other => {
                if matches!(line[i - 1], Cell::Own) {
                    blocked = true;
                }
                break;
            }
        }
        i += 1;
    }

    // Only the gaps inside the shape count from here on.
    empty = full_length - count;
    let mut i = REACH;
    while i > 0 {
        i -= 1;
        match line[i] {
            Cell::Own => {
                if empty + count > 4 {
                    break;
                }
                count += 1;
                length += 1;
                full_length = empty + count;
            }
            Cell::Empty => {
                if empty + count > 4 {
                    break;
                }
                length += 1;
                empty += 1;
            }
            Cell::Other => {
                if matches!(line[i + 1], Cell::Own) {
                    blocked = true;
                }
                break;
            }
        }
    }
    shape_pattern(length, full_length, count, blocked, full_length - count)
}

/// Rapfi's `checkFive`: whether a stone at `index` would join at least four
/// own stones in a row.
const fn completes_five(line: &Line, index: usize) -> bool {
    let mut count = 0;
    let mut j = index;
    while j > 0 && matches!(line[j - 1], Cell::Own) {
        count += 1;
        j -= 1;
    }
    let mut j = index + 1;
    while j < line.len() && matches!(line[j], Cell::Own) {
        count += 1;
        j += 1;
    }
    count >= 4
}

/// Rapfi's `checkFlex4`: two blocked fours on one line make a free four when
/// two different cells complete five.
const fn merge_fours(line: &Line, first: LinePattern, second: LinePattern) -> LinePattern {
    if completes_five(line, REACH) {
        return LinePattern::Five;
    }
    let mut fives = 0;
    let mut i = 0;
    while i < line.len() {
        if matches!(line[i], Cell::Empty) && completes_five(line, i) {
            fives += 1;
        }
        i += 1;
    }
    if fives >= 2 {
        LinePattern::FreeFour
    } else {
        stronger(first, second)
    }
}

/// Rapfi's `checkFlex3`: two blocked threes on one line make a split free
/// three when some stone turns them into a free four.
const fn merge_threes(line: &Line, first: LinePattern, second: LinePattern) -> LinePattern {
    let mut trial = *line;
    let mut i = 0;
    while i < trial.len() {
        if matches!(trial[i], Cell::Empty) {
            trial[i] = Cell::Own;
            let merged = merge_fours(&trial, first, second);
            trial[i] = Cell::Empty;
            if merged.rank() >= LinePattern::FreeFour.rank() {
                return LinePattern::FreeThreeSplit;
            }
        }
        i += 1;
    }
    stronger(first, second)
}

const fn stronger(first: LinePattern, second: LinePattern) -> LinePattern {
    if first.rank() >= second.rank() {
        first
    } else {
        second
    }
}

/// Rapfi's `getPattern`: the line read both ways, the stronger reading
/// kept, with the two same-line double shapes merged.
const fn classify(own: Window, blocked: Window) -> LinePattern {
    let mut line = [Cell::Empty; 2 * REACH + 1];
    line[REACH] = Cell::Own;
    let mut i = 0;
    while i < 2 * REACH {
        let index = if i < REACH { i } else { i + 1 };
        line[index] = if own & (1 << i) != 0 {
            Cell::Own
        } else if blocked & (1 << i) != 0 {
            Cell::Other
        } else {
            Cell::Empty
        };
        i += 1;
    }
    let forwards = short_line_pattern(&line);
    line.reverse();
    let backwards = short_line_pattern(&line);

    if matches!(forwards, LinePattern::BlockedFour) && matches!(backwards, LinePattern::BlockedFour)
    {
        merge_fours(&line, forwards, backwards)
    } else if forwards.is_blocked_three() && backwards.is_blocked_three() {
        merge_threes(&line, forwards, backwards)
    } else {
        stronger(forwards, backwards)
    }
}

// ── Tables and lookups ───────────────────────────────────────────────────────

/// The eight cells within reach on one line, as bits 0..8 in line order, the
/// centre cell left out.
type Window = u8;

/// Bit `REACH` of a `2 * REACH + 1` cell window is the centre cell.
const CENTRE: u32 = 1 << REACH;

/// Drops the centre bit out of a nine-cell window read.
fn drop_centre(window: u32) -> Window {
    let low = window & (CENTRE - 1);
    let high = (window >> (REACH + 1)) & (CENTRE - 1);
    u8::try_from(low | (high << REACH)).expect("eight cells fit in a byte")
}

/// Line pattern of an empty centre cell by the own and blocked (opponent or
/// off-board) cells within reach: index `own << 8 | blocked`.
pub type LinePatternTable = [LinePattern; 1 << 16];

static LINE_PATTERNS: LazyLock<Box<LinePatternTable>> = LazyLock::new(|| {
    let mut table = vec![LinePattern::Dead; 1 << 16];
    for (key, entry) in table.iter_mut().enumerate() {
        let own = u8::try_from(key >> 8).expect("high byte");
        let blocked = u8::try_from(key & 0xFF).expect("low byte");
        if own & blocked == 0 {
            *entry = classify(own, blocked);
        }
    }
    table
        .into_boxed_slice()
        .try_into()
        .expect("table has 1 << 16 entries")
});

/// The line pattern table, built on first use.
#[must_use]
pub fn line_pattern_table() -> &'static LinePatternTable {
    &LINE_PATTERNS
}

/// What one more own stone at empty cell `bit` of a line of `length` cells
/// makes on that line, given the line's own and opponent stones.
#[cfg(test)]
#[must_use]
pub fn line_pattern(own: u16, opponent: u16, length: u8, bit: u8) -> LinePattern {
    line_pattern_from(line_pattern_table(), own, opponent, length, bit)
}

/// The same, from a table reference held across many lookups.
#[must_use]
#[inline]
pub fn line_pattern_from(
    table: &LinePatternTable,
    own: u16,
    opponent: u16,
    length: u8,
    bit: u8,
) -> LinePattern {
    // Shift the line so the cell sits at bit REACH; cells before the line
    // start and past its end read as blocked.
    let own_window = (u32::from(own) << REACH) >> bit;
    let blocked = u32::from(opponent) | (!0u32 << length);
    let blocked_window = ((blocked << REACH) | (CENTRE - 1)) >> bit;
    let key =
        (usize::from(drop_centre(own_window)) << 8) | usize::from(drop_centre(blocked_window));
    table[key]
}

/// The four line patterns of one cell for one colour, one nibble per line in
/// `POSITION_LINES` direction order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CellPatterns(u16);

impl CellPatterns {
    /// Dead on every line: an occupied cell, or an empty one with no room
    /// for five anywhere.
    pub const NONE: Self = Self(0);

    #[must_use]
    pub const fn get(self, direction: usize) -> LinePattern {
        LinePattern::from_nibble((self.0 >> (4 * direction)) & 0xF)
    }

    #[must_use]
    pub const fn with(self, direction: usize, pattern: LinePattern) -> Self {
        let shift = 4 * direction;
        Self((self.0 & !(0xF << shift)) | (pattern.nibble() << shift))
    }

    /// The code of the four patterns taken together, from a table
    /// reference held across many lookups.
    #[must_use]
    #[inline]
    pub fn code_from(self, table: &PatternCodeTable) -> PatternCode {
        table[usize::from(self.0)]
    }

    /// The code of the four patterns taken together.
    #[cfg(test)]
    #[must_use]
    pub fn code(self) -> PatternCode {
        self.code_from(pattern_code_table())
    }
}

/// One of the 3876 unordered combinations of a cell's four line patterns:
/// the index into Rapfi's tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PatternCode(u16);

impl PatternCode {
    /// What a cell with this code is worth to its colour.
    #[must_use]
    #[inline]
    pub fn value(self) -> Score {
        Score::new(i32::from(VALUE[usize::from(self.0)]))
    }

    /// How promising a stone at a cell with this code is as a move.
    #[must_use]
    #[inline]
    pub fn score(self) -> Score {
        Score::new(i32::from(SCORE[usize::from(self.0)]))
    }

    /// What a stone at a cell with this code would create.
    #[must_use]
    #[inline]
    pub fn threat(self) -> Threat {
        THREATS[usize::from(self.0)]
    }
}

/// Pattern code of every `CellPatterns` value, indexed by its packed nibbles.
pub type PatternCodeTable = [PatternCode; 1 << 16];

/// Rapfi's `init`: codes are handed out in the order the combinations first
/// appear when all ordered four-tuples are enumerated, highest nibble
/// first, so the tables line up.
static PATTERN_CODES_BY_NIBBLES: LazyLock<Box<PatternCodeTable>> = LazyLock::new(|| {
    let n = LinePattern::COUNT;
    let mut code_of_sorted: Vec<Option<PatternCode>> = vec![None; n * n * n * n];
    let mut next = 0u16;
    let mut table = vec![PatternCode::default(); 1 << 16];
    for first in 0..n {
        for second in 0..n {
            for third in 0..n {
                for fourth in 0..n {
                    let mut patterns = [first, second, third, fourth];
                    patterns.sort_unstable();
                    let sorted =
                        ((patterns[0] * n + patterns[1]) * n + patterns[2]) * n + patterns[3];
                    let code = *code_of_sorted[sorted].get_or_insert_with(|| {
                        let code = PatternCode(next);
                        next += 1;
                        code
                    });
                    // Our packing puts the first direction in the lowest nibble.
                    let key = first | (second << 4) | (third << 8) | (fourth << 12);
                    table[key] = code;
                }
            }
        }
    }
    assert_eq!(usize::from(next), PATTERN_CODES, "3876 combinations");
    table
        .into_boxed_slice()
        .try_into()
        .expect("table has 1 << 16 entries")
});

/// The pattern code table, built on first use.
#[must_use]
pub fn pattern_code_table() -> &'static PatternCodeTable {
    &PATTERN_CODES_BY_NIBBLES
}

/// What one more stone at a cell would create, its four lines taken
/// together, weakest first: Rapfi's `Pattern4` classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Threat {
    Nothing = 0,
    /// Free twos on two lines (`J_FLEX2_2X`).
    TwoFreeTwos,
    /// Two blocked threes, or one with a free two (`I_BLOCK3_PLUS`).
    BlockedThreeAndMore,
    /// A free three on one line (`H_FLEX3`).
    FreeThree,
    /// A free three plus a blocked three or a free two (`G_FLEX3_PLUS`).
    FreeThreeAndMore,
    /// Free threes on two lines: the opponent can stop only one (`F_FLEX3_2X`).
    TwoFreeThrees,
    /// A blocked four on one line (`E_BLOCK4`).
    Four,
    /// A blocked four plus a blocked three or a free two (`D_BLOCK4_PLUS`).
    FourAndMore,
    /// A blocked four and a free three: block the four and the open four
    /// follows (`C_BLOCK4_FLEX3`).
    FourAndFreeThree,
    /// A free four, or blocked fours on two lines: five follows whatever is
    /// blocked (`B_FLEX4`).
    UnstoppableFour,
    /// Five in a row (`A_FIVE`).
    Five,
}

impl Threat {
    pub const COUNT: usize = 11;

    /// Rapfi's `getPattern4`: what the patterns of a cell's four lines add
    /// up to.
    #[must_use]
    pub fn combine(patterns: [LinePattern; 4]) -> Self {
        use LinePattern as P;
        let count = |wanted: fn(P) -> bool| patterns.iter().filter(|&&p| wanted(p)).count();
        let fives = count(|p| p == P::Five);
        let free_fours = count(|p| p == P::FreeFour);
        let blocked_fours = count(|p| p == P::BlockedFour);
        let free_threes = count(P::is_free_three);
        let blocked_threes = count(P::is_blocked_three);
        let free_twos = count(P::is_free_two);

        if fives >= 1 {
            Self::Five
        } else if blocked_fours >= 2 || free_fours >= 1 {
            Self::UnstoppableFour
        } else if blocked_fours >= 1 {
            if free_threes >= 1 {
                Self::FourAndFreeThree
            } else if blocked_threes >= 1 || free_twos >= 1 {
                Self::FourAndMore
            } else {
                Self::Four
            }
        } else if free_threes >= 1 {
            if free_threes >= 2 {
                Self::TwoFreeThrees
            } else if blocked_threes >= 1 || free_twos >= 1 {
                Self::FreeThreeAndMore
            } else {
                Self::FreeThree
            }
        } else if blocked_threes >= 2 || (blocked_threes >= 1 && free_twos >= 1) {
            Self::BlockedThreeAndMore
        } else if free_twos >= 2 {
            Self::TwoFreeTwos
        } else {
            Self::Nothing
        }
    }

    // A discriminant is its own index; `From` is not usable in a const fn.
    #[allow(clippy::as_conversions)]
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// Threat of every pattern code.
static THREATS: LazyLock<Box<[Threat; PATTERN_CODES]>> = LazyLock::new(|| {
    let codes = pattern_code_table();
    let mut table = vec![Threat::Nothing; PATTERN_CODES];
    for key in 0..1u32 << 16 {
        let patterns = CellPatterns(u16::try_from(key).expect("16-bit key"));
        let code = patterns.code_from(codes);
        table[usize::from(code.0)] = Threat::combine([
            patterns.get(0),
            patterns.get(1),
            patterns.get(2),
            patterns.get(3),
        ]);
    }
    table
        .into_boxed_slice()
        .try_into()
        .expect("table has 3876 entries")
});

#[cfg(test)]
mod tests {
    use super::super::lines::{
        completing_cells, four_making_cells, open_four_making_cells, open_three_making_cells,
        three_making_cells,
    };
    use super::*;

    const FULL: u8 = 15;

    fn line(own_bits: &[u8], opponent_bits: &[u8]) -> (u16, u16) {
        let mask = |bits: &[u8]| bits.iter().fold(0u16, |m, &b| m | (1 << b));
        (mask(own_bits), mask(opponent_bits))
    }

    #[test]
    fn line_pattern_reads_the_shapes_around_the_cell() {
        use LinePattern as P;
        // e XXX with room on both sides: a free four.
        let (own, opp) = line(&[6, 7, 8], &[]);
        assert_eq!(line_pattern(own, opp, FULL, 5), P::FreeFour);
        // O e XXX _: a blocked four.
        let (own, opp) = line(&[6, 7, 8], &[4]);
        assert_eq!(line_pattern(own, opp, FULL, 5), P::BlockedFour);
        // XX e XX: completes five.
        let (own, opp) = line(&[3, 4, 6, 7], &[]);
        assert_eq!(line_pattern(own, opp, FULL, 5), P::Five);
        // _ e XX _ _: a free three.
        let (own, opp) = line(&[6, 7], &[]);
        assert_eq!(line_pattern(own, opp, FULL, 5), P::FreeThree);
        // _ e X _ X _: a split free three.
        let (own, opp) = line(&[6, 8], &[]);
        assert_eq!(line_pattern(own, opp, FULL, 5), P::FreeThreeSplit);
        // O e XX _ _: a blocked three.
        let (own, opp) = line(&[6, 7], &[4]);
        assert_eq!(line_pattern(own, opp, FULL, 5), P::BlockedThree);
        // _ e X _ _: a free two.
        let (own, opp) = line(&[6], &[]);
        assert_eq!(line_pattern(own, opp, FULL, 5), P::FreeTwo);
        // _ e _ X _: a split free two; e _ _ _ X: a wide split.
        let (own, opp) = line(&[7], &[]);
        assert_eq!(line_pattern(own, opp, FULL, 5), P::FreeTwoSplit);
        let (own, opp) = line(&[8], &[]);
        assert_eq!(line_pattern(own, opp, FULL, 5), P::FreeTwoWideSplit);
        // Alone with room: a free one; hemmed to four cells: dead.
        assert_eq!(line_pattern(0, 0, FULL, 5), P::FreeOne);
        let (own, opp) = line(&[], &[3, 7]);
        assert_eq!(line_pattern(own, opp, FULL, 5), P::Dead);
    }

    #[test]
    fn line_pattern_merges_two_shapes_on_one_line() {
        use LinePattern as P;
        // XXX _ e _ XXX: a blocked four either way, but the stone gives two
        // cells that complete five, so it is a free four.
        let (own, opp) = line(&[1, 2, 3, 7, 8, 9], &[]);
        assert_eq!(line_pattern(own, opp, FULL, 5), P::FreeFour);
        // XX _ e _ XX with both ends closed: two blocked threes that a second
        // stone would turn into that free four, so a split free three.
        let (own, opp) = line(&[2, 3, 7, 8], &[0, 10]);
        assert_eq!(line_pattern(own, opp, FULL, 5), P::FreeThreeSplit);
    }

    #[test]
    fn line_pattern_treats_the_board_edge_as_blocked() {
        use LinePattern as P;
        let (own, opp) = line(&[0, 1, 2, 3], &[]);
        assert_eq!(line_pattern(own, opp, FULL, 4), P::Five);
        let (own, opp) = line(&[1, 2, 3], &[]);
        assert_eq!(line_pattern(own, opp, FULL, 0), P::BlockedFour);
        assert_eq!(line_pattern(0b1110, 0, 4, 0), P::Dead);
        let (own, opp) = line(&[11, 12, 13], &[]);
        assert_eq!(line_pattern(own, opp, FULL, 14), P::BlockedFour);
        assert_eq!(line_pattern(own, opp, FULL, 10), P::FreeFour);
    }

    #[test]
    fn line_patterns_agree_with_the_cell_functions_on_random_lines() {
        use LinePattern as P;
        let mut rng = fastrand::Rng::with_seed(99);
        for _ in 0..20_000 {
            let length = rng.u8(1..=FULL);
            let valid = (1u16 << length) - 1;
            let own = rng.u16(..) & valid;
            let opp = rng.u16(..) & valid & !own;
            let empty = valid & !(own | opp);
            for bit in 0..length {
                let cell = 1u16 << bit;
                if empty & cell == 0 {
                    continue;
                }
                let makes = |cells: fn(u16, u16) -> u16| cells(own, empty) & cell != 0;
                let pattern = line_pattern(own, opp, length, bit);
                let context = format!("own {own:#b} opp {opp:#b} length {length} bit {bit}");
                // The classifier reports the strongest shape the stone makes,
                // so the cell functions bound it from below. Rapfi reads at
                // most four cells each way, so a shape with stones beyond
                // that (`X_XXeX__X`) can be read as blocked when it is free;
                // the size of the shape is still right.
                assert_eq!(pattern == P::Five, makes(completing_cells), "{context}");
                if pattern != P::Five {
                    assert_eq!(
                        matches!(pattern, P::BlockedFour | P::FreeFour),
                        makes(four_making_cells),
                        "{context}"
                    );
                }
                if makes(open_four_making_cells) {
                    assert!(pattern >= P::BlockedFour, "{context}");
                }
                if makes(open_three_making_cells) {
                    assert!(pattern >= P::BlockedThree, "{context}");
                }
                if pattern.is_free_three() || pattern.is_blocked_three() {
                    assert!(makes(three_making_cells), "{context}");
                }
            }
        }
    }

    #[test]
    fn cell_patterns_pack_one_nibble_per_direction() {
        let patterns = CellPatterns::NONE
            .with(0, LinePattern::BlockedFour)
            .with(3, LinePattern::FreeThree)
            .with(1, LinePattern::BlockedOne);
        assert_eq!(patterns.get(0), LinePattern::BlockedFour);
        assert_eq!(patterns.get(1), LinePattern::BlockedOne);
        assert_eq!(patterns.get(2), LinePattern::Dead);
        assert_eq!(patterns.get(3), LinePattern::FreeThree);
        assert_eq!(
            patterns.with(0, LinePattern::Dead).get(0),
            LinePattern::Dead
        );
        assert_eq!(patterns.code().threat(), Threat::FourAndFreeThree);
    }

    #[test]
    fn pattern_codes_do_not_depend_on_direction_order() {
        use LinePattern as P;
        let a = CellPatterns::NONE
            .with(0, P::FreeTwo)
            .with(1, P::BlockedThree)
            .with(2, P::Dead)
            .with(3, P::FreeFour);
        let b = CellPatterns::NONE
            .with(0, P::FreeFour)
            .with(1, P::Dead)
            .with(2, P::FreeTwo)
            .with(3, P::BlockedThree);
        assert_eq!(a.code(), b.code());
        assert_ne!(a.code(), a.with(2, P::FreeOne).code());
    }

    #[test]
    fn pattern_codes_follow_rapfi_ordering() {
        // Rapfi enumerates the tuples highest nibble first, so the first
        // sixteen codes are a lone pattern on one line with the other three
        // dead, and its tables start with those sixteen cells.
        assert_eq!(CellPatterns::NONE.code(), PatternCode(0));
        assert_eq!(
            CellPatterns::NONE.with(0, LinePattern::Five).code(),
            PatternCode(15)
        );
        assert_eq!(
            CellPatterns::NONE.with(3, LinePattern::Five).code(),
            PatternCode(15)
        );
        assert_eq!(
            CellPatterns::NONE.with(2, LinePattern::FreeTwo).code(),
            PatternCode(5)
        );
        assert_eq!(CellPatterns::NONE.code().value(), Score::DRAW);
        assert_eq!(
            CellPatterns::NONE.with(0, LinePattern::Five).code().value(),
            Score::new(72)
        );
    }

    #[test]
    fn threats_combine_by_rapfi_priority() {
        use LinePattern as P;
        let combine = |a, b, c, d| Threat::combine([a, b, c, d]);
        assert_eq!(
            combine(P::Five, P::FreeFour, P::Dead, P::Dead),
            Threat::Five
        );
        assert_eq!(
            combine(P::FreeFour, P::Dead, P::Dead, P::Dead),
            Threat::UnstoppableFour
        );
        assert_eq!(
            combine(P::BlockedFour, P::BlockedFour, P::Dead, P::Dead),
            Threat::UnstoppableFour
        );
        assert_eq!(
            combine(P::BlockedFour, P::FreeThreeSplit, P::Dead, P::Dead),
            Threat::FourAndFreeThree
        );
        assert_eq!(
            combine(P::BlockedFour, P::BlockedThree, P::Dead, P::Dead),
            Threat::FourAndMore
        );
        assert_eq!(
            combine(P::BlockedFour, P::FreeTwoSplit, P::BlockedOne, P::Dead),
            Threat::FourAndMore
        );
        assert_eq!(
            combine(P::BlockedFour, P::BlockedTwo, P::FreeOne, P::Dead),
            Threat::Four
        );
        assert_eq!(
            combine(P::FreeThree, P::FreeThreeSplit, P::Dead, P::Dead),
            Threat::TwoFreeThrees
        );
        assert_eq!(
            combine(P::FreeThree, P::BlockedThreeSplit, P::Dead, P::Dead),
            Threat::FreeThreeAndMore
        );
        assert_eq!(
            combine(P::FreeThree, P::FreeTwo, P::Dead, P::Dead),
            Threat::FreeThreeAndMore
        );
        assert_eq!(
            combine(P::FreeThree, P::BlockedTwo, P::Dead, P::Dead),
            Threat::FreeThree
        );
        assert_eq!(
            combine(P::BlockedThree, P::BlockedThreeWideSplit, P::Dead, P::Dead),
            Threat::BlockedThreeAndMore
        );
        assert_eq!(
            combine(P::BlockedThree, P::FreeTwoWideSplit, P::Dead, P::Dead),
            Threat::BlockedThreeAndMore
        );
        assert_eq!(
            combine(P::FreeTwo, P::FreeTwoSplit, P::Dead, P::Dead),
            Threat::TwoFreeTwos
        );
        assert_eq!(
            combine(P::BlockedThree, P::BlockedTwo, P::Dead, P::Dead),
            Threat::Nothing
        );
        assert_eq!(
            combine(P::FreeTwo, P::BlockedOne, P::BlockedOne, P::Dead),
            Threat::Nothing
        );
    }

    #[test]
    fn tables_rank_stronger_cells_higher() {
        use LinePattern as P;
        let single = |p| CellPatterns::NONE.with(0, p).code();
        assert!(single(P::Five).value() > single(P::BlockedFour).value());
        assert!(single(P::FreeFour).value() >= single(P::BlockedFour).value());
        assert!(single(P::BlockedFour).value() > single(P::FreeThree).value());
        assert!(single(P::FreeThree).value() > single(P::BlockedThree).value());
        // A lone two is worth nothing in Rapfi's table; it only counts in
        // combination.
        assert_eq!(single(P::FreeTwo).value(), single(P::Dead).value());
        assert!(single(P::Five).score() > single(P::FreeThree).score());
        assert!(single(P::FreeTwo).score() > single(P::Dead).score());
        let double_three = CellPatterns::NONE
            .with(0, P::FreeThree)
            .with(1, P::FreeThree)
            .code();
        assert!(double_three.value() > single(P::FreeThree).value());
    }
}
