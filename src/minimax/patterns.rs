//! What one more stone at an empty cell would make: first on each of the
//! cell's four lines, then combined into one threat for the cell.
//!
//! This is the classical evaluation of engines such as Rapfi: a position is
//! worth the sum of its cells' threats. A line pattern is read from the eight
//! cells within reach along the line, so it is a table lookup, and the four
//! patterns of a cell combine through a second table.

use std::sync::LazyLock;

use super::lines::{
    REACH, completing_cells, four_making_cells, open_four_making_cells, open_three_making_cells,
    open_two_making_cells, three_making_cells, two_making_cells,
};
use super::score::Score;

/// What one more own stone at an empty cell makes on one line through it,
/// weakest first. Every name says what the stone would leave behind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum LinePattern {
    /// Nothing that can still become five.
    Nothing = 0,
    /// Two stones with room for five.
    Two,
    /// Two stones inside `_XX__`-like room: one step from an open three.
    OpenTwo,
    /// Three stones one step from a four.
    Three,
    /// `_XXX_` with room: one step from an open four.
    OpenThree,
    /// Four stones: one step from five, the opponent must block.
    Four,
    /// `_XXXX_`: the opponent cannot block both ends.
    OpenFour,
    /// Five in a row.
    Five,
}

impl LinePattern {
    const fn from_nibble(nibble: u16) -> Self {
        match nibble {
            0 => Self::Nothing,
            1 => Self::Two,
            2 => Self::OpenTwo,
            3 => Self::Three,
            4 => Self::OpenThree,
            5 => Self::Four,
            6 => Self::OpenFour,
            7 => Self::Five,
            _ => unreachable!(),
        }
    }

    const fn nibble(self) -> u16 {
        match self {
            Self::Nothing => 0,
            Self::Two => 1,
            Self::OpenTwo => 2,
            Self::Three => 3,
            Self::OpenThree => 4,
            Self::Four => 5,
            Self::OpenFour => 6,
            Self::Five => 7,
        }
    }
}

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

/// Puts an empty centre back into an eight-cell window.
fn insert_centre(window: Window) -> u16 {
    let window = u16::from(window);
    let low = window & u16::try_from(CENTRE - 1).expect("small");
    let high = window >> REACH;
    low | (high << (REACH + 1))
}

/// Line pattern of an empty centre cell by the own and blocked (opponent or
/// off-board) cells within reach: index `own << 8 | blocked`.
static LINE_PATTERNS: LazyLock<Box<LinePatternTable>> = LazyLock::new(|| {
    let mut table = vec![LinePattern::Nothing; 1 << 16];
    for (key, entry) in table.iter_mut().enumerate() {
        let own = insert_centre(u8::try_from(key >> 8).expect("high byte"));
        let blocked = insert_centre(u8::try_from(key & 0xFF).expect("low byte"));
        if own & blocked != 0 {
            continue; // a cell cannot be both; never looked up
        }
        let valid = (1u16 << (2 * REACH + 1)) - 1;
        let empty = !(own | blocked) & valid;
        let centre = u16::try_from(CENTRE).expect("small");
        let makes = |cells: fn(u16, u16) -> u16| cells(own, empty) & centre != 0;
        *entry = if makes(completing_cells) {
            LinePattern::Five
        } else if makes(open_four_making_cells) {
            LinePattern::OpenFour
        } else if makes(four_making_cells) {
            LinePattern::Four
        } else if makes(open_three_making_cells) {
            LinePattern::OpenThree
        } else if makes(three_making_cells) {
            LinePattern::Three
        } else if makes(open_two_making_cells) {
            LinePattern::OpenTwo
        } else if makes(two_making_cells) {
            LinePattern::Two
        } else {
            LinePattern::Nothing
        };
    }
    table
        .into_boxed_slice()
        .try_into()
        .expect("table has 1 << 16 entries")
});

/// Line pattern of an empty centre cell by the own and blocked (opponent or
/// off-board) cells within reach: index `own << 8 | blocked`.
pub type LinePatternTable = [LinePattern; 1 << 16];

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
    /// Nothing on any line: an occupied cell, or an empty one with no shape
    /// to complete.
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

    /// The four patterns combined into what a stone here would create.
    #[cfg(test)]
    #[must_use]
    pub fn threat(self) -> Threat {
        self.threat_from(threat_table())
    }

    /// The same, from a table reference held across many lookups.
    #[must_use]
    #[inline]
    pub fn threat_from(self, table: &ThreatTable) -> Threat {
        table[usize::from(self.0)]
    }
}

/// Threat of every `CellPatterns` value, indexed by its packed nibbles.
pub type ThreatTable = [Threat; 1 << 16];

/// The threat table, built on first use.
#[must_use]
pub fn threat_table() -> &'static ThreatTable {
    &THREATS
}

/// What one more stone at a cell would create, its four lines taken
/// together, weakest first. The classes and their order follow Rapfi's
/// classical evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum Threat {
    Nothing = 0,
    /// An open two on one line.
    OpenTwo,
    /// A three on one line.
    Three,
    /// Open twos on two lines.
    TwoOpenTwos,
    /// A three plus another three or an open two.
    ThreeAndMore,
    /// An open three on one line.
    OpenThree,
    /// An open three plus a three or an open two.
    OpenThreeAndMore,
    /// Open threes on two lines: the opponent can stop only one.
    TwoOpenThrees,
    /// A four on one line.
    Four,
    /// A four plus a three or an open two.
    FourAndMore,
    /// A four and an open three: block the four and the open four follows.
    FourAndOpenThree,
    /// An open four, or fours on two lines: five follows whatever is blocked.
    UnstoppableFour,
    /// Five in a row.
    Five,
}

impl Threat {
    pub const COUNT: usize = 13;

    /// What the patterns of a cell's four lines add up to.
    #[must_use]
    pub fn combine(patterns: [LinePattern; 4]) -> Self {
        let count = |wanted: LinePattern| patterns.iter().filter(|&&p| p == wanted).count();
        let fives = count(LinePattern::Five);
        let open_fours = count(LinePattern::OpenFour);
        let fours = count(LinePattern::Four);
        let open_threes = count(LinePattern::OpenThree);
        let threes = count(LinePattern::Three);
        let open_twos = count(LinePattern::OpenTwo);

        if fives > 0 {
            Self::Five
        } else if open_fours > 0 || fours >= 2 {
            Self::UnstoppableFour
        } else if fours == 1 && open_threes > 0 {
            Self::FourAndOpenThree
        } else if fours == 1 && (threes > 0 || open_twos > 0) {
            Self::FourAndMore
        } else if fours == 1 {
            Self::Four
        } else if open_threes >= 2 {
            Self::TwoOpenThrees
        } else if open_threes == 1 && (threes > 0 || open_twos > 0) {
            Self::OpenThreeAndMore
        } else if open_threes == 1 {
            Self::OpenThree
        } else if threes >= 2 || (threes == 1 && open_twos > 0) {
            Self::ThreeAndMore
        } else if open_twos >= 2 {
            Self::TwoOpenTwos
        } else if threes == 1 {
            Self::Three
        } else if open_twos == 1 {
            Self::OpenTwo
        } else {
            Self::Nothing
        }
    }

    /// What a cell offering this threat is worth to its colour.
    #[must_use]
    pub const fn value(self) -> Score {
        Score::new(match self {
            // The search always plays a five or an unstoppable four out, so
            // they never reach a quiet leaf; a value would only leak in
            // through the parent a leaf is averaged with and reward making
            // a four that is then blocked.
            Self::Nothing | Self::UnstoppableFour | Self::Five => 0,
            Self::OpenTwo => 20,
            Self::Three => 40,
            Self::TwoOpenTwos => 100,
            Self::ThreeAndMore => 150,
            Self::OpenThree => 300,
            Self::OpenThreeAndMore => 1_200,
            Self::TwoOpenThrees => 6_000,
            Self::Four => 1_500,
            Self::FourAndMore => 4_000,
            Self::FourAndOpenThree => 12_000,
        })
    }

    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Nothing => 0,
            Self::OpenTwo => 1,
            Self::Three => 2,
            Self::TwoOpenTwos => 3,
            Self::ThreeAndMore => 4,
            Self::OpenThree => 5,
            Self::OpenThreeAndMore => 6,
            Self::TwoOpenThrees => 7,
            Self::Four => 8,
            Self::FourAndMore => 9,
            Self::FourAndOpenThree => 10,
            Self::UnstoppableFour => 11,
            Self::Five => 12,
        }
    }

    /// The threat at `index`, the inverse of [`Threat::index`].
    #[cfg(test)]
    #[must_use]
    pub const fn from_index(index: usize) -> Self {
        match index {
            0 => Self::Nothing,
            1 => Self::OpenTwo,
            2 => Self::Three,
            3 => Self::TwoOpenTwos,
            4 => Self::ThreeAndMore,
            5 => Self::OpenThree,
            6 => Self::OpenThreeAndMore,
            7 => Self::TwoOpenThrees,
            8 => Self::Four,
            9 => Self::FourAndMore,
            10 => Self::FourAndOpenThree,
            11 => Self::UnstoppableFour,
            12 => Self::Five,
            _ => unreachable!(),
        }
    }
}

/// Threat of every `CellPatterns` value.
static THREATS: LazyLock<Box<ThreatTable>> = LazyLock::new(|| {
    let mut table = vec![Threat::Nothing; 1 << 16];
    for (key, entry) in table.iter_mut().enumerate() {
        let patterns = CellPatterns(u16::try_from(key).expect("16-bit key"));
        let nibble_valid = |direction: usize| (patterns.0 >> (4 * direction)) & 0xF < 8;
        if !(0..4).all(nibble_valid) {
            continue; // never looked up
        }
        *entry = Threat::combine([
            patterns.get(0),
            patterns.get(1),
            patterns.get(2),
            patterns.get(3),
        ]);
    }
    table
        .into_boxed_slice()
        .try_into()
        .expect("table has 1 << 16 entries")
});

#[cfg(test)]
mod tests {
    use super::*;

    const FULL: u8 = 15;

    fn line(own_bits: &[u8], opponent_bits: &[u8]) -> (u16, u16) {
        let mask = |bits: &[u8]| bits.iter().fold(0u16, |m, &b| m | (1 << b));
        (mask(own_bits), mask(opponent_bits))
    }

    #[test]
    fn line_pattern_reads_the_shapes_around_the_cell() {
        // e XXX with room on both sides: an open four.
        let (own, opp) = line(&[6, 7, 8], &[]);
        assert_eq!(line_pattern(own, opp, FULL, 5), LinePattern::OpenFour);
        // O e XXX _: a four, blocked on the left.
        let (own, opp) = line(&[6, 7, 8], &[4]);
        assert_eq!(line_pattern(own, opp, FULL, 5), LinePattern::Four);
        // XX e XX: a jump four completes five.
        let (own, opp) = line(&[3, 4, 6, 7], &[]);
        assert_eq!(line_pattern(own, opp, FULL, 5), LinePattern::Five);
        // _ e XX _ _: an open three.
        let (own, opp) = line(&[6, 7], &[]);
        assert_eq!(line_pattern(own, opp, FULL, 5), LinePattern::OpenThree);
        // O e XX _ _: a three that can only grow one way.
        let (own, opp) = line(&[6, 7], &[4]);
        assert_eq!(line_pattern(own, opp, FULL, 5), LinePattern::Three);
        // _ e X _ _: an open two.
        let (own, opp) = line(&[6], &[]);
        assert_eq!(line_pattern(own, opp, FULL, 5), LinePattern::OpenTwo);
        // e _ _ _ X: a two, too far apart for an open three in one move.
        let (own, opp) = line(&[9], &[]);
        assert_eq!(line_pattern(own, opp, FULL, 5), LinePattern::Two);
        // A lone stone is not yet a two, and a hemmed-in cell is nothing.
        assert_eq!(line_pattern(0, 0, FULL, 5), LinePattern::Nothing);
        let (own, opp) = line(&[6], &[3, 7]);
        assert_eq!(line_pattern(own, opp, FULL, 5), LinePattern::Nothing);
    }

    #[test]
    fn line_pattern_treats_the_board_edge_as_blocked() {
        // Bits 0..3 own, cell at 4: XXXX e with the edge on the left.
        let (own, opp) = line(&[0, 1, 2, 3], &[]);
        assert_eq!(line_pattern(own, opp, FULL, 4), LinePattern::Five);
        // Cell at 0 with XXX at 1..4 and an empty 4th: a four, not open.
        let (own, opp) = line(&[1, 2, 3], &[]);
        assert_eq!(line_pattern(own, opp, FULL, 0), LinePattern::Four);
        // A short diagonal of 4 cells can never hold five.
        assert_eq!(line_pattern(0b1110, 0, 4, 0), LinePattern::Nothing);
        // Near the far end of a full line the end reads as blocked too.
        let (own, opp) = line(&[11, 12, 13], &[]);
        assert_eq!(line_pattern(own, opp, FULL, 14), LinePattern::Four);
        assert_eq!(line_pattern(own, opp, FULL, 10), LinePattern::OpenFour);
    }

    #[test]
    fn line_pattern_matches_the_cell_functions_on_random_lines() {
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
                let expected = if makes(completing_cells) {
                    LinePattern::Five
                } else if makes(open_four_making_cells) {
                    LinePattern::OpenFour
                } else if makes(four_making_cells) {
                    LinePattern::Four
                } else if makes(open_three_making_cells) {
                    LinePattern::OpenThree
                } else if makes(three_making_cells) {
                    LinePattern::Three
                } else if makes(open_two_making_cells) {
                    LinePattern::OpenTwo
                } else if makes(two_making_cells) {
                    LinePattern::Two
                } else {
                    LinePattern::Nothing
                };
                assert_eq!(
                    line_pattern(own, opp, length, bit),
                    expected,
                    "own {own:#b} opp {opp:#b} length {length} bit {bit}"
                );
            }
        }
    }

    #[test]
    fn cell_patterns_pack_one_nibble_per_direction() {
        let patterns = CellPatterns::NONE
            .with(0, LinePattern::Four)
            .with(3, LinePattern::OpenThree)
            .with(1, LinePattern::Two);
        assert_eq!(patterns.get(0), LinePattern::Four);
        assert_eq!(patterns.get(1), LinePattern::Two);
        assert_eq!(patterns.get(2), LinePattern::Nothing);
        assert_eq!(patterns.get(3), LinePattern::OpenThree);
        assert_eq!(
            patterns.with(0, LinePattern::Nothing).get(0),
            LinePattern::Nothing
        );
        assert_eq!(patterns.threat(), Threat::FourAndOpenThree);
    }

    #[test]
    fn threats_combine_by_rapfi_priority() {
        use LinePattern as P;
        let combine = |a, b, c, d| Threat::combine([a, b, c, d]);
        assert_eq!(
            combine(P::Five, P::OpenFour, P::Nothing, P::Nothing),
            Threat::Five
        );
        assert_eq!(
            combine(P::OpenFour, P::Nothing, P::Nothing, P::Nothing),
            Threat::UnstoppableFour
        );
        assert_eq!(
            combine(P::Four, P::Four, P::Nothing, P::Nothing),
            Threat::UnstoppableFour
        );
        assert_eq!(
            combine(P::Four, P::OpenThree, P::Nothing, P::Nothing),
            Threat::FourAndOpenThree
        );
        assert_eq!(
            combine(P::Four, P::Three, P::Nothing, P::Nothing),
            Threat::FourAndMore
        );
        assert_eq!(
            combine(P::Four, P::OpenTwo, P::Two, P::Nothing),
            Threat::FourAndMore
        );
        assert_eq!(combine(P::Four, P::Two, P::Two, P::Two), Threat::Four);
        assert_eq!(
            combine(P::OpenThree, P::OpenThree, P::Nothing, P::Nothing),
            Threat::TwoOpenThrees
        );
        assert_eq!(
            combine(P::OpenThree, P::Three, P::Nothing, P::Nothing),
            Threat::OpenThreeAndMore
        );
        assert_eq!(
            combine(P::OpenThree, P::OpenTwo, P::Nothing, P::Nothing),
            Threat::OpenThreeAndMore
        );
        assert_eq!(
            combine(P::OpenThree, P::Two, P::Nothing, P::Nothing),
            Threat::OpenThree
        );
        assert_eq!(
            combine(P::Three, P::Three, P::Nothing, P::Nothing),
            Threat::ThreeAndMore
        );
        assert_eq!(
            combine(P::Three, P::OpenTwo, P::Nothing, P::Nothing),
            Threat::ThreeAndMore
        );
        assert_eq!(
            combine(P::OpenTwo, P::OpenTwo, P::Nothing, P::Nothing),
            Threat::TwoOpenTwos
        );
        assert_eq!(
            combine(P::Three, P::Two, P::Nothing, P::Nothing),
            Threat::Three
        );
        assert_eq!(combine(P::OpenTwo, P::Two, P::Two, P::Two), Threat::OpenTwo);
        assert_eq!(combine(P::Two, P::Two, P::Two, P::Two), Threat::Nothing);
    }

    #[test]
    fn threat_values_grow_with_the_threat() {
        let mut previous = Score::DRAW;
        for index in 0..Threat::COUNT {
            let threat = [
                Threat::Nothing,
                Threat::OpenTwo,
                Threat::Three,
                Threat::TwoOpenTwos,
                Threat::ThreeAndMore,
                Threat::OpenThree,
                Threat::OpenThreeAndMore,
                Threat::TwoOpenThrees,
                Threat::Four,
                Threat::FourAndMore,
                Threat::FourAndOpenThree,
                Threat::UnstoppableFour,
                Threat::Five,
            ][index];
            assert_eq!(threat.index(), index);
            assert_eq!(Threat::from_index(index), threat);
            // Rapfi ranks a four above two open threes, but a plain four is
            // not worth more than the double three it outranks; the two
            // classes the search always plays out are worth nothing.
            if !matches!(
                threat,
                Threat::Four | Threat::FourAndMore | Threat::UnstoppableFour | Threat::Five
            ) {
                assert!(threat.value() >= previous, "{threat:?}");
                previous = threat.value();
            }
        }
    }
}
