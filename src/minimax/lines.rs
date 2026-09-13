use crate::position_id::PositionId;
use crate::position_map::PositionArray;

use super::score::Score;

const W: usize = PositionId::WIDTH;
const _: () = assert!(W <= 15, "line masks use u16; increase to u32 for W > 15");

/// Number of scorable lines: W rows + W columns + (2W-1) DR diags + (2W-1) DL diags.
pub const NUM_LINES: usize = 6 * W - 2;

const COL_START: usize = W;
const DR_START: usize = 2 * W;
const DL_START: usize = 4 * W - 1;

/// Length (number of board cells) of each line.
pub const LINE_LENGTHS: [u8; NUM_LINES] = compute_line_lengths();

/// For each board position, its 4 (`line_id`, `bit_index`) pairs.
/// Order: row, column, DR diagonal, DL diagonal.
pub const POSITION_LINES: PositionArray<[(u8, u8); 4]> = compute_position_lines();

// `TryFrom` and `Ord::min` are not stable as const traits on stable Rust, so
// `as u8` is the only option in const fn context. All values are bounded by
// board geometry (W=15) and provably fit in u8.
#[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
const fn compute_line_lengths() -> [u8; NUM_LINES] {
    let mut lengths = [0u8; NUM_LINES];
    let mut i = 0;
    while i < W {
        lengths[i] = W as u8; // rows
        i += 1;
    }
    while i < 2 * W {
        lengths[i] = W as u8; // columns
        i += 1;
    }
    // DR diagonals: index d in 0..2W-1, k = d - (W-1), length = W - |k|
    while i < DL_START {
        let d = i - DR_START;
        let k_abs = d.abs_diff(W - 1);
        lengths[i] = (W - k_abs) as u8;
        i += 1;
    }
    // DL diagonals: index s in 0..2W-1, length = W - |s - (W-1)|
    while i < NUM_LINES {
        let s = i - DL_START;
        let s_abs = s.abs_diff(W - 1);
        lengths[i] = (W - s_abs) as u8;
        i += 1;
    }
    lengths
}

// `TryFrom` and `Ord::min` are not stable as const traits on stable Rust, so
// `as u8` is the only option in const fn context. All values are bounded by
// board geometry (W=15) and provably fit in u8.
#[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
const fn compute_position_lines() -> PositionArray<[(u8, u8); 4]> {
    let mut table = PositionArray::new([(0u8, 0u8); 4]);
    let mut index = 0;
    while index < PositionId::COUNT {
        let position = PositionId::from_index(index);
        let r = position.row();
        let c = position.col();
        let entry = table.get_mut(position);

        entry[0] = (r as u8, c as u8);
        entry[1] = ((COL_START + c) as u8, r as u8);

        let dr_bit = if r < c { r } else { c };
        entry[2] = ((DR_START + r + W - 1 - c) as u8, dr_bit as u8);

        let dl_bit_index = if r < W - 1 - c { r } else { W - 1 - c };
        entry[3] = ((DL_START + r + c) as u8, dl_bit_index as u8);

        index += 1;
    }
    table
}

fn popcount(bits: u16) -> usize {
    usize::try_from(bits.count_ones()).expect("u16 popcount fits in usize")
}

/// What one line contributes to the evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineScore {
    /// Score from Black's perspective (positive = Black advantage).
    pub score: Score,
    /// Whether the line holds a four for Black (index 0) or White (index 1),
    /// meaning one more stone of that colour completes five.
    pub has_four: [bool; 2],
    /// Whether the line holds an open three for Black (index 0) or White
    /// (index 1), meaning one more stone of that colour makes an open four.
    pub has_open_three: [bool; 2],
}

/// What one colour's stones on a line are worth.
struct SideScore {
    score: Score,
    has_four: bool,
    has_open_three: bool,
}

/// Scores a single line.
///
/// Uses u16 shift-AND (same algorithm as the bitboard pattern counter but on
/// single registers). Boundary handling is automatic: shifted-in zeros at
/// bit 0 and beyond the line length act as blockers.
pub fn score_line(black: u16, white: u16, length: u8) -> LineScore {
    if length < 2 {
        return LineScore {
            score: Score::DRAW,
            has_four: [false, false],
            has_open_three: [false, false],
        };
    }
    let valid = (1u16 << length) - 1;
    let empty = !(black | white) & valid;
    let black_side = score_side(black, valid, empty);
    let white_side = score_side(white, valid, empty);
    LineScore {
        score: black_side.score - white_side.score,
        has_four: [black_side.has_four, white_side.has_four],
        has_open_three: [black_side.has_open_three, white_side.has_open_three],
    }
}

/// A run of stones only matters if it can still become five: some 5-cell
/// window holding the whole run must have every other cell empty. A run that
/// fails this is dead and scores nothing, whatever its neighbours look like.
fn score_side(own: u16, valid: u16, empty: u16) -> SideScore {
    let c2 = own & (own >> 1);
    let c3 = c2 & (own >> 2);
    let c4 = c2 & (c2 >> 2);
    let c5 = c4 & (own >> 4);

    let not_preceded = !(own << 1) & valid;

    let exact5 = c5 & not_preceded;
    let exact4 = (c4 & !c5) & not_preceded;
    let exact3 = (c3 & !c4) & not_preceded;
    let exact2 = (c2 & !c3) & not_preceded;

    // Empty cells around a run that starts at bit i, named by offset from i.
    let empty_before_3 = empty << 3;
    let empty_before_2 = empty << 2;
    let empty_before_1 = empty << 1;
    let empty_after_2 = empty >> 2;
    let empty_after_3 = empty >> 3;
    let empty_after_4 = empty >> 4;

    // Jump fours: 4 own stones in a 5-cell window with one empty gap.
    // Filling the gap completes 5-in-a-row, so the opponent has exactly one
    // forced reply — semantically equivalent to a straight HALF_OPEN_FOUR.
    // The three patterns are mutually exclusive at each bit (different gap
    // offsets), so OR + single popcount is correct.
    let jump_four = (own & (empty >> 1) & (c3 >> 2)) // X_XXX
        | (c2 & (empty >> 2) & (c2 >> 3))            // XX_XX
        | (c3 & (empty >> 3) & (own >> 4)); // XXX_X

    // A four is live when an empty cell completes five.
    let open_four = exact4 & empty_before_1 & empty_after_4;
    let half_open_four = exact4 & (empty_before_1 ^ empty_after_4);

    // A three is live when one of the three 5-cell windows holding it has its
    // other two cells empty; `OXXX_O` fails and scores nothing. It is open when
    // one move makes an open four: both neighbours empty and room for a fifth
    // cell on at least one side. `O_XXX_O` has no room, so it is half-open.
    let live_three = exact3
        & ((empty_before_2 & empty_before_1)
            | (empty_before_1 & empty_after_3)
            | (empty_after_3 & empty_after_4));
    let open_three = exact3 & empty_before_1 & empty_after_3 & (empty_before_2 | empty_after_4);
    let half_open_three = live_three & !open_three;

    // Split threes: 3 own stones in a 4-cell window with one empty gap
    // (X_XX or XX_X). Filling the gap makes a straight four, so the window
    // counts as a three whose openness is that of the four it makes. A window
    // preceded or followed by an own stone is a jump four, counted above.
    let split_three = ((own & (empty >> 1) & (c2 >> 2)) | (c2 & (empty >> 2) & (own >> 3)))
        & not_preceded
        & !(own >> 4);
    let open_split_three = split_three & empty_before_1 & empty_after_4;
    let half_open_split_three = split_three & (empty_before_1 ^ empty_after_4);

    // A two is live when one of the four 5-cell windows holding it has its
    // other three cells empty; `O_XX_O` fails and scores nothing.
    let live_two = exact2
        & ((empty_before_3 & empty_before_2 & empty_before_1)
            | (empty_before_2 & empty_before_1 & empty_after_2)
            | (empty_before_1 & empty_after_2 & empty_after_3)
            | (empty_after_2 & empty_after_3 & empty_after_4));
    let open_two = live_two & empty_before_1 & empty_after_2;
    let half_open_two = live_two & (empty_before_1 ^ empty_after_2);

    let score = Score::WIN * popcount(exact5)
        + Score::OPEN_FOUR * popcount(open_four)
        + Score::HALF_OPEN_FOUR * popcount(half_open_four)
        + Score::HALF_OPEN_FOUR * popcount(jump_four)
        + Score::OPEN_THREE * popcount(open_three | open_split_three)
        + Score::HALF_OPEN_THREE * popcount(half_open_three | half_open_split_three)
        + Score::OPEN_TWO * popcount(open_two)
        + Score::HALF_OPEN_TWO * popcount(half_open_two);

    SideScore {
        score,
        has_four: (open_four | half_open_four | jump_four) != 0,
        has_open_three: (open_three | open_split_three) != 0,
    }
}

// Test positions are written as `W as u8` and small literal bit indices; all
// of them are bounded by the board width and fit in u8.
#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
mod tests {
    use super::*;
    use crate::position::Position;

    fn pos(row: usize, col: usize) -> PositionId {
        PositionId::from_position(Position::new(row, col))
    }

    #[test]
    fn line_lengths_rows_and_columns() {
        for i in 0..2 * W {
            assert_eq!(LINE_LENGTHS[i], W as u8, "Line {i} should be length {W}");
        }
    }

    #[test]
    fn line_lengths_diagonals_symmetric() {
        assert_eq!(LINE_LENGTHS[DR_START + W - 1], W as u8);
        assert_eq!(LINE_LENGTHS[DL_START + W - 1], W as u8);
        assert_eq!(LINE_LENGTHS[DR_START], 1);
        assert_eq!(LINE_LENGTHS[DR_START + 2 * (W - 1)], 1);
        assert_eq!(LINE_LENGTHS[DL_START], 1);
        assert_eq!(LINE_LENGTHS[DL_START + 2 * (W - 1)], 1);
    }

    #[test]
    fn position_lines_center() {
        let half = W / 2;
        let lines = POSITION_LINES.get(PositionId::center());
        assert_eq!(lines[0], (half as u8, half as u8));
        assert_eq!(lines[1], ((COL_START + half) as u8, half as u8));
        assert_eq!(lines[2], ((DR_START + W - 1) as u8, half as u8));
        assert_eq!(lines[3], ((DL_START + W - 1) as u8, half as u8));
    }

    #[test]
    fn position_lines_corners() {
        let tl = POSITION_LINES.get(pos(0, 0));
        assert_eq!(tl[2], ((DR_START + W - 1) as u8, 0));
        assert_eq!(tl[3], (DL_START as u8, 0));

        let tr = POSITION_LINES.get(pos(0, W - 1));
        assert_eq!(tr[2], (DR_START as u8, 0));
        assert_eq!(tr[3], ((DL_START + W - 1) as u8, 0));

        let bl = POSITION_LINES.get(pos(W - 1, 0));
        assert_eq!(bl[2], ((DR_START + 2 * (W - 1)) as u8, 0));
        assert_eq!(bl[3], ((DL_START + W - 1) as u8, (W - 1) as u8));
    }

    #[test]
    fn score_line_empty_is_zero() {
        for &len in &[1, 2, 5, 10, W as u8] {
            assert_eq!(score_line(0, 0, len).score, Score::DRAW);
        }
    }

    #[test]
    fn score_line_reports_which_colour_has_a_four() {
        // Black: XXXX_ at 3..=6; White: OO_OO at 9..=13 (jump four).
        let black = (1u16 << 3) | (1 << 4) | (1 << 5) | (1 << 6);
        let white = (1u16 << 9) | (1 << 10) | (1 << 12) | (1 << 13);
        assert_eq!(score_line(black, 0, W as u8).has_four, [true, false]);
        assert_eq!(score_line(0, white, W as u8).has_four, [false, true]);
        assert_eq!(score_line(black, white, W as u8).has_four, [true, true]);
    }

    #[test]
    fn score_line_dead_four_is_not_a_four() {
        // OXXXXO: no empty cell completes five.
        let black = (1u16 << 4) | (1 << 5) | (1 << 6) | (1 << 7);
        let white = (1u16 << 3) | (1 << 8);
        let line = score_line(black, white, W as u8);
        assert_eq!(line.has_four, [false, false]);
        assert_eq!(line.score, Score::DRAW);
    }

    #[test]
    fn score_line_open_three() {
        let own = (1u16 << 5) | (1u16 << 6) | (1u16 << 7);
        assert_eq!(score_one_side(own, 0, W as u8), Score::OPEN_THREE);
    }

    #[test]
    fn score_line_half_open_at_edge() {
        let own = 0b111u16;
        assert_eq!(score_one_side(own, 0, W as u8), Score::HALF_OPEN_THREE);
    }

    #[test]
    fn score_line_closed_three() {
        let own: u16 = (1 << 5) | (1 << 6) | (1 << 7);
        let opp: u16 = (1 << 4) | (1 << 8);
        assert_eq!(score_one_side(own, opp, W as u8), Score::DRAW);
    }

    fn score_one_side(own: u16, opp: u16, length: u8) -> Score {
        let valid = (1u16 << length) - 1;
        let empty = !(own | opp) & valid;
        score_side(own, valid, empty).score
    }

    #[test]
    fn score_line_three_without_room_for_an_open_four_is_half_open() {
        // O_XXX_O: both neighbours empty, but either four it makes is blocked.
        let own: u16 = (1 << 5) | (1 << 6) | (1 << 7);
        let opp: u16 = (1 << 3) | (1 << 9);
        assert_eq!(score_one_side(own, opp, W as u8), Score::HALF_OPEN_THREE);
    }

    #[test]
    fn score_line_three_with_room_on_one_side_is_open() {
        // O_XXX__: extending right makes _XXXX_.
        let own: u16 = (1 << 5) | (1 << 6) | (1 << 7);
        let opp: u16 = 1 << 3;
        assert_eq!(score_one_side(own, opp, W as u8), Score::OPEN_THREE);
    }

    #[test]
    fn score_line_split_three_with_both_ends_open_is_open() {
        // _X_XX_: filling the gap makes _XXXX_. The XX pair also counts as an
        // open two, matching how jump fours keep their inner twos.
        let own: u16 = (1 << 4) | (1 << 6) | (1 << 7);
        assert_eq!(
            score_one_side(own, 0, W as u8),
            Score::OPEN_THREE + Score::OPEN_TWO
        );
    }

    #[test]
    fn score_line_split_three_with_one_end_blocked_is_half_open() {
        // OX_XX_: filling the gap makes OXXXX_, a forcing four.
        let own: u16 = (1 << 4) | (1 << 6) | (1 << 7);
        let opp: u16 = 1 << 3;
        assert_eq!(
            score_one_side(own, opp, W as u8),
            Score::HALF_OPEN_THREE + Score::OPEN_TWO
        );
    }

    #[test]
    fn score_line_split_three_with_both_ends_blocked_is_not_a_three() {
        // OX_XXO: filling the gap makes a dead four, and the XX pair has no
        // 5-cell window left either, so the whole thing scores nothing.
        let own: u16 = (1 << 4) | (1 << 6) | (1 << 7);
        let opp: u16 = (1 << 3) | (1 << 8);
        assert_eq!(score_one_side(own, opp, W as u8), Score::DRAW);
    }

    #[test]
    fn score_line_three_that_can_only_make_a_dead_four_is_nothing() {
        // OXXX_O: the only extension gives OXXXXO, which cannot become five.
        let own: u16 = (1 << 4) | (1 << 5) | (1 << 6);
        let opp: u16 = (1 << 3) | (1 << 8);
        assert_eq!(score_one_side(own, opp, W as u8), Score::DRAW);
    }

    #[test]
    fn score_line_three_with_one_open_end_and_room_is_half_open() {
        // OXXX__: extending right gives OXXXX_, a live four.
        let own: u16 = (1 << 4) | (1 << 5) | (1 << 6);
        let opp: u16 = 1 << 3;
        assert_eq!(score_one_side(own, opp, W as u8), Score::HALF_OPEN_THREE);
    }

    #[test]
    fn score_line_two_without_a_five_cell_window_is_nothing() {
        // O_XX_O and OXX_O: no 5-cell window holds both stones with three
        // empty cells, so neither pair can ever become five.
        let own: u16 = (1 << 5) | (1 << 6);
        assert_eq!(
            score_one_side(own, (1 << 3) | (1 << 8), W as u8),
            Score::DRAW
        );
        assert_eq!(
            score_one_side(own, (1 << 4) | (1 << 8), W as u8),
            Score::DRAW
        );
    }

    #[test]
    fn score_line_two_with_room_on_one_side_is_half_open() {
        // OXX___: the window right of the pair is free.
        let own: u16 = (1 << 5) | (1 << 6);
        let opp: u16 = 1 << 4;
        assert_eq!(score_one_side(own, opp, W as u8), Score::HALF_OPEN_TWO);
    }

    #[test]
    fn score_line_reports_which_colour_has_an_open_three() {
        // Black: _XXX_ with room at 3..=5; White: _O_OO_ split three at 9..=12.
        let black = (1u16 << 3) | (1 << 4) | (1 << 5);
        let white = (1u16 << 9) | (1 << 11) | (1 << 12);
        assert_eq!(score_line(black, 0, W as u8).has_open_three, [true, false]);
        assert_eq!(score_line(0, white, W as u8).has_open_three, [false, true]);

        // O_XXX_O is only half-open, so it does not count.
        let hemmed_black = (1u16 << 5) | (1 << 6) | (1 << 7);
        let hemming_white = (1u16 << 3) | (1 << 9);
        assert_eq!(
            score_line(hemmed_black, hemming_white, W as u8).has_open_three,
            [false, false]
        );
    }

    #[test]
    fn score_line_jump_four_x_xxx() {
        // X_XXX: own at positions 3, 5, 6, 7 (gap at 4)
        // Also detects open three at 5,6,7 (open ends at 4 and 8).
        let own = (1 << 3) | (1 << 5) | (1 << 6) | (1 << 7);
        assert_eq!(
            score_one_side(own, 0, W as u8),
            Score::HALF_OPEN_FOUR + Score::OPEN_THREE
        );
    }

    #[test]
    fn score_line_jump_four_xx_xx() {
        // XX_XX: own at positions 3, 4, 6, 7 (gap at 5)
        // Also detects two open twos at (3,4) and (6,7).
        let own = (1 << 3) | (1 << 4) | (1 << 6) | (1 << 7);
        assert_eq!(
            score_one_side(own, 0, W as u8),
            Score::HALF_OPEN_FOUR + Score::OPEN_TWO * 2
        );
    }

    #[test]
    fn score_line_jump_four_xxx_x() {
        // XXX_X: own at positions 3, 4, 5, 7 (gap at 6)
        // Also detects open three at 3,4,5 (open ends at 2 and 6).
        let own = (1 << 3) | (1 << 4) | (1 << 5) | (1 << 7);
        assert_eq!(
            score_one_side(own, 0, W as u8),
            Score::HALF_OPEN_FOUR + Score::OPEN_THREE
        );
    }

    #[test]
    fn score_line_jump_four_blocked_by_opponent() {
        // XX_XX with opponent stone in the gap — no jump four.
        // Splits into two half-open twos (each blocked on the gap side).
        let own = (1 << 3) | (1 << 4) | (1 << 6) | (1 << 7);
        let opp = 1 << 5;
        assert_eq!(score_one_side(own, opp, W as u8), Score::HALF_OPEN_TWO * 2);
    }

    #[test]
    fn score_line_jump_four_on_short_line() {
        // XX_XX spanning the entire 5-cell line (positions 0,1,_,3,4).
        // Neither pair has a 5-cell window of its own, so only the jump four
        // scores.
        let own = (1 << 0) | (1 << 1) | (1 << 3) | (1 << 4);
        assert_eq!(score_one_side(own, 0, 5), Score::HALF_OPEN_FOUR);
    }
}
