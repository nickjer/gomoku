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

/// Scores a single line from Black's perspective (positive = Black advantage).
///
/// Uses u16 shift-AND (same algorithm as the bitboard pattern counter but on
/// single registers). Boundary handling is automatic: shifted-in zeros at
/// bit 0 and beyond the line length act as blockers.
pub fn score_line(black: u16, white: u16, length: u8) -> Score {
    if length < 2 {
        return Score::DRAW;
    }
    let valid = (1u16 << length) - 1;
    let empty = !(black | white) & valid;
    score_side(black, valid, empty) - score_side(white, valid, empty)
}

fn score_side(own: u16, valid: u16, empty: u16) -> Score {
    let c2 = own & (own >> 1);
    let c3 = c2 & (own >> 2);
    let c4 = c2 & (c2 >> 2);
    let c5 = c4 & (own >> 4);

    let not_preceded = !(own << 1) & valid;

    let exact5 = c5 & not_preceded;
    let exact4 = (c4 & !c5) & not_preceded;
    let exact3 = (c3 & !c4) & not_preceded;
    let exact2 = (c2 & !c3) & not_preceded;

    let open_before = empty << 1;
    let open_after_2 = empty >> 2;
    let open_after_3 = empty >> 3;
    let open_after_4 = empty >> 4;

    // Jump fours: 4 own stones in a 5-cell window with one empty gap.
    // Filling the gap completes 5-in-a-row, so the opponent has exactly one
    // forced reply — semantically equivalent to a straight HALF_OPEN_FOUR.
    // The three patterns are mutually exclusive at each bit (different gap
    // offsets), so OR + single popcount is correct.
    let jump_four = (own & (empty >> 1) & (c3 >> 2)) // X_XXX
        | (c2 & (empty >> 2) & (c2 >> 3))            // XX_XX
        | (c3 & (empty >> 3) & (own >> 4)); // XXX_X

    Score::WIN * popcount(exact5)
        + Score::OPEN_FOUR * popcount(exact4 & open_before & open_after_4)
        + Score::HALF_OPEN_FOUR * popcount(exact4 & (open_before ^ open_after_4))
        + Score::HALF_OPEN_FOUR * popcount(jump_four)
        + Score::OPEN_THREE * popcount(exact3 & open_before & open_after_3)
        + Score::HALF_OPEN_THREE * popcount(exact3 & (open_before ^ open_after_3))
        + Score::OPEN_TWO * popcount(exact2 & open_before & open_after_2)
        + Score::HALF_OPEN_TWO * popcount(exact2 & (open_before ^ open_after_2))
}

#[cfg(test)]
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
            assert_eq!(score_line(0, 0, len), Score::DRAW);
        }
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
        score_side(own, valid, empty)
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
        // Both twos are half-open (blocked by line boundaries).
        let own = (1 << 0) | (1 << 1) | (1 << 3) | (1 << 4);
        assert_eq!(
            score_one_side(own, 0, 5),
            Score::HALF_OPEN_FOUR + Score::HALF_OPEN_TWO * 2
        );
    }
}
