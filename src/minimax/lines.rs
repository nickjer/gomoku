use crate::position_id::PositionId;
use crate::position_map::PositionArray;

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

/// For each line, the board position behind each bit. Entries past a
/// diagonal's length are never read.
pub const LINE_CELLS: [[PositionId; W]; NUM_LINES] = compute_line_cells();

/// How far along a line a stone can change what an empty cell offers: every
/// shape that matters fits in a six-cell window holding both cells.
pub const REACH: usize = 4;

/// The cells whose line patterns a stone at one position can change: up to
/// `REACH` cells each way along each of its four lines, with the direction
/// index (0..4, in `POSITION_LINES` order) of the line they share.
#[derive(Clone, Copy)]
pub struct Neighbourhood {
    cells: [(PositionId, u8); 8 * REACH],
    len: u8,
}

impl Neighbourhood {
    /// The neighbouring cells and the direction of the line shared with the
    /// centre.
    #[must_use]
    pub fn cells(&self) -> &[(PositionId, u8)] {
        &self.cells[..usize::from(self.len)]
    }
}

/// The neighbourhood of every board position.
pub static NEIGHBOURHOODS: PositionArray<Neighbourhood> = compute_neighbourhoods();

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

// Inverts `POSITION_LINES`. The u8 indices are bounded by the board geometry
// and only widen here; `usize::from` is not usable in a const fn.
#[allow(clippy::as_conversions)]
const fn compute_line_cells() -> [[PositionId; W]; NUM_LINES] {
    let mut table = [[PositionId::from_index(0); W]; NUM_LINES];
    let mut index = 0;
    while index < PositionId::COUNT {
        let position = PositionId::from_index(index);
        let lines = POSITION_LINES.get(position);
        let mut i = 0;
        while i < lines.len() {
            let (line_id, bit) = lines[i];
            table[line_id as usize][bit as usize] = position;
            i += 1;
        }
        index += 1;
    }
    table
}

// Same widening casts as above; the direction index is 0..4.
#[allow(clippy::cast_possible_truncation, clippy::as_conversions)]
const fn compute_neighbourhoods() -> PositionArray<Neighbourhood> {
    let mut table = PositionArray::new(Neighbourhood {
        cells: [(PositionId::from_index(0), 0u8); 8 * REACH],
        len: 0,
    });
    let mut index = 0;
    while index < PositionId::COUNT {
        let position = PositionId::from_index(index);
        let lines = POSITION_LINES.get(position);
        let entry = table.get_mut(position);
        let mut direction = 0;
        while direction < lines.len() {
            let (line_id, bit) = lines[direction];
            let length = LINE_LENGTHS[line_id as usize] as usize;
            let bit = bit as usize;
            let mut step = 1;
            while step <= REACH {
                if bit >= step {
                    entry.cells[entry.len as usize] =
                        (LINE_CELLS[line_id as usize][bit - step], direction as u8);
                    entry.len += 1;
                }
                if bit + step < length {
                    entry.cells[entry.len as usize] =
                        (LINE_CELLS[line_id as usize][bit + step], direction as u8);
                    entry.len += 1;
                }
                step += 1;
            }
            direction += 1;
        }
        index += 1;
    }
    table
}

/// The five cells of a window, as offsets from its first cell.
const FIVE_CELLS: u8 = 0b1_1111;

/// The two end cells and the four middle cells of a six-cell window, as
/// offsets from its first cell. An open four is empty ends around four own
/// stones; an open three is empty ends around three own stones and a gap.
const SIX_ENDS: u8 = 0b10_0001;
const SIX_MIDDLE: u8 = 0b01_1110;

/// Starts of the windows whose cells at the `own_at` offsets hold own stones
/// and whose cells at the `empty_at` offsets are empty. Offsets run 0..6 so a
/// six-cell window can be asked for as well.
fn window_starts(own: u16, empty: u16, own_at: u8, empty_at: u8) -> u16 {
    let mut starts = !0u16;
    for offset in 0..6u8 {
        if own_at & (1 << offset) != 0 {
            starts &= own >> offset;
        }
        if empty_at & (1 << offset) != 0 {
            starts &= empty >> offset;
        }
    }
    starts
}

/// The cells at the `offsets` of every window in `starts`.
fn cells_at(starts: u16, offsets: u8) -> u16 {
    let mut cells = 0;
    for offset in 0..6u8 {
        if offsets & (1 << offset) != 0 {
            cells |= starts << offset;
        }
    }
    cells
}

/// Empty cells where one more own stone makes five in a row.
#[must_use]
pub fn completing_cells(own: u16, empty: u16) -> u16 {
    let mut cells = 0;
    for gap in 0..5u8 {
        let gaps = 1u8 << gap;
        cells |= cells_at(window_starts(own, empty, FIVE_CELLS & !gaps, gaps), gaps);
    }
    cells
}

/// Empty cells where one more own stone makes a four: a five-cell window
/// left with four own stones and one empty cell, so the next stone wins.
#[must_use]
pub fn four_making_cells(own: u16, empty: u16) -> u16 {
    let mut cells = 0;
    for first in 0..5u8 {
        for second in first + 1..5u8 {
            let gaps = (1u8 << first) | (1u8 << second);
            cells |= cells_at(window_starts(own, empty, FIVE_CELLS & !gaps, gaps), gaps);
        }
    }
    cells
}

/// Empty cells where one more own stone makes a three: a five-cell window
/// left with three own stones and two empty cells, one step from a four.
#[must_use]
pub fn three_making_cells(own: u16, empty: u16) -> u16 {
    let mut cells = 0;
    for first in 0..5u8 {
        for second in first + 1..5u8 {
            let owns = (1u8 << first) | (1u8 << second);
            let gaps = FIVE_CELLS & !owns;
            cells |= cells_at(window_starts(own, empty, owns, gaps), gaps);
        }
    }
    cells
}

/// Empty cells where one more own stone makes a two: a five-cell window
/// left with two own stones and three empty cells.
#[must_use]
pub fn two_making_cells(own: u16, empty: u16) -> u16 {
    let mut cells = 0;
    for only in 0..5u8 {
        let owns = 1u8 << only;
        let gaps = FIVE_CELLS & !owns;
        cells |= cells_at(window_starts(own, empty, owns, gaps), gaps);
    }
    cells
}

/// Empty cells where one more own stone makes an open four, `_XXXX_`, which
/// the opponent cannot stop.
#[must_use]
pub fn open_four_making_cells(own: u16, empty: u16) -> u16 {
    let mut cells = 0;
    for gap in 1..5u8 {
        let gaps = 1u8 << gap;
        cells |= cells_at(
            window_starts(own, empty, SIX_MIDDLE & !gaps, SIX_ENDS | gaps),
            gaps,
        );
    }
    cells
}

/// Empty cells where one more own stone makes an open three: the middle of
/// a six-cell window with empty ends then holds three own stones and a gap.
#[must_use]
pub fn open_three_making_cells(own: u16, empty: u16) -> u16 {
    let mut cells = 0;
    for first in 1..5u8 {
        for second in first + 1..5u8 {
            let gaps = (1u8 << first) | (1u8 << second);
            cells |= cells_at(
                window_starts(own, empty, SIX_MIDDLE & !gaps, SIX_ENDS | gaps),
                gaps,
            );
        }
    }
    cells
}

/// Empty cells where one more own stone makes an open two: the middle of a
/// six-cell window with empty ends then holds two own stones and two gaps,
/// one step from an open three.
#[must_use]
pub fn open_two_making_cells(own: u16, empty: u16) -> u16 {
    let mut cells = 0;
    for only in 1..5u8 {
        let owns = 1u8 << only;
        let gaps = SIX_MIDDLE & !owns;
        cells |= cells_at(window_starts(own, empty, owns, SIX_ENDS | gaps), gaps);
    }
    cells
}

/// Empty cells where an opponent stone stops one more own stone at `cell`
/// from making a four on this line: the cell itself, or a cell of the
/// five-cell window that four would fill.
#[must_use]
pub fn four_breaking_cells(own: u16, empty: u16, cell: u16) -> u16 {
    let mut cells = 0;
    let mut rest = empty;
    while rest != 0 {
        let candidate = rest.isolate_lowest_one();
        rest ^= candidate;
        if four_making_cells(own, empty & !candidate) & cell == 0 {
            cells |= candidate;
        }
    }
    cells
}

/// Empty cells where an opponent stone leaves `own` unable to make an open
/// four anywhere on this line: the replies that answer an open three.
#[must_use]
pub fn defusing_cells(own: u16, empty: u16) -> u16 {
    let mut cells = 0;
    let mut rest = empty;
    while rest != 0 {
        let cell = rest.isolate_lowest_one();
        rest ^= cell;
        if open_four_making_cells(own, empty & !cell) == 0 {
            cells |= cell;
        }
    }
    cells
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
        for &length in &LINE_LENGTHS[..2 * W] {
            assert_eq!(length, W as u8);
        }
    }

    #[test]
    fn line_lengths_diagonals_symmetric() {
        // Main diagonals (k=0) have length W; corners have length 1.
        assert_eq!(LINE_LENGTHS[DR_START + W - 1], W as u8);
        assert_eq!(LINE_LENGTHS[DR_START], 1);
        assert_eq!(LINE_LENGTHS[DR_START + 2 * W - 2], 1);
        assert_eq!(LINE_LENGTHS[DL_START + W - 1], W as u8);
        assert_eq!(LINE_LENGTHS[DL_START], 1);
        assert_eq!(LINE_LENGTHS[DL_START + 2 * W - 2], 1);
    }

    #[test]
    fn position_lines_center() {
        let center = PositionId::center();
        let lines = POSITION_LINES.get(center);
        let mid = W / 2;
        assert_eq!(lines[0], (mid as u8, mid as u8));
        assert_eq!(lines[1], ((COL_START + mid) as u8, mid as u8));
        assert_eq!(lines[2], ((DR_START + W - 1) as u8, mid as u8));
        assert_eq!(lines[3], ((DL_START + W - 1) as u8, mid as u8));
    }

    #[test]
    fn position_lines_corners() {
        let top_left = pos(0, 0);
        let lines = POSITION_LINES.get(top_left);
        assert_eq!(lines[0], (0, 0));
        assert_eq!(lines[1], (COL_START as u8, 0));
        assert_eq!(lines[2], ((DR_START + W - 1) as u8, 0));
        assert_eq!(lines[3], (DL_START as u8, 0));

        let bottom_right = pos(W - 1, W - 1);
        let lines = POSITION_LINES.get(bottom_right);
        assert_eq!(lines[0], ((W - 1) as u8, (W - 1) as u8));
        assert_eq!(lines[1], ((COL_START + W - 1) as u8, (W - 1) as u8));
        assert_eq!(lines[2], ((DR_START + W - 1) as u8, (W - 1) as u8));
        assert_eq!(lines[3], ((DL_START + 2 * W - 2) as u8, 0));
    }

    #[test]
    fn line_cells_invert_position_lines() {
        for position in PositionId::iter() {
            for &(line_id, bit) in POSITION_LINES.get(position) {
                assert_eq!(
                    LINE_CELLS[usize::from(line_id)][usize::from(bit)],
                    position,
                    "line {line_id} bit {bit}"
                );
            }
        }
    }

    #[test]
    fn neighbourhood_holds_the_cells_within_reach_on_each_line() {
        for position in PositionId::iter() {
            let mut expected = Vec::new();
            for (direction, &(line_id, bit)) in POSITION_LINES.get(position).iter().enumerate() {
                let length = usize::from(LINE_LENGTHS[usize::from(line_id)]);
                for (other, &cell) in LINE_CELLS[usize::from(line_id)]
                    .iter()
                    .enumerate()
                    .take(length)
                {
                    if (1..=REACH).contains(&other.abs_diff(usize::from(bit))) {
                        expected.push((cell, direction as u8));
                    }
                }
            }
            let mut actual = NEIGHBOURHOODS.get(position).cells().to_vec();
            actual.sort_by_key(|(cell, direction)| (cell.to_index(), *direction));
            expected.sort_by_key(|(cell, direction)| (cell.to_index(), *direction));
            assert_eq!(actual, expected, "neighbourhood of {position:?}");
        }
        assert_eq!(
            NEIGHBOURHOODS.get(PositionId::center()).cells().len(),
            8 * REACH
        );
        assert_eq!(NEIGHBOURHOODS.get(pos(0, 0)).cells().len(), 3 * REACH);
    }

    // ── Cell functions against a brute-force reading of the line ────────────

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Cell {
        Own,
        Other,
        Empty,
    }

    fn masks(cells: &[Cell]) -> (u16, u16) {
        let mut own = 0;
        let mut empty = 0;
        for (i, cell) in cells.iter().enumerate() {
            match cell {
                Cell::Own => own |= 1 << i,
                Cell::Empty => empty |= 1 << i,
                Cell::Other => {}
            }
        }
        (own, empty)
    }

    /// Every colouring of a line of `length` cells.
    fn all_lines(length: usize) -> impl Iterator<Item = Vec<Cell>> {
        (0..3usize.pow(length as u32)).map(move |mut code| {
            (0..length)
                .map(|_| {
                    let cell = [Cell::Own, Cell::Other, Cell::Empty][code % 3];
                    code /= 3;
                    cell
                })
                .collect()
        })
    }

    fn random_lines(length: usize, count: usize) -> Vec<Vec<Cell>> {
        let mut rng = fastrand::Rng::with_seed(2024);
        (0..count)
            .map(|_| {
                (0..length)
                    .map(|_| [Cell::Own, Cell::Other, Cell::Empty][rng.usize(0..3)])
                    .collect()
            })
            .collect()
    }

    fn with_own_at(cells: &[Cell], index: usize) -> Vec<Cell> {
        let mut after = cells.to_vec();
        after[index] = Cell::Own;
        after
    }

    fn with_other_at(cells: &[Cell], index: usize) -> Vec<Cell> {
        let mut after = cells.to_vec();
        after[index] = Cell::Other;
        after
    }

    fn owns(w: &[Cell]) -> usize {
        w.iter().filter(|&&c| c == Cell::Own).count()
    }

    fn empties(w: &[Cell]) -> usize {
        w.iter().filter(|&&c| c == Cell::Empty).count()
    }

    /// A five-cell window with exactly `own` own stones and the rest empty.
    fn has_five_window(cells: &[Cell], own: usize) -> bool {
        cells
            .windows(5)
            .any(|w| owns(w) == own && empties(w) == 5 - own)
    }

    /// A six-cell window with empty ends whose middle has exactly `own` own
    /// stones and the rest empty.
    fn has_six_window(cells: &[Cell], own: usize) -> bool {
        cells.windows(6).any(|w| {
            w[0] == Cell::Empty
                && w[5] == Cell::Empty
                && owns(&w[1..5]) == own
                && empties(&w[1..5]) == 4 - own
        })
    }

    fn has_five(cells: &[Cell]) -> bool {
        cells.windows(5).any(|w| owns(w) == 5)
    }

    fn has_four(cells: &[Cell]) -> bool {
        has_five_window(cells, 4)
    }

    fn has_three(cells: &[Cell]) -> bool {
        has_five_window(cells, 3)
    }

    fn has_two(cells: &[Cell]) -> bool {
        has_five_window(cells, 2)
    }

    fn has_open_four(cells: &[Cell]) -> bool {
        has_six_window(cells, 4)
    }

    fn has_open_three(cells: &[Cell]) -> bool {
        has_six_window(cells, 3)
    }

    fn has_open_two(cells: &[Cell]) -> bool {
        has_six_window(cells, 2)
    }

    /// The bits of the empty cells `index` for which `after_own(index)` holds.
    fn cells_where(cells: &[Cell], after_own: impl Fn(&[Cell]) -> bool) -> u16 {
        (0..cells.len())
            .filter(|&i| cells[i] == Cell::Empty && after_own(&with_own_at(cells, i)))
            .fold(0, |bits, i| bits | (1 << i))
    }

    /// Checks one cell function against the shape it claims to make. A line
    /// that already holds the shape is skipped: there any stone "makes" it.
    fn check_makes(
        name: &str,
        cells: &[Cell],
        cell_function: fn(u16, u16) -> u16,
        makes: fn(&[Cell]) -> bool,
    ) {
        if makes(cells) {
            return;
        }
        let (own, empty) = masks(cells);
        assert_eq!(
            cell_function(own, empty),
            cells_where(cells, makes),
            "{name} of {own:#b}/{empty:#b}"
        );
    }

    fn check_cells_against_oracle(cells: &[Cell]) {
        // A line with five is a finished game; the search never asks about it.
        if has_five(cells) {
            return;
        }
        check_makes("completing", cells, completing_cells, has_five);
        check_makes("four-making", cells, four_making_cells, has_four);
        check_makes("three-making", cells, three_making_cells, has_three);
        check_makes("two-making", cells, two_making_cells, has_two);
        check_makes(
            "open-four-making",
            cells,
            open_four_making_cells,
            has_open_four,
        );
        check_makes(
            "open-three-making",
            cells,
            open_three_making_cells,
            has_open_three,
        );
        check_makes(
            "open-two-making",
            cells,
            open_two_making_cells,
            has_open_two,
        );

        if !has_open_four(cells) {
            let (own, empty) = masks(cells);
            let expected_defusers = (0..cells.len())
                .filter(|&i| {
                    cells[i] == Cell::Empty
                        && cells_where(&with_other_at(cells, i), has_open_four) == 0
                })
                .fold(0, |bits, i| bits | (1 << i));
            assert_eq!(
                defusing_cells(own, empty),
                expected_defusers,
                "defusing cells of {own:#b}/{empty:#b}"
            );
        }
    }

    #[test]
    fn cell_functions_match_brute_force_on_every_short_line() {
        for length in 1..=9 {
            for cells in all_lines(length) {
                check_cells_against_oracle(&cells);
            }
        }
    }

    #[test]
    fn cell_functions_match_brute_force_on_random_full_lines() {
        for cells in random_lines(W, 20_000) {
            check_cells_against_oracle(&cells);
        }
    }

    #[test]
    fn four_breaking_cells_are_the_cell_and_the_window_it_would_fill() {
        // O X X X e _ : the four at `e` (bit 8) is broken by taking e or the
        // cell that would complete it (bit 9); nothing else on the line helps.
        let own: u16 = (1 << 5) | (1 << 6) | (1 << 7);
        let other: u16 = 1 << 4;
        let empty = !(own | other) & ((1 << W) - 1);
        assert_eq!(four_breaking_cells(own, empty, 1 << 8), (1 << 8) | (1 << 9));

        // X _ X X e: the gap (bit 5) breaks the jump four too.
        let own: u16 = (1 << 4) | (1 << 6) | (1 << 7);
        let other: u16 = (1 << 3) | (1 << 9);
        let empty = !(own | other) & ((1 << W) - 1);
        assert_eq!(four_breaking_cells(own, empty, 1 << 8), (1 << 5) | (1 << 8));
    }

    #[test]
    fn defusing_cells_of_a_one_sided_open_three_include_the_far_cell() {
        // O_XXX__: the open four can only grow rightwards, so the far right
        // cell stops it just like the two neighbours do.
        let own: u16 = (1 << 5) | (1 << 6) | (1 << 7);
        let other: u16 = 1 << 3;
        let empty = !(own | other) & ((1 << W) - 1);
        assert_eq!(defusing_cells(own, empty), (1 << 4) | (1 << 8) | (1 << 9));

        // __XXX__: only the neighbours; a stone two away still leaves the
        // other side room for _XXXX_.
        let empty = !own & ((1 << W) - 1);
        assert_eq!(defusing_cells(own, empty), (1 << 4) | (1 << 8));
    }
}
