use std::ops;

use crate::position_id::PositionId;

/// Number of `u64` words needed to store `PositionId::COUNT` bits.
const WORD_COUNT: usize = PositionId::COUNT.div_ceil(64);

/// Number of valid bits in the last word (0 means the last word is fully used).
const TAIL_BITS: usize = PositionId::COUNT % 64;

/// Mask for valid bits in the last word. Prevents `Shl` from leaving dirty
/// high bits that could leak back into valid positions on a subsequent `Shr`.
const LAST_WORD_MASK: u64 = if TAIL_BITS == 0 {
    u64::MAX
} else {
    (1u64 << TAIL_BITS) - 1
};

/// Raw backing storage, sized to fit `PositionId::COUNT` bits.
type Storage = [u64; WORD_COUNT];

const WIDTH: usize = PositionId::WIDTH;

// ── Column masks for wraparound prevention ──────────────────────────────

/// Builds raw storage with all board positions set EXCEPT those in columns
/// whose bit is set in `exclude`. E.g., `exclude = 1 << 14` clears column 14.
const fn build_exclude_columns_mask(exclude: u16) -> Storage {
    let mut result = [0u64; PositionId::COUNT.div_ceil(64)];
    let mut index = 0;
    while index < PositionId::COUNT {
        let col = index % WIDTH;
        if (exclude >> col) & 1 == 0 {
            result[index / 64] |= 1 << (index % 64);
        }
        index += 1;
    }
    result
}

/// All positions set — no column masking (for vertical direction).
const ALL_COLS: BitBoard = BitBoard::from_raw(build_exclude_columns_mask(0));

// Masks for col_delta = +1 (horizontal, diagonal down-right):
// Shift by 1: exclude last column to prevent right-edge wraparound.
const NOT_LAST_COL: BitBoard = BitBoard::from_raw(build_exclude_columns_mask(1 << (WIDTH - 1)));
// Shift by 2: exclude last 2 columns.
const NOT_LAST_2_COLS: BitBoard = BitBoard::from_raw(build_exclude_columns_mask(
    (1 << (WIDTH - 1)) | (1 << (WIDTH - 2)),
));
// Shift by 3: exclude last 3 columns.
const NOT_LAST_3_COLS: BitBoard = BitBoard::from_raw(build_exclude_columns_mask(
    (1 << (WIDTH - 1)) | (1 << (WIDTH - 2)) | (1 << (WIDTH - 3)),
));
// Shift by 4: exclude last 4 columns.
const NOT_LAST_4_COLS: BitBoard = BitBoard::from_raw(build_exclude_columns_mask(
    (1 << (WIDTH - 1)) | (1 << (WIDTH - 2)) | (1 << (WIDTH - 3)) | (1 << (WIDTH - 4)),
));

// Masks for col_delta = -1 (diagonal down-left):
// Shift by 1: exclude first column to prevent left-edge wraparound.
const NOT_FIRST_COL: BitBoard = BitBoard::from_raw(build_exclude_columns_mask(1));
// Shift by 2: exclude first 2 columns.
const NOT_FIRST_2_COLS: BitBoard = BitBoard::from_raw(build_exclude_columns_mask(1 | (1 << 1)));
// Shift by 3: exclude first 3 columns.
const NOT_FIRST_3_COLS: BitBoard =
    BitBoard::from_raw(build_exclude_columns_mask(1 | (1 << 1) | (1 << 2)));
// Shift by 4: exclude first 4 columns.
const NOT_FIRST_4_COLS: BitBoard = BitBoard::from_raw(build_exclude_columns_mask(
    1 | (1 << 1) | (1 << 2) | (1 << 3),
));

// ── BitBoard ────────────────────────────────────────────────────────────

/// A fixed-size bitboard for a Gomoku board.
///
/// Backed by `[u64; N]` where N is derived from `PositionId::COUNT`.
/// Stack-allocated and `Copy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitBoard {
    words: Storage,
}

impl BitBoard {
    /// An empty bitboard with all bits unset.
    pub const EMPTY: Self = Self {
        words: [0; WORD_COUNT],
    };

    /// Creates a bitboard from raw storage.
    const fn from_raw(data: Storage) -> Self {
        Self { words: data }
    }

    /// Returns `true` if the bit at the given position is set.
    #[must_use]
    pub fn is_set(&self, position_id: PositionId) -> bool {
        let index = usize::from(position_id);
        (self.words[index / 64] >> (index % 64)) & 1 != 0
    }

    /// Sets the bit at the given position.
    pub fn set(&mut self, position_id: PositionId) {
        let index = usize::from(position_id);
        self.words[index / 64] |= 1 << (index % 64);
    }

    /// Clears the bit at the given position.
    pub fn clear(&mut self, position_id: PositionId) {
        let index = usize::from(position_id);
        self.words[index / 64] &= !(1 << (index % 64));
    }

    /// Returns `true` if any bit is set.
    #[must_use]
    pub fn any(self) -> bool {
        let mut idx = 0;
        while idx < WORD_COUNT {
            if self.words[idx] != 0 {
                return true;
            }
            idx += 1;
        }
        false
    }

    /// Returns the number of set bits.
    #[must_use]
    pub fn count_ones(self) -> usize {
        let mut count = 0;
        let mut idx = 0;
        while idx < WORD_COUNT {
            count += usize::try_from(self.words[idx].count_ones()).expect("u32 fits in usize");
            idx += 1;
        }
        count
    }

    /// Returns an iterator over all positions with set bits, yielding
    /// `PositionId`s in ascending index order. Uses `trailing_zeros` to
    /// skip unset bits efficiently.
    pub fn iter_set(self) -> BitBoardIter {
        BitBoardIter {
            words: self.words,
            word_idx: 0,
        }
    }

    /// Returns `true` if the bitboard contains five or more consecutive stones
    /// in any direction (horizontal, vertical, diagonal).
    #[must_use]
    pub fn has_five_in_a_row(self) -> bool {
        // Horizontal: stride 1, col delta +1
        check_direction(self, 1, NOT_LAST_COL, NOT_LAST_2_COLS, NOT_LAST_4_COLS)
        // Vertical: stride WIDTH, col delta 0 (no wraparound possible)
        || check_direction(self, WIDTH, ALL_COLS, ALL_COLS, ALL_COLS)
        // Diagonal down-right: stride WIDTH+1, col delta +1
        || check_direction(self, WIDTH + 1, NOT_LAST_COL, NOT_LAST_2_COLS, NOT_LAST_4_COLS)
        // Diagonal down-left: stride WIDTH-1, col delta -1
        || check_direction(self, WIDTH - 1, NOT_FIRST_COL, NOT_FIRST_2_COLS, NOT_FIRST_4_COLS)
    }
}

/// Iterator over set bit positions in a `BitBoard`.
pub struct BitBoardIter {
    words: Storage,
    word_idx: usize,
}

impl Iterator for BitBoardIter {
    type Item = PositionId;

    fn next(&mut self) -> Option<PositionId> {
        while self.word_idx < WORD_COUNT {
            let word = self.words[self.word_idx];
            if word != 0 {
                let bit = word.trailing_zeros() as usize;
                self.words[self.word_idx] = word & (word - 1); // clear lowest set bit
                return Some(PositionId::from(self.word_idx * 64 + bit));
            }
            self.word_idx += 1;
        }
        None
    }
}

/// Detects five or more consecutive stones in one direction using shift-AND.
///
/// For stride `s`, the masks prevent row-wraparound after shifts:
/// - `mask1` applied after shifting by `s`   (single step)
/// - `mask2` applied after shifting by `2s`  (double step)
/// - `mask4` applied after shifting by `4s`  (quadruple step)
fn check_direction(
    board: BitBoard,
    stride: usize,
    mask1: BitBoard,
    mask2: BitBoard,
    mask4: BitBoard,
) -> bool {
    let pairs = board & ((board >> stride) & mask1);
    let quads = pairs & ((pairs >> (2 * stride)) & mask2);
    let fives = quads & ((board >> (4 * stride)) & mask4);
    fives.any()
}

// ── Winning threat detection (shift-AND) ─────────────────────────────

/// Returns a bitboard of all positions where placing a stone would create
/// five-in-a-row. Computed as a batch operation: for each direction, finds
/// lines with exactly four stones and one gap, then maps each gap back to
/// its board position.
pub fn winning_threats(board: BitBoard) -> BitBoard {
    // Horizontal: stride 1, col delta +1
    threat_direction(
        board,
        1,
        NOT_LAST_COL,
        NOT_LAST_2_COLS,
        NOT_LAST_3_COLS,
        NOT_LAST_4_COLS,
    )
    // Vertical: stride WIDTH, col delta 0
    | threat_direction(board, WIDTH, ALL_COLS, ALL_COLS, ALL_COLS, ALL_COLS)
    // Diagonal down-right: stride WIDTH+1, col delta +1
    | threat_direction(
        board,
        WIDTH + 1,
        NOT_LAST_COL,
        NOT_LAST_2_COLS,
        NOT_LAST_3_COLS,
        NOT_LAST_4_COLS,
    )
    // Diagonal down-left: stride WIDTH-1, col delta -1
    | threat_direction(
        board,
        WIDTH - 1,
        NOT_FIRST_COL,
        NOT_FIRST_2_COLS,
        NOT_FIRST_3_COLS,
        NOT_FIRST_4_COLS,
    )
}

/// Finds gap positions that complete a five-in-a-row in one direction.
///
/// For each possible alignment of 5 consecutive positions along the given
/// stride, identifies where exactly 4 are occupied and maps the gap back
/// to its actual board position.
fn threat_direction(
    board: BitBoard,
    stride: usize,
    fwd1: BitBoard,
    fwd2: BitBoard,
    fwd3: BitBoard,
    fwd4: BitBoard,
) -> BitBoard {
    // Shifted views: s[k] has bit i set iff board has a stone at offset +k from i.
    let s0 = board;
    let s1 = (board >> stride) & fwd1;
    let s2 = (board >> (2 * stride)) & fwd2;
    let s3 = (board >> (3 * stride)) & fwd3;
    let s4 = (board >> (4 * stride)) & fwd4;

    // miss_at_k: four of five set, gap at offset k from the five's start.
    let miss0 = !s0 & s1 & s2 & s3 & s4;
    let miss1 = s0 & !s1 & s2 & s3 & s4;
    let miss2 = s0 & s1 & !s2 & s3 & s4;
    let miss3 = s0 & s1 & s2 & !s3 & s4;
    // miss4 loses the fwd4 column constraint from !s4, re-apply to prevent
    // the gap position from wrapping to the next row.
    let miss4 = s0 & s1 & s2 & s3 & !s4 & fwd4;

    // Shift each miss back to the actual gap position.
    miss0
        | (miss1 << stride)
        | (miss2 << (2 * stride))
        | (miss3 << (3 * stride))
        | (miss4 << (4 * stride))
}

// ── Pattern counting (shift-AND) ─────────────────────────────────────

/// Counts of consecutive stone patterns by length and openness.
#[derive(Debug, Default)]
pub struct PatternCounts {
    pub fives: usize,
    pub open_fours: usize,
    pub half_open_fours: usize,
    pub open_threes: usize,
    pub half_open_threes: usize,
    pub open_twos: usize,
    pub half_open_twos: usize,
}

/// Counts all consecutive stone patterns across all four directions.
///
/// For each direction, detects runs of exactly 2, 3, 4, and 5+ own stones
/// and classifies them by openness (both ends empty, one end empty, or closed).
pub fn count_all_patterns(own: BitBoard, opponent: BitBoard) -> PatternCounts {
    let mut counts = PatternCounts::default();

    // Horizontal: stride 1, col delta +1
    count_direction_patterns(
        own,
        opponent,
        1,
        NOT_LAST_COL,
        NOT_LAST_2_COLS,
        NOT_LAST_3_COLS,
        NOT_LAST_4_COLS,
        NOT_FIRST_COL,
        &mut counts,
    );
    // Vertical: stride WIDTH, col delta 0 (no wraparound possible)
    count_direction_patterns(
        own,
        opponent,
        WIDTH,
        ALL_COLS,
        ALL_COLS,
        ALL_COLS,
        ALL_COLS,
        ALL_COLS,
        &mut counts,
    );
    // Diagonal down-right: stride WIDTH+1, col delta +1
    count_direction_patterns(
        own,
        opponent,
        WIDTH + 1,
        NOT_LAST_COL,
        NOT_LAST_2_COLS,
        NOT_LAST_3_COLS,
        NOT_LAST_4_COLS,
        NOT_FIRST_COL,
        &mut counts,
    );
    // Diagonal down-left: stride WIDTH-1, col delta -1
    count_direction_patterns(
        own,
        opponent,
        WIDTH - 1,
        NOT_FIRST_COL,
        NOT_FIRST_2_COLS,
        NOT_FIRST_3_COLS,
        NOT_FIRST_4_COLS,
        NOT_LAST_COL,
        &mut counts,
    );

    counts
}

/// Counts patterns in a single direction using shift-AND.
///
/// Uses the same decomposition as `has_five_in_a_row`: consecutive runs via
/// shift-AND, then filters by exact length and openness (empty cells at ends).
#[allow(clippy::too_many_arguments)]
fn count_direction_patterns(
    own: BitBoard,
    opponent: BitBoard,
    stride: usize,
    fwd1: BitBoard,
    fwd2: BitBoard,
    fwd3: BitBoard,
    fwd4: BitBoard,
    bwd1: BitBoard,
    counts: &mut PatternCounts,
) {
    // Step 1: Consecutive runs (same decomposition as has_five_in_a_row)
    let consecutive2 = own & ((own >> stride) & fwd1);
    let consecutive3 = consecutive2 & ((own >> (2 * stride)) & fwd2);
    let consecutive4 = consecutive2 & ((consecutive2 >> (2 * stride)) & fwd2);
    let consecutive5 = consecutive4 & ((own >> (4 * stride)) & fwd4);

    // Step 2: Run starts (position before is not our stone)
    let not_preceded = !((own << stride) & bwd1);

    // Step 3: Exact run lengths
    let exact5 = consecutive5 & not_preceded;
    let exact4 = (consecutive4 & !consecutive5) & not_preceded;
    let exact3 = (consecutive3 & !consecutive4) & not_preceded;
    let exact2 = (consecutive2 & !consecutive3) & not_preceded;

    // Step 4: Openness (empty cell before/after the run)
    let empty = !(own | opponent);
    let open_before = (empty << stride) & bwd1;
    let open_after_4 = (empty >> (4 * stride)) & fwd4;
    let open_after_3 = (empty >> (3 * stride)) & fwd3;
    let open_after_2 = (empty >> (2 * stride)) & fwd2;

    // Step 5: Count patterns by length and openness
    counts.fives += exact5.count_ones();
    counts.open_fours += (exact4 & open_before & open_after_4).count_ones();
    counts.half_open_fours += (exact4 & (open_before ^ open_after_4)).count_ones();
    counts.open_threes += (exact3 & open_before & open_after_3).count_ones();
    counts.half_open_threes += (exact3 & (open_before ^ open_after_3)).count_ones();
    counts.open_twos += (exact2 & open_before & open_after_2).count_ones();
    counts.half_open_twos += (exact2 & (open_before ^ open_after_2)).count_ones();
}

impl Default for BitBoard {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl ops::Shr<usize> for BitBoard {
    type Output = Self;

    /// Shifts bits toward lower indices (arithmetic right-shift).
    fn shr(self, amount: usize) -> Self {
        if amount == 0 {
            return self;
        }
        if amount >= PositionId::COUNT {
            return Self::EMPTY;
        }
        let word_shift = amount / 64;
        let bit_shift = amount % 64;
        Self {
            words: std::array::from_fn(|idx| {
                let src = idx + word_shift;
                if src >= WORD_COUNT {
                    0
                } else if bit_shift == 0 {
                    self.words[src]
                } else {
                    let lo = self.words[src] >> bit_shift;
                    let hi = if src + 1 < WORD_COUNT {
                        self.words[src + 1] << (64 - bit_shift)
                    } else {
                        0
                    };
                    lo | hi
                }
            }),
        }
    }
}

impl ops::Shl<usize> for BitBoard {
    type Output = Self;

    /// Shifts bits toward higher indices (arithmetic left-shift).
    /// Cleans unused high bits in the last word to prevent leakage.
    fn shl(self, amount: usize) -> Self {
        if amount == 0 {
            return self;
        }
        if amount >= PositionId::COUNT {
            return Self::EMPTY;
        }
        let word_shift = amount / 64;
        let bit_shift = amount % 64;
        let mut result = Self {
            words: std::array::from_fn(|idx| {
                if idx < word_shift {
                    0
                } else {
                    let src = idx - word_shift;
                    if bit_shift == 0 {
                        self.words[src]
                    } else {
                        let hi = self.words[src] << bit_shift;
                        let lo = if src > 0 {
                            self.words[src - 1] >> (64 - bit_shift)
                        } else {
                            0
                        };
                        hi | lo
                    }
                }
            }),
        };
        result.words[WORD_COUNT - 1] &= LAST_WORD_MASK;
        result
    }
}

impl ops::BitAnd for BitBoard {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self {
        Self {
            words: std::array::from_fn(|idx| self.words[idx] & rhs.words[idx]),
        }
    }
}

impl ops::BitOr for BitBoard {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        Self {
            words: std::array::from_fn(|idx| self.words[idx] | rhs.words[idx]),
        }
    }
}

impl ops::BitXor for BitBoard {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self {
        Self {
            words: std::array::from_fn(|idx| self.words[idx] ^ rhs.words[idx]),
        }
    }
}

impl ops::Not for BitBoard {
    type Output = Self;

    fn not(self) -> Self {
        Self {
            words: std::array::from_fn(|idx| !self.words[idx]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;

    fn pos(row: usize, col: usize) -> PositionId {
        PositionId::from_position(Position::new(row, col))
    }

    fn bitboard_from_positions(positions: &[(usize, usize)]) -> BitBoard {
        let mut board = BitBoard::EMPTY;
        for &(row, col) in positions {
            board.set(pos(row, col));
        }
        board
    }

    // ── Basic operations ────────────────────────────────────────────

    #[test]
    fn empty_bitboard_has_no_bits_set() {
        let board = BitBoard::EMPTY;

        for position in PositionId::iter() {
            assert!(!board.is_set(position));
        }
    }

    #[test]
    fn set_and_query_single_bit() {
        let mut board = BitBoard::EMPTY;
        let center = PositionId::center();
        board.set(center);

        assert!(board.is_set(center));
    }

    #[test]
    fn unset_position_returns_false() {
        let board = BitBoard::EMPTY;

        assert!(!board.is_set(pos(0, 0)));
    }

    #[test]
    fn set_multiple_bits() {
        let mut board = BitBoard::EMPTY;
        board.set(pos(0, 0));
        board.set(pos(7, 7));
        board.set(pos(14, 14));

        assert!(board.is_set(pos(0, 0)));
        assert!(board.is_set(pos(7, 7)));
        assert!(board.is_set(pos(14, 14)));
        assert!(!board.is_set(pos(0, 1)));
    }

    #[test]
    fn default_is_empty() {
        let board = BitBoard::default();

        assert_eq!(board, BitBoard::EMPTY);
    }

    #[test]
    fn any_returns_false_for_empty() {
        assert!(!BitBoard::EMPTY.any());
    }

    #[test]
    fn any_returns_true_when_bit_set() {
        let mut board = BitBoard::EMPTY;
        board.set(pos(3, 7));

        assert!(board.any());
    }

    #[test]
    fn shr_shifts_bit_to_lower_index() {
        let mut board = BitBoard::EMPTY;
        board.set(pos(0, 5));

        let shifted = board >> 1;

        assert!(shifted.is_set(pos(0, 4)));
        assert!(!shifted.is_set(pos(0, 5)));
    }

    #[test]
    fn bitand_keeps_common_bits() {
        let mut left = BitBoard::EMPTY;
        left.set(pos(0, 0));
        left.set(pos(0, 1));

        let mut right = BitBoard::EMPTY;
        right.set(pos(0, 1));
        right.set(pos(0, 2));

        let result = left & right;

        assert!(!result.is_set(pos(0, 0)));
        assert!(result.is_set(pos(0, 1)));
        assert!(!result.is_set(pos(0, 2)));
    }

    // ── has_five_in_a_row tests ─────────────────────────────────────

    #[test]
    fn horizontal_five_detected() {
        let board = bitboard_from_positions(&[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]);

        assert!(board.has_five_in_a_row());
    }

    #[test]
    fn vertical_five_detected() {
        let board = bitboard_from_positions(&[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]);

        assert!(board.has_five_in_a_row());
    }

    #[test]
    fn diagonal_down_right_five_detected() {
        let board = bitboard_from_positions(&[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]);

        assert!(board.has_five_in_a_row());
    }

    #[test]
    fn diagonal_down_left_five_detected() {
        let board = bitboard_from_positions(&[(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)]);

        assert!(board.has_five_in_a_row());
    }

    #[test]
    fn four_in_a_row_not_detected() {
        let board = bitboard_from_positions(&[(0, 0), (0, 1), (0, 2), (0, 3)]);

        assert!(!board.has_five_in_a_row());
    }

    #[test]
    fn six_in_a_row_detected() {
        let board = bitboard_from_positions(&[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]);

        assert!(board.has_five_in_a_row());
    }

    #[test]
    fn horizontal_five_at_right_edge() {
        let board = bitboard_from_positions(&[(0, 10), (0, 11), (0, 12), (0, 13), (0, 14)]);

        assert!(board.has_five_in_a_row());
    }

    #[test]
    fn vertical_five_at_bottom_edge() {
        let board = bitboard_from_positions(&[(10, 0), (11, 0), (12, 0), (13, 0), (14, 0)]);

        assert!(board.has_five_in_a_row());
    }

    #[test]
    fn diagonal_five_at_bottom_right_corner() {
        let board = bitboard_from_positions(&[(10, 10), (11, 11), (12, 12), (13, 13), (14, 14)]);

        assert!(board.has_five_in_a_row());
    }

    #[test]
    fn no_false_horizontal_across_rows() {
        let board = bitboard_from_positions(&[(0, 13), (0, 14), (1, 0), (1, 1), (1, 2)]);

        assert!(!board.has_five_in_a_row());
    }

    #[test]
    fn no_false_diagonal_across_rows() {
        let board = bitboard_from_positions(&[(0, 12), (0, 13), (0, 14), (1, 0), (1, 1)]);

        assert!(!board.has_five_in_a_row());
    }

    #[test]
    fn empty_board_has_no_five() {
        assert!(!BitBoard::EMPTY.has_five_in_a_row());
    }

    // ── New operator tests ─────────────────────────────────────────

    #[test]
    fn shl_shifts_bit_to_higher_index() {
        let mut board = BitBoard::EMPTY;
        board.set(pos(0, 4));

        let shifted = board << 1;

        assert!(shifted.is_set(pos(0, 5)));
        assert!(!shifted.is_set(pos(0, 4)));
    }

    #[test]
    fn bitor_combines_bits() {
        let mut left = BitBoard::EMPTY;
        left.set(pos(0, 0));
        let mut right = BitBoard::EMPTY;
        right.set(pos(0, 1));

        let result = left | right;

        assert!(result.is_set(pos(0, 0)));
        assert!(result.is_set(pos(0, 1)));
    }

    #[test]
    fn not_inverts_bits() {
        let mut board = BitBoard::EMPTY;
        board.set(pos(0, 0));

        let inverted = !board;

        assert!(!inverted.is_set(pos(0, 0)));
        assert!(inverted.is_set(pos(0, 1)));
    }

    #[test]
    fn bitxor_keeps_exclusive_bits() {
        let mut left = BitBoard::EMPTY;
        left.set(pos(0, 0));
        left.set(pos(0, 1));
        let mut right = BitBoard::EMPTY;
        right.set(pos(0, 1));
        right.set(pos(0, 2));

        let result = left ^ right;

        assert!(result.is_set(pos(0, 0)));
        assert!(!result.is_set(pos(0, 1)));
        assert!(result.is_set(pos(0, 2)));
    }

    #[test]
    fn count_ones_counts_set_bits() {
        let board = bitboard_from_positions(&[(0, 0), (3, 7), (14, 14)]);

        assert_eq!(board.count_ones(), 3);
    }

    #[test]
    fn count_ones_empty_board_is_zero() {
        assert_eq!(BitBoard::EMPTY.count_ones(), 0);
    }

    // ── Pattern counting tests ─────────────────────────────────────

    #[test]
    fn empty_board_has_no_patterns() {
        let counts = count_all_patterns(BitBoard::EMPTY, BitBoard::EMPTY);

        assert_eq!(counts.fives, 0);
        assert_eq!(counts.open_fours, 0);
        assert_eq!(counts.half_open_fours, 0);
        assert_eq!(counts.open_threes, 0);
        assert_eq!(counts.half_open_threes, 0);
        assert_eq!(counts.open_twos, 0);
        assert_eq!(counts.half_open_twos, 0);
    }

    #[test]
    fn five_in_a_row_counted() {
        let own = bitboard_from_positions(&[(7, 5), (7, 6), (7, 7), (7, 8), (7, 9)]);
        let counts = count_all_patterns(own, BitBoard::EMPTY);

        assert_eq!(counts.fives, 1);
    }

    #[test]
    fn open_four_counted() {
        // Four in a row with empty cells on both sides
        let own = bitboard_from_positions(&[(7, 5), (7, 6), (7, 7), (7, 8)]);
        let counts = count_all_patterns(own, BitBoard::EMPTY);

        assert_eq!(counts.open_fours, 1);
    }

    #[test]
    fn half_open_four_with_opponent_block() {
        // Four in a row, opponent blocks one end
        let own = bitboard_from_positions(&[(7, 5), (7, 6), (7, 7), (7, 8)]);
        let opponent = bitboard_from_positions(&[(7, 4)]);
        let counts = count_all_patterns(own, opponent);

        assert_eq!(counts.open_fours, 0);
        assert_eq!(counts.half_open_fours, 1);
    }

    #[test]
    fn open_three_counted() {
        let own = bitboard_from_positions(&[(7, 5), (7, 6), (7, 7)]);
        let counts = count_all_patterns(own, BitBoard::EMPTY);

        assert_eq!(counts.open_threes, 1);
    }

    #[test]
    fn open_two_counted() {
        let own = bitboard_from_positions(&[(7, 5), (7, 6)]);
        let counts = count_all_patterns(own, BitBoard::EMPTY);

        assert_eq!(counts.open_twos, 1);
    }

    #[test]
    fn closed_pattern_not_counted() {
        // Three blocked on both sides
        let own = bitboard_from_positions(&[(7, 5), (7, 6), (7, 7)]);
        let opponent = bitboard_from_positions(&[(7, 4), (7, 8)]);
        let counts = count_all_patterns(own, opponent);

        assert_eq!(counts.open_threes, 0);
        assert_eq!(counts.half_open_threes, 0);
    }

    #[test]
    fn half_open_at_board_edge() {
        // Three against left edge — edge counts as blocked
        let own = bitboard_from_positions(&[(7, 0), (7, 1), (7, 2)]);
        let counts = count_all_patterns(own, BitBoard::EMPTY);

        assert_eq!(counts.open_threes, 0);
        assert_eq!(counts.half_open_threes, 1);
    }

    #[test]
    fn open_three_in_each_direction() {
        // Horizontal
        let horizontal = bitboard_from_positions(&[(7, 5), (7, 6), (7, 7)]);
        let horizontal_counts = count_all_patterns(horizontal, BitBoard::EMPTY);

        // Vertical
        let vertical = bitboard_from_positions(&[(5, 7), (6, 7), (7, 7)]);
        let vertical_counts = count_all_patterns(vertical, BitBoard::EMPTY);

        // Diagonal down-right
        let diag_right = bitboard_from_positions(&[(5, 5), (6, 6), (7, 7)]);
        let diag_right_counts = count_all_patterns(diag_right, BitBoard::EMPTY);

        // Diagonal down-left
        let diag_left = bitboard_from_positions(&[(5, 9), (6, 8), (7, 7)]);
        let diag_left_counts = count_all_patterns(diag_left, BitBoard::EMPTY);

        assert_eq!(horizontal_counts.open_threes, 1);
        assert_eq!(vertical_counts.open_threes, 1);
        assert_eq!(diag_right_counts.open_threes, 1);
        assert_eq!(diag_left_counts.open_threes, 1);
    }

    // ── iter_set tests ───────────────────────────────────────────────

    #[test]
    fn iter_set_yields_set_positions_in_order() {
        let board = bitboard_from_positions(&[(0, 0), (7, 7), (14, 14)]);
        let positions: Vec<PositionId> = board.iter_set().collect();

        assert_eq!(positions.len(), 3);
        assert_eq!(positions[0], pos(0, 0));
        assert_eq!(positions[1], pos(7, 7));
        assert_eq!(positions[2], pos(14, 14));
    }

    #[test]
    fn iter_set_empty_board_yields_nothing() {
        let positions: Vec<PositionId> = BitBoard::EMPTY.iter_set().collect();

        assert!(positions.is_empty());
    }

    // ── Winning threat tests ────────────────────────────────────────

    #[test]
    fn winning_threats_finds_horizontal_completion() {
        let board = bitboard_from_positions(&[(7, 5), (7, 6), (7, 7), (7, 8)]);
        let threats = winning_threats(board);

        assert!(threats.is_set(pos(7, 4)));
        assert!(threats.is_set(pos(7, 9)));
    }

    #[test]
    fn winning_threats_finds_vertical_completion() {
        let board = bitboard_from_positions(&[(3, 7), (4, 7), (5, 7), (6, 7)]);
        let threats = winning_threats(board);

        assert!(threats.is_set(pos(2, 7)));
        assert!(threats.is_set(pos(7, 7)));
    }

    #[test]
    fn winning_threats_finds_gap_in_middle() {
        // X X _ X X — gap at position 2
        let board = bitboard_from_positions(&[(7, 5), (7, 6), (7, 8), (7, 9)]);
        let threats = winning_threats(board);

        assert!(threats.is_set(pos(7, 7)));
    }

    #[test]
    fn winning_threats_empty_for_three_in_a_row() {
        let board = bitboard_from_positions(&[(7, 5), (7, 6), (7, 7)]);
        let threats = winning_threats(board);

        assert_eq!(threats.count_ones(), 0);
    }

    #[test]
    fn winning_threats_at_board_edges() {
        // Four at right edge
        let board = bitboard_from_positions(&[(0, 11), (0, 12), (0, 13), (0, 14)]);
        let threats = winning_threats(board);
        assert!(threats.is_set(pos(0, 10)));

        // Four at bottom edge
        let board = bitboard_from_positions(&[(11, 0), (12, 0), (13, 0), (14, 0)]);
        let threats = winning_threats(board);
        assert!(threats.is_set(pos(10, 0)));
    }

    #[test]
    fn winning_threats_agrees_with_would_win() {
        let mut rng = fastrand::Rng::with_seed(54321);
        for _ in 0..200 {
            let stone_count = rng.usize(4..20);
            let mut board = BitBoard::EMPTY;
            while board.count_ones() < stone_count {
                let index = rng.usize(0..PositionId::COUNT);
                board.set(PositionId::from(index));
            }

            let threats = winning_threats(board);

            for position in PositionId::iter() {
                let mut hypothetical = board;
                hypothetical.set(position);
                let would_win = hypothetical.has_five_in_a_row();

                if threats.is_set(position) && !board.is_set(position) {
                    assert!(
                        would_win,
                        "Threat at {position:?} but placing there doesn't create five"
                    );
                }
                if would_win && !board.has_five_in_a_row() && !board.is_set(position) {
                    assert!(
                        threats.is_set(position),
                        "Placing at {position:?} creates five but not flagged as threat"
                    );
                }
            }
        }
    }
}
