use crate::offset::Offset;
use crate::position::Position;

/// A board position represented as a single index (0-224).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PositionId {
    index: usize,
}

impl PositionId {
    /// Board width (15).
    pub const WIDTH: usize = 15;

    /// Total number of board positions (15 × 15 = 225).
    pub const COUNT: usize = Self::WIDTH.checked_mul(Self::WIDTH).unwrap();

    const fn new(index: usize) -> Self {
        Self { index }
    }

    #[must_use]
    pub const fn from_position(position: Position) -> Self {
        Self::new(position.row() * Self::WIDTH + position.col())
    }

    #[must_use]
    pub const fn row(self) -> usize {
        self.index / Self::WIDTH
    }

    #[must_use]
    pub const fn col(self) -> usize {
        self.index % Self::WIDTH
    }

    #[must_use]
    pub const fn position(self) -> Position {
        Position::new(self.row(), self.col())
    }

    #[must_use]
    pub const fn center() -> Self {
        Self::new(Self::COUNT / 2)
    }

    /// Returns a new position offset by the given delta, or `None` if out of bounds.
    ///
    /// # Panics
    ///
    /// Panics if the resulting index overflows.
    #[must_use]
    pub fn from_offset(self, offset: Offset) -> Option<Self> {
        let new_row = self.row().checked_add_signed(offset.row_delta())?;
        let new_col = self.col().checked_add_signed(offset.col_delta())?;

        if new_row >= Self::WIDTH || new_col >= Self::WIDTH {
            return None;
        }

        let new_index = new_row
            .checked_mul(Self::WIDTH)
            .and_then(|i| i.checked_add(new_col))
            .expect("position index overflow");
        Some(Self::new(new_index))
    }

    #[must_use]
    pub fn neighbor_count(self, offsets: &[Offset]) -> usize {
        offsets
            .iter()
            .filter(|&&offset| self.from_offset(offset).is_some())
            .count()
    }

    pub fn iter() -> impl Iterator<Item = Self> {
        (0..Self::COUNT).map(Self::new)
    }

    pub fn map<T, F>(f: F) -> Vec<T>
    where
        F: FnMut(Self) -> T,
    {
        Self::iter().map(f).collect()
    }

    /// Returns an iterator over all rows of the board.
    ///
    /// # Panics
    ///
    /// Panics if the position index overflows.
    pub fn rows() -> impl Iterator<Item = Vec<Self>> {
        (0..Self::WIDTH).map(|row| {
            (0..Self::WIDTH)
                .map(move |col| {
                    let index = row
                        .checked_mul(Self::WIDTH)
                        .and_then(|i| i.checked_add(col))
                        .expect("position index overflow");
                    Self::new(index)
                })
                .collect()
        })
    }

    // ---- Grid transformation primitives ----

    /// Precomputed base for vertical flip: COUNT - WIDTH.
    const FLIP_V_BASE: usize = Self::COUNT - Self::WIDTH;

    /// Mirrors vertically (top ↔ bottom, reflects across horizontal axis).
    ///
    /// # Panics
    ///
    /// Panics on arithmetic overflow.
    #[must_use]
    pub const fn flip_vertical(self) -> Self {
        let col = self.index % Self::WIDTH;
        let twice_col = col.checked_mul(2).expect("flip_vertical: overflow");
        let sum = Self::FLIP_V_BASE
            .checked_add(twice_col)
            .expect("flip_vertical: overflow");
        let result = sum
            .checked_sub(self.index)
            .expect("flip_vertical: underflow");
        Self::new(result)
    }

    /// Mirrors horizontally (left ↔ right, reflects across vertical axis).
    ///
    /// # Panics
    ///
    /// Panics on arithmetic overflow.
    #[must_use]
    pub const fn flip_horizontal(self) -> Self {
        let col = self.index % Self::WIDTH;
        let sum = self
            .index
            .checked_add(Self::WIDTH - 1)
            .expect("flip_horizontal: overflow");
        let twice_col = col.checked_mul(2).expect("flip_horizontal: overflow");
        let result = sum
            .checked_sub(twice_col)
            .expect("flip_horizontal: underflow");
        Self::new(result)
    }

    /// Swaps row and column (reflects across main diagonal).
    ///
    /// # Panics
    ///
    /// Panics on arithmetic overflow.
    #[must_use]
    pub const fn transpose(self) -> Self {
        let col = self.index % Self::WIDTH;
        let row = self.index / Self::WIDTH;
        let product = col.checked_mul(Self::WIDTH).expect("transpose: overflow");
        let result = product.checked_add(row).expect("transpose: overflow");
        Self::new(result)
    }

    /// Reflects through the center point (180° rotation).
    ///
    /// # Panics
    ///
    /// Panics on arithmetic overflow.
    #[must_use]
    pub const fn invert(self) -> Self {
        let result = (Self::COUNT - 1)
            .checked_sub(self.index)
            .expect("invert: underflow");
        Self::new(result)
    }
}

impl From<PositionId> for usize {
    fn from(position_id: PositionId) -> Self {
        position_id.index
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pos(row: usize, col: usize) -> PositionId {
        PositionId::from_position(Position::new(row, col))
    }

    #[test]
    fn row_and_col_from_position() {
        let id = pos(2, 7);

        assert_eq!(id.row(), 2);
        assert_eq!(id.col(), 7);
    }

    #[test]
    fn center_returns_middle_position() {
        let center = PositionId::center();

        assert_eq!(center.row(), 7);
        assert_eq!(center.col(), 7);
    }

    #[test]
    fn from_offset_returns_valid_position() {
        let center = PositionId::center();
        let offset = Offset::new(1, 1);

        let result = center.from_offset(offset);

        assert_eq!(result, Some(pos(8, 8)));
    }

    #[test]
    fn from_offset_returns_none_when_out_of_bounds() {
        let corner = pos(0, 0);
        let offset = Offset::new(-1, 0);

        let result = corner.from_offset(offset);

        assert_eq!(result, None);
    }

    #[test]
    fn neighbor_count_counts_valid_neighbors() {
        let corner = pos(0, 0);
        let offsets = [
            Offset::new(0, 1),
            Offset::new(1, 0),
            Offset::new(0, -1),
            Offset::new(-1, 0),
        ];

        assert_eq!(corner.neighbor_count(&offsets), 2);
    }

    #[test]
    fn neighbor_count_at_center() {
        let center = PositionId::center();
        let offsets = [
            Offset::new(0, 1),
            Offset::new(1, 0),
            Offset::new(0, -1),
            Offset::new(-1, 0),
        ];

        assert_eq!(center.neighbor_count(&offsets), 4);
    }

    #[test]
    fn converts_to_usize() {
        let corner = pos(0, 0);
        let center = PositionId::center();

        assert_eq!(usize::from(corner), 0);
        assert_eq!(usize::from(center), 112);
    }

    // ---- Grid transformation primitive tests ----

    #[test]
    fn flip_vertical_mirrors_top_to_bottom() {
        assert_eq!(pos(0, 0).flip_vertical(), pos(14, 0));
        assert_eq!(pos(0, 14).flip_vertical(), pos(14, 14));
        assert_eq!(pos(14, 0).flip_vertical(), pos(0, 0));
        assert_eq!(pos(2, 5).flip_vertical(), pos(12, 5));
    }

    #[test]
    fn flip_vertical_is_self_inverse() {
        for p in PositionId::iter() {
            assert_eq!(p.flip_vertical().flip_vertical(), p);
        }
    }

    #[test]
    fn flip_horizontal_mirrors_left_to_right() {
        assert_eq!(pos(0, 0).flip_horizontal(), pos(0, 14));
        assert_eq!(pos(0, 14).flip_horizontal(), pos(0, 0));
        assert_eq!(pos(14, 0).flip_horizontal(), pos(14, 14));
        assert_eq!(pos(5, 2).flip_horizontal(), pos(5, 12));
    }

    #[test]
    fn flip_horizontal_is_self_inverse() {
        for p in PositionId::iter() {
            assert_eq!(p.flip_horizontal().flip_horizontal(), p);
        }
    }

    #[test]
    fn transpose_swaps_row_and_col() {
        assert_eq!(pos(0, 5).transpose(), pos(5, 0));
        assert_eq!(pos(2, 3).transpose(), pos(3, 2));
        assert_eq!(pos(14, 0).transpose(), pos(0, 14));
    }

    #[test]
    fn transpose_is_self_inverse() {
        for p in PositionId::iter() {
            assert_eq!(p.transpose().transpose(), p);
        }
    }

    #[test]
    fn invert_reflects_through_center() {
        assert_eq!(pos(0, 0).invert(), pos(14, 14));
        assert_eq!(pos(0, 14).invert(), pos(14, 0));
        assert_eq!(pos(14, 0).invert(), pos(0, 14));
        assert_eq!(pos(14, 14).invert(), pos(0, 0));
    }

    #[test]
    fn invert_is_self_inverse() {
        for p in PositionId::iter() {
            assert_eq!(p.invert().invert(), p);
        }
    }

    #[test]
    fn center_is_fixed_by_all_transforms() {
        let center = PositionId::center();
        assert_eq!(center.flip_vertical(), center);
        assert_eq!(center.flip_horizontal(), center);
        assert_eq!(center.transpose(), center);
        assert_eq!(center.invert(), center);
    }
}
