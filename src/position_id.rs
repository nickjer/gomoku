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
}
