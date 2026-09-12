use std::ops::Neg;

/// A direction vector representing a change in row and column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Offset {
    row_delta: isize,
    col_delta: isize,
}

impl Offset {
    /// A position itself followed by its eight neighbors, clockwise from north.
    ///
    /// Odd indices (1, 3, 5, 7) are straight across; even ones after the
    /// center are diagonal.
    pub const CENTER_AND_NEIGHBORS: [Self; 9] = [
        Self::new(0, 0),   // 0: center
        Self::new(-1, 0),  // 1: N
        Self::new(-1, 1),  // 2: NE
        Self::new(0, 1),   // 3: E
        Self::new(1, 1),   // 4: SE
        Self::new(1, 0),   // 5: S
        Self::new(1, -1),  // 6: SW
        Self::new(0, -1),  // 7: W
        Self::new(-1, -1), // 8: NW
    ];

    #[must_use]
    pub const fn new(row_delta: isize, col_delta: isize) -> Self {
        Self {
            row_delta,
            col_delta,
        }
    }

    #[must_use]
    pub const fn row_delta(&self) -> isize {
        self.row_delta
    }

    #[must_use]
    pub const fn col_delta(&self) -> isize {
        self.col_delta
    }
}

impl Neg for Offset {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(
            self.row_delta.checked_neg().unwrap(),
            self.col_delta.checked_neg().unwrap(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negation_inverts_deltas() {
        let offset = Offset::new(1, -2);

        assert_eq!(-offset, Offset::new(-1, 2));
    }

    #[test]
    fn center_and_neighbors_starts_at_the_center_and_goes_clockwise() {
        let [center, north, north_east, .., west, north_west] = Offset::CENTER_AND_NEIGHBORS;

        assert_eq!(center, Offset::new(0, 0));
        assert_eq!(north, Offset::new(-1, 0));
        assert_eq!(north_east, Offset::new(-1, 1));
        assert_eq!(west, Offset::new(0, -1));
        assert_eq!(north_west, Offset::new(-1, -1));
    }

    #[test]
    fn center_and_neighbors_covers_every_offset_within_one_step_once() {
        let mut offsets = Offset::CENTER_AND_NEIGHBORS.to_vec();
        offsets.sort_by_key(|offset| (offset.row_delta(), offset.col_delta()));
        offsets.dedup();

        assert_eq!(offsets.len(), 9);
        assert!(
            offsets
                .iter()
                .all(|offset| offset.row_delta().abs() <= 1 && offset.col_delta().abs() <= 1)
        );
    }
}
