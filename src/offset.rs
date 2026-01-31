use std::ops::Neg;

/// A direction vector representing a change in row and column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Offset {
    row_delta: isize,
    col_delta: isize,
}

impl Offset {
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
}
