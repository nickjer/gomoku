use std::ops::Neg;

/// A direction vector representing a change in row and column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Offset {
    pub row_delta: i8,
    pub col_delta: i8,
}

impl Offset {
    #[must_use]
    pub const fn new(row_delta: i8, col_delta: i8) -> Self {
        Self { row_delta, col_delta }
    }
}

impl Neg for Offset {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.row_delta, -self.col_delta)
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
