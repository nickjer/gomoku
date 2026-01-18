use std::fmt;

/// A human-readable board position with row and column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Position {
    pub row: u8,
    pub col: u8,
}

impl Position {
    #[must_use]
    pub const fn new(row: u8, col: u8) -> Self {
        Self { row, col }
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let col_char = char::from(b'A' + self.col);
        write!(f, "{}{}", col_char, self.row + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_top_left_corner() {
        let position = Position::new(0, 0);

        assert_eq!(position.to_string(), "A1");
    }

    #[test]
    fn display_bottom_right_corner() {
        let position = Position::new(14, 14);

        assert_eq!(position.to_string(), "O15");
    }

    #[test]
    fn display_middle_position() {
        let position = Position::new(9, 12);

        assert_eq!(position.to_string(), "M10");
    }
}
