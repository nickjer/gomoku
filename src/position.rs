use std::fmt;

/// A human-readable board position with row and column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Position {
    row: usize,
    col: usize,
}

impl Position {
    #[must_use]
    pub const fn new(row: usize, col: usize) -> Self {
        Self { row, col }
    }

    #[must_use]
    pub const fn row(&self) -> usize {
        self.row
    }

    #[must_use]
    pub const fn col(&self) -> usize {
        self.col
    }
}

/// Converts a 0-indexed column number to Excel-style column label (A, B, ..., Z, AA, AB, ...).
fn col_to_label(mut col: usize) -> String {
    let mut result = String::new();
    col += 1; // Convert to 1-indexed
    while col > 0 {
        col -= 1;
        let c = u8::try_from(col % 26).expect("mod 26 fits in u8") + b'A';
        result.insert(0, char::from(c));
        col /= 26;
    }
    result
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let col_label = col_to_label(self.col);
        let row_display = self.row + 1;
        write!(f, "{col_label}{row_display}")
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

    #[test]
    fn col_label_single_letters() {
        assert_eq!(col_to_label(0), "A");
        assert_eq!(col_to_label(25), "Z");
    }

    #[test]
    fn col_label_double_letters() {
        assert_eq!(col_to_label(26), "AA");
        assert_eq!(col_to_label(27), "AB");
        assert_eq!(col_to_label(51), "AZ");
        assert_eq!(col_to_label(52), "BA");
        assert_eq!(col_to_label(701), "ZZ");
    }

    #[test]
    fn col_label_triple_letters() {
        assert_eq!(col_to_label(702), "AAA");
    }
}
