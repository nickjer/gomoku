use std::fmt;

/// Represents a stone color on the Gomoku board.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stone {
    Black,
    White,
}

impl Stone {
    /// Returns the opponent's stone color.
    #[must_use]
    pub const fn opponent(self) -> Self {
        match self {
            Self::Black => Self::White,
            Self::White => Self::Black,
        }
    }
}

/// The color name with the symbol the board display uses for it, e.g. `Black (X)`.
impl fmt::Display for Stone {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Black => write!(f, "Black (X)"),
            Self::White => write!(f, "White (O)"),
        }
    }
}

impl From<Stone> for usize {
    fn from(stone: Stone) -> usize {
        match stone {
            Stone::Black => 0,
            Stone::White => 1,
        }
    }
}
