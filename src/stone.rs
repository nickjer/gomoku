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
