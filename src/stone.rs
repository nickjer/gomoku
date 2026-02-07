/// Represents a stone on the Gomoku board.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stone {
    Black,
    White,
    Empty,
}

impl Stone {
    /// Returns the opponent's stone color.
    ///
    /// # Panics
    ///
    /// Panics if called on `Stone::Empty`.
    #[must_use]
    pub fn opponent(self) -> Self {
        match self {
            Stone::Black => Stone::White,
            Stone::White => Stone::Black,
            Stone::Empty => panic!("Empty has no opponent"),
        }
    }
}
