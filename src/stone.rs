/// Represents a stone on the Gomoku board.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stone {
    Black,
    White,
    Empty,
}
