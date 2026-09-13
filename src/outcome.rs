use crate::stone::Stone;

/// The result of a completed game.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Win(Stone),
    Draw,
}
