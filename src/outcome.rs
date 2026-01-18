/// The result of a completed game.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    BlackWins,
    WhiteWins,
    Draw,
}
