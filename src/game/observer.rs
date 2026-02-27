use std::ops::ControlFlow;

use crate::board::Board;
use crate::position_id::PositionId;
use crate::stone::Stone;

/// Hook called after each move is chosen but before it is placed on the board.
///
/// The board reflects the state *before* the chosen move is placed. Return
/// `ControlFlow::Break(())` to abort the game early; `play_from` will return `None`.
pub trait GameObserver {
    fn on_move(&mut self, stone: Stone, position: PositionId, board: &Board) -> ControlFlow<()>;
}

/// No-op observer — zero-sized type used by the [`Play::play`] default implementation.
pub struct NoOpObserver;

impl GameObserver for NoOpObserver {
    fn on_move(&mut self, _: Stone, _: PositionId, _: &Board) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_observer_always_continues() {
        let mut observer = NoOpObserver;
        let board = Board::new();
        let position = PositionId::from_index(0);

        let result = observer.on_move(Stone::Black, position, &board);

        assert!(result.is_continue());
    }
}
