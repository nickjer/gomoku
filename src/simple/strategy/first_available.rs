use crate::board::Board;
use crate::position_id::PositionId;
use crate::stone::Stone;
use crate::strategy::Strategy;

/// A strategy that chooses the first empty position.
pub struct FirstAvailable {
    label: String,
}

impl FirstAvailable {
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
        }
    }
}

impl Strategy for FirstAvailable {
    fn choose_move(
        &self,
        _current_stone: Stone,
        board: &Board,
        _rng: &mut fastrand::Rng,
    ) -> PositionId {
        board.empty_position_ids()[0]
    }

    fn label(&self) -> &str {
        &self.label
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;

    fn pos(row: usize, col: usize) -> PositionId {
        PositionId::from_position(Position::new(row, col))
    }

    #[test]
    fn chooses_first_empty_position() {
        let strategy = FirstAvailable::new("test");
        let board = Board::new();
        let mut rng = fastrand::Rng::new();

        let chosen = strategy.choose_move(Stone::Black, &board, &mut rng);

        assert_eq!(chosen, pos(0, 0));
    }
}
