use crate::board::Board;
use crate::cache_id::CacheId;
use crate::cache_repository::CacheRepository;
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
    fn cache_dependencies(&self) -> &[CacheId] {
        &[]
    }

    fn choose_move(
        &self,
        _current_stone: Stone,
        board: &Board,
        _cache_repo: &CacheRepository,
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
        let cache_repo = CacheRepository::new();
        let mut rng = fastrand::Rng::new();

        let chosen = strategy.choose_move(Stone::Black, &board, &cache_repo, &mut rng);

        assert_eq!(chosen, pos(0, 0));
    }

    #[test]
    fn has_no_cache_dependencies() {
        let strategy = FirstAvailable::new("test");

        assert!(strategy.cache_dependencies().is_empty());
    }
}
