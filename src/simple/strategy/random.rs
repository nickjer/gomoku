use crate::cache_id::CacheId;
use crate::game_state::GameState;
use crate::position_id::PositionId;
use crate::stone::Stone;
use crate::strategy::Strategy;

/// A strategy that chooses a random empty position.
pub struct Random {
    label: String,
}

impl Random {
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
        }
    }
}

impl Strategy for Random {
    fn cache_dependencies(&self) -> &[CacheId] {
        &[]
    }

    fn choose_move(
        &self,
        _current_stone: Stone,
        state: &GameState,
        rng: &mut fastrand::Rng,
    ) -> PositionId {
        let empty = state.empty_position_ids();
        let index = rng.usize(..empty.len());
        empty[index]
    }

    fn label(&self) -> &str {
        &self.label
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chooses_from_empty_positions() {
        let strategy = Random::new("test");
        let state = GameState::new();
        let mut rng = fastrand::Rng::new();

        let chosen = strategy.choose_move(Stone::Black, &state, &mut rng);

        assert!(state.empty_position_ids().contains(&chosen));
    }

    #[test]
    fn has_no_cache_dependencies() {
        let strategy = Random::new("test");

        assert!(strategy.cache_dependencies().is_empty());
    }
}
