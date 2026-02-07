use std::fmt;

use anyhow::Result;

use crate::board::Board;
use crate::cache_repository::CacheRepository;
use crate::outcome::Outcome;
use crate::position_id::PositionId;
use crate::stone::Stone;
use crate::strategy::Strategy;

/// Combined board and cache state for a Gomoku game.
///
/// Ensures board and cache stay synchronized: every placement
/// atomically updates both.
#[derive(Debug, Clone)]
pub struct GameState {
    board: Board,
    cache_repo: CacheRepository,
}

impl GameState {
    #[must_use]
    pub fn new() -> Self {
        Self {
            board: Board::new(),
            cache_repo: CacheRepository::new(),
        }
    }

    /// Activates all cache dependencies required by the given strategy.
    pub fn activate_for(&mut self, strategy: &dyn Strategy) {
        for &dep in strategy.cache_dependencies() {
            self.cache_repo.activate(dep);
        }
    }

    /// Places a stone, updating both the board and cache atomically.
    ///
    /// # Errors
    ///
    /// Returns an error if the placement is invalid (see [`Board::place`]).
    pub fn place(&mut self, position_id: PositionId, stone: Stone) -> Result<()> {
        self.board.place(position_id, stone)?;
        self.cache_repo.place(position_id, stone);
        Ok(())
    }

    #[must_use]
    pub fn cache(&self) -> &CacheRepository {
        &self.cache_repo
    }

    #[must_use]
    pub fn stone(&self, position_id: PositionId) -> Stone {
        self.board.stone(position_id)
    }

    #[must_use]
    pub fn empty_position_ids(&self) -> &[PositionId] {
        self.board.empty_position_ids()
    }

    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.board.is_finished()
    }

    #[must_use]
    pub fn outcome(&self) -> Option<Outcome> {
        self.board.outcome()
    }

    #[must_use]
    pub fn is_full(&self) -> bool {
        self.board.is_full()
    }
}

impl Default for GameState {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for GameState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.board.fmt(f)
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
    fn new_state_has_empty_board() {
        let state = GameState::new();

        assert_eq!(state.stone(pos(0, 0)), Stone::Empty);
        assert_eq!(state.empty_position_ids().len(), PositionId::COUNT);
    }

    #[test]
    fn place_updates_board_and_cache() {
        let mut state = GameState::new();
        let center = PositionId::center();

        state.place(center, Stone::Black).unwrap();

        assert_eq!(state.stone(center), Stone::Black);
        assert!(!state.empty_position_ids().contains(&center));
    }

    #[test]
    fn place_returns_error_for_occupied_position() {
        let mut state = GameState::new();
        let center = PositionId::center();
        state.place(center, Stone::Black).unwrap();

        let err = state.place(center, Stone::White).unwrap_err();

        assert_eq!(err.to_string(), "Position is not empty");
    }

    #[test]
    fn display_delegates_to_board() {
        let state = GameState::new();
        let expected_row = vec!["·"; 15].join(" ");
        let expected = vec![expected_row; 15].join("\n");

        assert_eq!(state.to_string(), expected);
    }
}
