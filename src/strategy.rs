use crate::board::Board;
use crate::cache_id::CacheId;
use crate::cache_repository::CacheRepository;
use crate::position_id::PositionId;
use crate::stone::Stone;

/// A strategy for choosing moves in Gomoku.
pub trait Strategy {
    /// Returns the cache dependencies required by this strategy.
    fn cache_dependencies(&self) -> &[CacheId];

    /// Chooses a move for the current player.
    fn choose_move(
        &self,
        current_stone: Stone,
        board: &Board,
        cache_repo: &CacheRepository,
        rng: &mut fastrand::Rng,
    ) -> PositionId;

    /// Returns a label identifying this strategy.
    fn label(&self) -> &str;
}
