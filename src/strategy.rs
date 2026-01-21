use crate::board::Board;
use crate::cache_id::CacheId;
use crate::cache_repository::CacheRepository;
use crate::gene::Gene;
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

/// A strategy that can be evolved through genetic algorithms.
pub trait EvolvableStrategy: Strategy + Sized {
    /// Generates random genes for a new strategy.
    fn random_genes(rng: &mut fastrand::Rng) -> Vec<Gene>;

    /// Creates a strategy from the given genes.
    fn from_genes(label: impl Into<String>, genes: Vec<Gene>) -> Self;

    /// Returns the genes of this strategy.
    fn genes(&self) -> &[Gene];

    /// Creates a strategy with random genes.
    fn random(label: impl Into<String>, rng: &mut fastrand::Rng) -> Self {
        Self::from_genes(label, Self::random_genes(rng))
    }
}
