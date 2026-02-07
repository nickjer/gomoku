use crate::cache_id::CacheId;
use crate::evolution::EvolvableGenes;
use crate::game_state::GameState;
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
        state: &GameState,
        rng: &mut fastrand::Rng,
    ) -> PositionId;

    /// Returns a label identifying this strategy.
    fn label(&self) -> &str;
}

/// A strategy that can be evolved through genetic algorithms.
pub trait EvolvableStrategy: Strategy + Sized {
    /// The type of genes used by this strategy.
    type Genes: EvolvableGenes;

    /// Creates a strategy with random genes.
    fn random(label: impl Into<String>, rng: &mut fastrand::Rng) -> Self;

    /// Returns the genes of this strategy.
    fn genes(&self) -> &Self::Genes;

    /// Creates a strategy from the given genes.
    fn from_genes(label: impl Into<String>, genes: Self::Genes) -> Self;
}
