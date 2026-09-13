use crate::board::Board;
use crate::evolution::EvolvableGenes;
use crate::position_id::PositionId;
use crate::stone::Stone;

/// A strategy for choosing moves in Gomoku.
pub trait Strategy {
    /// Chooses a move for the current player.
    fn choose_move(
        &self,
        current_stone: Stone,
        board: &Board,
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

    /// The empty positions from the one this strategy likes most for the
    /// current player to the one it likes least, reading the board as it
    /// lies. Positions it cannot tell apart come in random order.
    fn rank_positions(
        &self,
        current_stone: Stone,
        board: &Board,
        rng: &mut fastrand::Rng,
    ) -> Vec<PositionId>;
}
