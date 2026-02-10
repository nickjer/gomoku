use super::crossover::Crossover;
use super::mutation::Mutation;

/// Trait for gene types that can be evolved through crossover and mutation.
pub trait EvolvableGenes: Clone {
    /// Performs crossover with another gene set.
    #[must_use]
    fn crossover(&self, other: &Self, crossover: Crossover, rng: &mut fastrand::Rng) -> Self;

    /// Performs mutation on this gene set.
    #[must_use]
    fn mutate(&self, mutation: Mutation, rng: &mut fastrand::Rng) -> Self;
}
