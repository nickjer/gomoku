use crate::gene::Gene;

use super::crossover::{Crossover, order_crossover, pmx_crossover};
use super::mutation::{Mutation, insert_mutate, inversion_mutate, swap_mutate};

/// Trait for gene types that can be evolved through crossover and mutation.
pub trait EvolvableGenes: Clone {
    /// Performs crossover with another gene set.
    #[must_use]
    fn crossover(&self, other: &Self, crossover: Crossover, rng: &mut fastrand::Rng) -> Self;

    /// Performs mutation on this gene set.
    #[must_use]
    fn mutate(&self, mutation: Mutation, rng: &mut fastrand::Rng) -> Self;
}

impl EvolvableGenes for Vec<Gene> {
    fn crossover(&self, other: &Self, crossover: Crossover, rng: &mut fastrand::Rng) -> Self {
        match crossover {
            Crossover::Order => order_crossover(self, other, rng),
            Crossover::Pmx => pmx_crossover(self, other, rng),
            Crossover::Uniform => {
                panic!("Uniform crossover is not supported for permutation genes")
            }
        }
    }

    fn mutate(&self, mutation: Mutation, rng: &mut fastrand::Rng) -> Self {
        match mutation {
            Mutation::Swap => swap_mutate(self, rng),
            Mutation::Insert => insert_mutate(self, rng),
            Mutation::Inversion => inversion_mutate(self, rng),
            Mutation::Gaussian { .. } => {
                panic!("Gaussian mutation is not supported for permutation genes")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn genes(values: &[usize]) -> Vec<Gene> {
        values.iter().copied().map(Gene::new).collect()
    }

    #[test]
    fn vec_gene_crossover_with_order_produces_valid_permutation() {
        let parent1 = genes(&[0, 1, 2, 3, 4]);
        let parent2 = genes(&[4, 3, 2, 1, 0]);
        let mut rng = fastrand::Rng::with_seed(42);

        let child = parent1.crossover(&parent2, Crossover::Order, &mut rng);

        let mut sorted: Vec<_> = child.iter().map(|g| g.index()).collect();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn vec_gene_crossover_with_pmx_produces_valid_permutation() {
        let parent1 = genes(&[0, 1, 2, 3, 4]);
        let parent2 = genes(&[4, 3, 2, 1, 0]);
        let mut rng = fastrand::Rng::with_seed(42);

        let child = parent1.crossover(&parent2, Crossover::Pmx, &mut rng);

        let mut sorted: Vec<_> = child.iter().map(|g| g.index()).collect();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn vec_gene_mutate_with_swap_produces_valid_permutation() {
        let g = genes(&[0, 1, 2, 3, 4]);
        let mut rng = fastrand::Rng::with_seed(42);

        let result = g.mutate(Mutation::Swap, &mut rng);

        let mut sorted: Vec<_> = result.iter().map(|g| g.index()).collect();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn vec_gene_mutate_with_insert_produces_valid_permutation() {
        let g = genes(&[0, 1, 2, 3, 4]);
        let mut rng = fastrand::Rng::with_seed(42);

        let result = g.mutate(Mutation::Insert, &mut rng);

        let mut sorted: Vec<_> = result.iter().map(|g| g.index()).collect();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn vec_gene_mutate_with_inversion_produces_valid_permutation() {
        let g = genes(&[0, 1, 2, 3, 4]);
        let mut rng = fastrand::Rng::with_seed(42);

        let result = g.mutate(Mutation::Inversion, &mut rng);

        let mut sorted: Vec<_> = result.iter().map(|g| g.index()).collect();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }
}
