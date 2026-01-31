mod gaussian;
mod insert;
mod inversion;
mod swap;

pub use gaussian::gaussian_mutate;
pub use insert::insert_mutate;
pub use inversion::inversion_mutate;
pub use swap::swap_mutate;

/// Mutation method for genetic algorithms.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub enum Mutation {
    /// Swap mutation: exchanges two random elements (permutation genes).
    #[default]
    Swap,
    /// Insert mutation: removes an element and reinserts it at a different position (permutation genes).
    Insert,
    /// Inversion mutation: reverses a random segment of the sequence (permutation genes).
    Inversion,
    /// Gaussian mutation: adds N(0, sigma) noise to all weights (continuous genes).
    Gaussian { sigma: f32 },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gene::Gene;

    fn genes(values: &[usize]) -> Vec<Gene> {
        values.iter().copied().map(Gene::new).collect()
    }

    #[test]
    fn swap_mutate_produces_valid_mutation() {
        let g = genes(&[0, 1, 2, 3, 4]);
        let mut rng = fastrand::Rng::with_seed(42);

        let result = swap_mutate(&g, &mut rng);

        let mut sorted: Vec<_> = result.iter().map(|g| g.index()).collect();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn insert_mutate_produces_valid_mutation() {
        let g = genes(&[0, 1, 2, 3, 4]);
        let mut rng = fastrand::Rng::with_seed(42);

        let result = insert_mutate(&g, &mut rng);

        let mut sorted: Vec<_> = result.iter().map(|g| g.index()).collect();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn inversion_mutate_produces_valid_mutation() {
        let g = genes(&[0, 1, 2, 3, 4]);
        let mut rng = fastrand::Rng::with_seed(42);

        let result = inversion_mutate(&g, &mut rng);

        let mut sorted: Vec<_> = result.iter().map(|g| g.index()).collect();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }
}
