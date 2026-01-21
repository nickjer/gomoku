mod insert;
mod inversion;
mod swap;

use enum_dispatch::enum_dispatch;

use crate::gene::Gene;

pub use insert::Insert;
pub use inversion::Inversion;
pub use swap::Swap;

/// Trait for mutation operations.
#[enum_dispatch]
pub trait RunMutation {
    fn mutate(&self, genes: &[Gene], rng: &mut fastrand::Rng) -> Vec<Gene>;
}

/// Enum for polymorphic mutation dispatch.
#[enum_dispatch(RunMutation)]
#[derive(Debug, Clone, Copy)]
pub enum Mutation {
    Swap,
    Insert,
    Inversion,
}

impl Default for Mutation {
    fn default() -> Self {
        Swap.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn genes(values: &[usize]) -> Vec<Gene> {
        values.iter().copied().map(Gene::new).collect()
    }

    #[test]
    fn swap_variant_produces_valid_mutation() {
        let g = genes(&[0, 1, 2, 3, 4]);
        let mut rng = fastrand::Rng::with_seed(42);

        let mutation: Mutation = Swap.into();
        let result = mutation.mutate(&g, &mut rng);

        let mut sorted: Vec<_> = result.iter().map(|g| g.index()).collect();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn insert_variant_produces_valid_mutation() {
        let g = genes(&[0, 1, 2, 3, 4]);
        let mut rng = fastrand::Rng::with_seed(42);

        let mutation: Mutation = Insert.into();
        let result = mutation.mutate(&g, &mut rng);

        let mut sorted: Vec<_> = result.iter().map(|g| g.index()).collect();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn inversion_variant_produces_valid_mutation() {
        let g = genes(&[0, 1, 2, 3, 4]);
        let mut rng = fastrand::Rng::with_seed(42);

        let mutation: Mutation = Inversion.into();
        let result = mutation.mutate(&g, &mut rng);

        let mut sorted: Vec<_> = result.iter().map(|g| g.index()).collect();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }
}
