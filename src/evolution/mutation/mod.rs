mod insert;
mod inversion;
mod swap;

use enum_dispatch::enum_dispatch;

pub use insert::Insert;
pub use inversion::Inversion;
pub use swap::Swap;

/// Trait for mutation operations.
#[enum_dispatch]
pub trait RunMutation {
    fn mutate<T: Clone>(&self, genes: &[T], rng: &mut fastrand::Rng) -> Vec<T>;
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

    #[test]
    fn swap_variant_produces_valid_mutation() {
        let genes = vec![1, 2, 3, 4, 5];
        let mut rng = fastrand::Rng::with_seed(42);

        let mutation: Mutation = Swap.into();
        let result = mutation.mutate(&genes, &mut rng);

        let mut sorted = result.clone();
        sorted.sort();
        assert_eq!(sorted, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn insert_variant_produces_valid_mutation() {
        let genes = vec![1, 2, 3, 4, 5];
        let mut rng = fastrand::Rng::with_seed(42);

        let mutation: Mutation = Insert.into();
        let result = mutation.mutate(&genes, &mut rng);

        let mut sorted = result.clone();
        sorted.sort();
        assert_eq!(sorted, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn inversion_variant_produces_valid_mutation() {
        let genes = vec![1, 2, 3, 4, 5];
        let mut rng = fastrand::Rng::with_seed(42);

        let mutation: Mutation = Inversion.into();
        let result = mutation.mutate(&genes, &mut rng);

        let mut sorted = result.clone();
        sorted.sort();
        assert_eq!(sorted, vec![1, 2, 3, 4, 5]);
    }
}
