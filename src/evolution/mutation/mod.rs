mod insert;
mod inversion;
mod swap;

pub use insert::{Insert, InsertResult};
pub use inversion::{Inversion, InversionResult};
pub use swap::{Swap, SwapResult};

/// Enum for polymorphic mutation dispatch.
#[derive(Debug, Clone, Copy)]
pub enum Mutation {
    Swap(Swap),
    Insert(Insert),
    Inversion(Inversion),
}

impl Mutation {
    pub fn mutate<T: Clone>(&self, genes: &[T], rng: &mut fastrand::Rng) -> Vec<T> {
        match self {
            Mutation::Swap(m) => m.mutate(genes, rng).into_child(),
            Mutation::Insert(m) => m.mutate(genes, rng).into_child(),
            Mutation::Inversion(m) => m.mutate(genes, rng).into_child(),
        }
    }
}

impl Default for Mutation {
    fn default() -> Self {
        Mutation::Swap(Swap::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swap_variant_produces_valid_mutation() {
        let genes = vec![1, 2, 3, 4, 5];
        let mut rng = fastrand::Rng::with_seed(42);

        let result = Mutation::Swap(Swap::new()).mutate(&genes, &mut rng);

        let mut sorted = result.clone();
        sorted.sort();
        assert_eq!(sorted, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn insert_variant_produces_valid_mutation() {
        let genes = vec![1, 2, 3, 4, 5];
        let mut rng = fastrand::Rng::with_seed(42);

        let result = Mutation::Insert(Insert::new()).mutate(&genes, &mut rng);

        let mut sorted = result.clone();
        sorted.sort();
        assert_eq!(sorted, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn inversion_variant_produces_valid_mutation() {
        let genes = vec![1, 2, 3, 4, 5];
        let mut rng = fastrand::Rng::with_seed(42);

        let result = Mutation::Inversion(Inversion::new()).mutate(&genes, &mut rng);

        let mut sorted = result.clone();
        sorted.sort();
        assert_eq!(sorted, vec![1, 2, 3, 4, 5]);
    }
}
