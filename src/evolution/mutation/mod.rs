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
