mod order;
mod pmx;

pub use order::{Order, OrderResult};
pub use pmx::{Pmx, PmxResult};

/// Enum for polymorphic crossover dispatch.
#[derive(Debug, Clone, Copy)]
pub enum Crossover {
    Order(Order),
    Pmx(Pmx),
}

impl Crossover {
    pub fn crossover<T: Clone + Eq + std::hash::Hash>(
        &self,
        parent1: &[T],
        parent2: &[T],
        rng: &mut fastrand::Rng,
    ) -> Vec<T> {
        match self {
            Crossover::Order(c) => c.crossover(parent1, parent2, rng).into_child(),
            Crossover::Pmx(c) => c.crossover(parent1, parent2, rng).into_child(),
        }
    }
}

impl Default for Crossover {
    fn default() -> Self {
        Crossover::Order(Order::new())
    }
}
