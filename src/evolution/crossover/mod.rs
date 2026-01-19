mod order;
mod pmx;

use std::hash::Hash;

use enum_dispatch::enum_dispatch;

pub use order::Order;
pub use pmx::Pmx;

/// Trait for crossover operations.
#[enum_dispatch]
pub trait RunCrossover {
    fn crossover<T: Clone + Eq + Hash>(
        &self,
        parent1: &[T],
        parent2: &[T],
        rng: &mut fastrand::Rng,
    ) -> Vec<T>;
}

/// Enum for polymorphic crossover dispatch.
#[enum_dispatch(RunCrossover)]
#[derive(Debug, Clone, Copy)]
pub enum Crossover {
    Order,
    Pmx,
}

impl Default for Crossover {
    fn default() -> Self {
        Order.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order_variant_produces_valid_child() {
        let parent1 = vec![1, 2, 3, 4, 5];
        let parent2 = vec![5, 4, 3, 2, 1];
        let mut rng = fastrand::Rng::with_seed(42);

        let crossover: Crossover = Order.into();
        let child = crossover.crossover(&parent1, &parent2, &mut rng);

        let mut sorted = child.clone();
        sorted.sort();
        assert_eq!(sorted, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn pmx_variant_produces_valid_child() {
        let parent1 = vec![1, 2, 3, 4, 5];
        let parent2 = vec![5, 4, 3, 2, 1];
        let mut rng = fastrand::Rng::with_seed(42);

        let crossover: Crossover = Pmx.into();
        let child = crossover.crossover(&parent1, &parent2, &mut rng);

        let mut sorted = child.clone();
        sorted.sort();
        assert_eq!(sorted, vec![1, 2, 3, 4, 5]);
    }
}
