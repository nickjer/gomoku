mod order;
mod pmx;

use enum_dispatch::enum_dispatch;

use crate::gene::Gene;

pub use order::Order;
pub use pmx::Pmx;

/// Trait for crossover operations.
#[enum_dispatch]
pub trait RunCrossover {
    fn crossover(&self, parent1: &[Gene], parent2: &[Gene], rng: &mut fastrand::Rng) -> Vec<Gene>;
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

    fn genes(values: &[usize]) -> Vec<Gene> {
        values.iter().copied().map(Gene::new).collect()
    }

    #[test]
    fn order_variant_produces_valid_child() {
        let parent1 = genes(&[0, 1, 2, 3, 4]);
        let parent2 = genes(&[4, 3, 2, 1, 0]);
        let mut rng = fastrand::Rng::with_seed(42);

        let crossover: Crossover = Order.into();
        let child = crossover.crossover(&parent1, &parent2, &mut rng);

        let mut sorted: Vec<_> = child.iter().map(|g| g.index()).collect();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn pmx_variant_produces_valid_child() {
        let parent1 = genes(&[0, 1, 2, 3, 4]);
        let parent2 = genes(&[4, 3, 2, 1, 0]);
        let mut rng = fastrand::Rng::with_seed(42);

        let crossover: Crossover = Pmx.into();
        let child = crossover.crossover(&parent1, &parent2, &mut rng);

        let mut sorted: Vec<_> = child.iter().map(|g| g.index()).collect();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }
}
