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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order_variant_produces_valid_child() {
        let parent1 = vec![1, 2, 3, 4, 5];
        let parent2 = vec![5, 4, 3, 2, 1];
        let mut rng = fastrand::Rng::with_seed(42);

        let child = Crossover::Order(Order::new()).crossover(&parent1, &parent2, &mut rng);

        let mut sorted = child.clone();
        sorted.sort();
        assert_eq!(sorted, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn pmx_variant_produces_valid_child() {
        let parent1 = vec![1, 2, 3, 4, 5];
        let parent2 = vec![5, 4, 3, 2, 1];
        let mut rng = fastrand::Rng::with_seed(42);

        let child = Crossover::Pmx(Pmx::new()).crossover(&parent1, &parent2, &mut rng);

        let mut sorted = child.clone();
        sorted.sort();
        assert_eq!(sorted, vec![1, 2, 3, 4, 5]);
    }
}
