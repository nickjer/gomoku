mod order;
mod pmx;
mod uniform;

pub use order::order_crossover;
pub use pmx::pmx_crossover;
pub use uniform::uniform_crossover;

/// Crossover method for genetic algorithms.
#[derive(Debug, Default, Clone, Copy)]
pub enum Crossover {
    /// Order crossover (OX) for permutation genes.
    #[default]
    Order,
    /// Partially Mapped Crossover (PMX) for permutation genes.
    Pmx,
    /// Uniform crossover for continuous genes.
    Uniform,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gene::Gene;

    fn genes(values: &[usize]) -> Vec<Gene> {
        values.iter().copied().map(Gene::new).collect()
    }

    #[test]
    fn order_crossover_produces_valid_child() {
        let parent1 = genes(&[0, 1, 2, 3, 4]);
        let parent2 = genes(&[4, 3, 2, 1, 0]);
        let mut rng = fastrand::Rng::with_seed(42);

        let child = order_crossover(&parent1, &parent2, &mut rng);

        let mut sorted: Vec<_> = child.iter().map(|g| g.index()).collect();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn pmx_crossover_produces_valid_child() {
        let parent1 = genes(&[0, 1, 2, 3, 4]);
        let parent2 = genes(&[4, 3, 2, 1, 0]);
        let mut rng = fastrand::Rng::with_seed(42);

        let child = pmx_crossover(&parent1, &parent2, &mut rng);

        let mut sorted: Vec<_> = child.iter().map(|g| g.index()).collect();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }
}
