use super::RunMutation;
use crate::gene::Gene;

/// Swap mutation: exchanges two random elements.
#[derive(Debug, Default, Clone, Copy)]
pub struct Swap;

impl Swap {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    #[allow(clippy::unused_self)]
    fn mutate_internal(
        self,
        genes: &[Gene],
        rng: &mut fastrand::Rng,
    ) -> (Vec<Gene>, (usize, usize)) {
        assert!(genes.len() >= 2, "swap requires at least 2 genes");

        let a = rng.usize(..genes.len());
        let b = rng.usize(..genes.len() - 1);
        let b = if b >= a { b + 1 } else { b };

        let mut child = genes.to_vec();
        child.swap(a, b);

        (child, (a, b))
    }
}

impl RunMutation for Swap {
    fn mutate(&self, genes: &[Gene], rng: &mut fastrand::Rng) -> Vec<Gene> {
        self.mutate_internal(genes, rng).0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn genes(values: &[usize]) -> Vec<Gene> {
        values.iter().copied().map(Gene::new).collect()
    }

    #[test]
    fn mutate_returns_child_from_internal() {
        let g = genes(&[0, 1, 2, 3, 4]);

        let (child, _) = Swap.mutate_internal(&g, &mut fastrand::Rng::with_seed(42));
        let public = Swap.mutate(&g, &mut fastrand::Rng::with_seed(42));

        assert_eq!(public, child);
    }

    #[test]
    fn swaps_two_elements() {
        let g = genes(&[0, 1, 2, 3, 4]);
        let mut rng = fastrand::Rng::with_seed(42);

        let (child, (a, b)) = Swap.mutate_internal(&g, &mut rng);

        assert_eq!(child[a], g[b]);
        assert_eq!(child[b], g[a]);
    }

    #[test]
    fn preserves_length() {
        let g = genes(&[0, 1, 2, 3, 4]);
        let mut rng = fastrand::Rng::with_seed(42);

        let (child, _) = Swap.mutate_internal(&g, &mut rng);

        assert_eq!(child.len(), g.len());
    }

    #[test]
    fn indices_are_different() {
        let g = genes(&[0, 1, 2, 3, 4]);
        let mut rng = fastrand::Rng::with_seed(42);

        for _ in 0..100 {
            let (_, (a, b)) = Swap.mutate_internal(&g, &mut rng);
            assert_ne!(a, b);
        }
    }

    #[test]
    #[should_panic(expected = "swap requires at least 2 genes")]
    fn panics_with_single_element() {
        let g = genes(&[0]);
        let mut rng = fastrand::Rng::with_seed(42);

        Swap.mutate_internal(&g, &mut rng);
    }

    #[test]
    fn preserves_all_elements() {
        let g = genes(&[0, 1, 2, 3, 4]);
        let mut rng = fastrand::Rng::with_seed(42);

        let (mut child, _) = Swap.mutate_internal(&g, &mut rng);
        child.sort();

        assert_eq!(child, genes(&[0, 1, 2, 3, 4]));
    }
}
