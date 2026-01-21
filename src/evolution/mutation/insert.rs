use super::RunMutation;
use crate::gene::Gene;

/// Insert mutation: removes an element and reinserts it at a different position.
#[derive(Debug, Default, Clone, Copy)]
pub struct Insert;

impl Insert {
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
        assert!(genes.len() >= 2, "insert requires at least 2 genes");

        let from = rng.usize(..genes.len());
        let to = rng.usize(..genes.len() - 1);
        let to = if to >= from { to + 1 } else { to };

        let mut child = genes.to_vec();
        let element = child.remove(from);
        child.insert(to, element);

        (child, (from, to))
    }
}

impl RunMutation for Insert {
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

        let (child, _) = Insert.mutate_internal(&g, &mut fastrand::Rng::with_seed(42));
        let public = Insert.mutate(&g, &mut fastrand::Rng::with_seed(42));

        assert_eq!(public, child);
    }

    #[test]
    fn moves_element_to_new_position() {
        let g = genes(&[0, 1, 2, 3, 4]);
        let mut rng = fastrand::Rng::with_seed(42);

        let (child, (from, to)) = Insert.mutate_internal(&g, &mut rng);

        assert_eq!(child[to], g[from]);
    }

    #[test]
    fn preserves_length() {
        let g = genes(&[0, 1, 2, 3, 4]);
        let mut rng = fastrand::Rng::with_seed(42);

        let (child, _) = Insert.mutate_internal(&g, &mut rng);

        assert_eq!(child.len(), g.len());
    }

    #[test]
    fn indices_are_different() {
        let g = genes(&[0, 1, 2, 3, 4]);
        let mut rng = fastrand::Rng::with_seed(42);

        for _ in 0..100 {
            let (_, (from, to)) = Insert.mutate_internal(&g, &mut rng);
            assert_ne!(from, to);
        }
    }

    #[test]
    #[should_panic(expected = "insert requires at least 2 genes")]
    fn panics_with_single_element() {
        let g = genes(&[0]);
        let mut rng = fastrand::Rng::with_seed(42);

        Insert.mutate_internal(&g, &mut rng);
    }

    #[test]
    fn preserves_all_elements() {
        let g = genes(&[0, 1, 2, 3, 4]);
        let mut rng = fastrand::Rng::with_seed(42);

        let (mut child, _) = Insert.mutate_internal(&g, &mut rng);
        child.sort();

        assert_eq!(child, genes(&[0, 1, 2, 3, 4]));
    }

    #[test]
    fn elements_outside_range_unchanged() {
        let g = genes(&[0, 1, 2, 3, 4]);
        let mut rng = fastrand::Rng::with_seed(42);

        for _ in 0..100 {
            let (child, (from, to)) = Insert.mutate_internal(&g, &mut rng);
            let (lo, hi) = if from < to { (from, to) } else { (to, from) };

            for i in 0..lo {
                assert_eq!(child[i], g[i]);
            }
            for i in (hi + 1)..g.len() {
                assert_eq!(child[i], g[i]);
            }
        }
    }

    #[test]
    fn forward_move_shifts_elements_left() {
        let g = genes(&[0, 1, 2, 3, 4]);
        let mut rng = fastrand::Rng::with_seed(42);

        for _ in 0..1000 {
            let (child, (from, to)) = Insert.mutate_internal(&g, &mut rng);

            if from < to {
                for i in from..to {
                    assert_eq!(child[i], g[i + 1]);
                }
                return;
            }
        }
        panic!("no forward move in 1000 iterations");
    }

    #[test]
    fn backward_move_shifts_elements_right() {
        let g = genes(&[0, 1, 2, 3, 4]);
        let mut rng = fastrand::Rng::with_seed(42);

        for _ in 0..1000 {
            let (child, (from, to)) = Insert.mutate_internal(&g, &mut rng);

            if from > to {
                for i in (to + 1)..=from {
                    assert_eq!(child[i], g[i - 1]);
                }
                return;
            }
        }
        panic!("no backward move in 1000 iterations");
    }
}
