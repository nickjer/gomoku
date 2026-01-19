use super::RunMutation;

/// Insert mutation: removes an element and reinserts it at a different position.
#[derive(Debug, Default, Clone, Copy)]
pub struct Insert;

impl Insert {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    fn mutate_internal<T: Clone>(
        &self,
        genes: &[T],
        rng: &mut fastrand::Rng,
    ) -> (Vec<T>, (usize, usize)) {
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
    fn mutate<T: Clone>(&self, genes: &[T], rng: &mut fastrand::Rng) -> Vec<T> {
        self.mutate_internal(genes, rng).0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mutate_returns_child_from_internal() {
        let genes = vec![1, 2, 3, 4, 5];

        let (child, _) = Insert.mutate_internal(&genes, &mut fastrand::Rng::with_seed(42));
        let public = Insert.mutate(&genes, &mut fastrand::Rng::with_seed(42));

        assert_eq!(public, child);
    }

    #[test]
    fn moves_element_to_new_position() {
        let genes = vec![1, 2, 3, 4, 5];
        let mut rng = fastrand::Rng::with_seed(42);

        let (child, (from, to)) = Insert.mutate_internal(&genes, &mut rng);

        assert_eq!(child[to], genes[from]);
    }

    #[test]
    fn preserves_length() {
        let genes = vec![1, 2, 3, 4, 5];
        let mut rng = fastrand::Rng::with_seed(42);

        let (child, _) = Insert.mutate_internal(&genes, &mut rng);

        assert_eq!(child.len(), genes.len());
    }

    #[test]
    fn indices_are_different() {
        let genes = vec![1, 2, 3, 4, 5];
        let mut rng = fastrand::Rng::with_seed(42);

        for _ in 0..100 {
            let (_, (from, to)) = Insert.mutate_internal(&genes, &mut rng);
            assert_ne!(from, to);
        }
    }

    #[test]
    #[should_panic(expected = "insert requires at least 2 genes")]
    fn panics_with_single_element() {
        let genes = vec![1];
        let mut rng = fastrand::Rng::with_seed(42);

        Insert.mutate_internal(&genes, &mut rng);
    }

    #[test]
    fn preserves_all_elements() {
        let genes = vec![1, 2, 3, 4, 5];
        let mut rng = fastrand::Rng::with_seed(42);

        let (mut child, _) = Insert.mutate_internal(&genes, &mut rng);
        child.sort();

        assert_eq!(child, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn elements_outside_range_unchanged() {
        let genes = vec![1, 2, 3, 4, 5];
        let mut rng = fastrand::Rng::with_seed(42);

        for _ in 0..100 {
            let (child, (from, to)) = Insert.mutate_internal(&genes, &mut rng);
            let (lo, hi) = if from < to { (from, to) } else { (to, from) };

            for i in 0..lo {
                assert_eq!(child[i], genes[i]);
            }
            for i in (hi + 1)..genes.len() {
                assert_eq!(child[i], genes[i]);
            }
        }
    }

    #[test]
    fn forward_move_shifts_elements_left() {
        let genes = vec![1, 2, 3, 4, 5];
        let mut rng = fastrand::Rng::with_seed(42);

        for _ in 0..1000 {
            let (child, (from, to)) = Insert.mutate_internal(&genes, &mut rng);

            if from < to {
                for i in from..to {
                    assert_eq!(child[i], genes[i + 1]);
                }
                return;
            }
        }
        panic!("no forward move in 1000 iterations");
    }

    #[test]
    fn backward_move_shifts_elements_right() {
        let genes = vec![1, 2, 3, 4, 5];
        let mut rng = fastrand::Rng::with_seed(42);

        for _ in 0..1000 {
            let (child, (from, to)) = Insert.mutate_internal(&genes, &mut rng);

            if from > to {
                for i in (to + 1)..=from {
                    assert_eq!(child[i], genes[i - 1]);
                }
                return;
            }
        }
        panic!("no backward move in 1000 iterations");
    }
}
