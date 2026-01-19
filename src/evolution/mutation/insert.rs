use super::RunMutation;

/// Result of an insert mutation.
struct InsertResult<T> {
    child: Vec<T>,
    moved: (usize, usize),
}

/// Insert mutation: removes an element and reinserts it at a different position.
#[derive(Debug, Default, Clone, Copy)]
pub struct Insert;

impl Insert {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    fn mutate_internal<T: Clone>(&self, genes: &[T], rng: &mut fastrand::Rng) -> InsertResult<T> {
        assert!(genes.len() >= 2, "insert requires at least 2 genes");

        let from = rng.usize(..genes.len());
        let to = rng.usize(..genes.len() - 1);
        let to = if to >= from { to + 1 } else { to };

        let mut child = genes.to_vec();
        let element = child.remove(from);
        child.insert(to, element);

        InsertResult {
            child,
            moved: (from, to),
        }
    }
}

impl RunMutation for Insert {
    fn mutate<T: Clone>(&self, genes: &[T], rng: &mut fastrand::Rng) -> Vec<T> {
        self.mutate_internal(genes, rng).child
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mutate_returns_child_from_internal() {
        let genes = vec![1, 2, 3, 4, 5];

        let internal = Insert.mutate_internal(&genes, &mut fastrand::Rng::with_seed(42));
        let public = Insert.mutate(&genes, &mut fastrand::Rng::with_seed(42));

        assert_eq!(public, internal.child);
    }

    #[test]
    fn moves_element_to_new_position() {
        let genes = vec![1, 2, 3, 4, 5];
        let mut rng = fastrand::Rng::with_seed(42);

        let result = Insert.mutate_internal(&genes, &mut rng);
        let (from, to) = result.moved;

        assert_eq!(result.child[to], genes[from]);
    }

    #[test]
    fn preserves_length() {
        let genes = vec![1, 2, 3, 4, 5];
        let mut rng = fastrand::Rng::with_seed(42);

        let result = Insert.mutate_internal(&genes, &mut rng);

        assert_eq!(result.child.len(), genes.len());
    }

    #[test]
    fn indices_are_different() {
        let genes = vec![1, 2, 3, 4, 5];
        let mut rng = fastrand::Rng::with_seed(42);

        for _ in 0..100 {
            let result = Insert.mutate_internal(&genes, &mut rng);
            let (from, to) = result.moved;
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

        let result = Insert.mutate_internal(&genes, &mut rng);
        let mut sorted = result.child;
        sorted.sort();

        assert_eq!(sorted, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn elements_outside_range_unchanged() {
        let genes = vec![1, 2, 3, 4, 5];
        let mut rng = fastrand::Rng::with_seed(42);

        for _ in 0..100 {
            let result = Insert.mutate_internal(&genes, &mut rng);
            let (from, to) = result.moved;
            let (lo, hi) = if from < to { (from, to) } else { (to, from) };

            for i in 0..lo {
                assert_eq!(result.child[i], genes[i]);
            }
            for i in (hi + 1)..genes.len() {
                assert_eq!(result.child[i], genes[i]);
            }
        }
    }

    #[test]
    fn forward_move_shifts_elements_left() {
        let genes = vec![1, 2, 3, 4, 5];
        let mut rng = fastrand::Rng::with_seed(42);

        for _ in 0..1000 {
            let result = Insert.mutate_internal(&genes, &mut rng);
            let (from, to) = result.moved;

            if from < to {
                for i in from..to {
                    assert_eq!(result.child[i], genes[i + 1]);
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
            let result = Insert.mutate_internal(&genes, &mut rng);
            let (from, to) = result.moved;

            if from > to {
                for i in (to + 1)..=from {
                    assert_eq!(result.child[i], genes[i - 1]);
                }
                return;
            }
        }
        panic!("no backward move in 1000 iterations");
    }
}
