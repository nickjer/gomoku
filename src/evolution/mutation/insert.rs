/// Result of an insert mutation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InsertResult<T> {
    child: Vec<T>,
    moved: (usize, usize),
}

impl<T> InsertResult<T> {
    #[must_use]
    pub fn into_child(self) -> Vec<T> {
        self.child
    }

    /// Returns (from, to) indices of the moved element.
    #[cfg(test)]
    fn moved(&self) -> (usize, usize) {
        self.moved
    }
}

/// Insert mutation: removes an element and reinserts it at a different position.
#[derive(Debug, Default, Clone, Copy)]
pub struct Insert;

impl Insert {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// # Panics
    ///
    /// Panics if genes has fewer than 2 elements.
    pub fn mutate<T: Clone>(&self, genes: &[T], rng: &mut fastrand::Rng) -> InsertResult<T> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moves_element_to_new_position() {
        let genes = vec![1, 2, 3, 4, 5];
        let mut rng = fastrand::Rng::with_seed(42);

        let result = Insert::new().mutate(&genes, &mut rng);
        let (from, to) = result.moved();

        assert_eq!(result.child[to], genes[from]);
    }

    #[test]
    fn preserves_length() {
        let genes = vec![1, 2, 3, 4, 5];
        let mut rng = fastrand::Rng::with_seed(42);

        let result = Insert::new().mutate(&genes, &mut rng);

        assert_eq!(result.child.len(), genes.len());
    }

    #[test]
    fn indices_are_different() {
        let genes = vec![1, 2, 3, 4, 5];
        let mut rng = fastrand::Rng::with_seed(42);

        for _ in 0..100 {
            let result = Insert::new().mutate(&genes, &mut rng);
            let (from, to) = result.moved();
            assert_ne!(from, to);
        }
    }

    #[test]
    #[should_panic(expected = "insert requires at least 2 genes")]
    fn panics_with_single_element() {
        let genes = vec![1];
        let mut rng = fastrand::Rng::with_seed(42);

        Insert::new().mutate(&genes, &mut rng);
    }

    #[test]
    fn preserves_all_elements() {
        let genes = vec![1, 2, 3, 4, 5];
        let mut rng = fastrand::Rng::with_seed(42);

        let result = Insert::new().mutate(&genes, &mut rng);
        let mut sorted = result.into_child();
        sorted.sort();

        assert_eq!(sorted, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn elements_outside_range_unchanged() {
        let genes = vec![1, 2, 3, 4, 5];
        let mut rng = fastrand::Rng::with_seed(42);

        for _ in 0..100 {
            let result = Insert::new().mutate(&genes, &mut rng);
            let (from, to) = result.moved();
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
            let result = Insert::new().mutate(&genes, &mut rng);
            let (from, to) = result.moved();

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
            let result = Insert::new().mutate(&genes, &mut rng);
            let (from, to) = result.moved();

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
