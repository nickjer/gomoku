/// Result of a swap mutation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwapResult<T> {
    child: Vec<T>,
    swapped: (usize, usize),
}

impl<T> SwapResult<T> {
    #[must_use]
    pub fn into_child(self) -> Vec<T> {
        self.child
    }

    #[cfg(test)]
    fn swapped(&self) -> (usize, usize) {
        self.swapped
    }
}

/// Swap mutation: exchanges two random elements.
#[derive(Debug, Default, Clone, Copy)]
pub struct Swap;

impl Swap {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// # Panics
    ///
    /// Panics if genes has fewer than 2 elements.
    pub fn mutate<T: Clone>(&self, genes: &[T], rng: &mut fastrand::Rng) -> SwapResult<T> {
        assert!(genes.len() >= 2, "swap requires at least 2 genes");

        let a = rng.usize(..genes.len());
        let b = rng.usize(..genes.len() - 1);
        let b = if b >= a { b + 1 } else { b };

        let mut child = genes.to_vec();
        child.swap(a, b);

        SwapResult {
            child,
            swapped: (a, b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swaps_two_elements() {
        let genes = vec![1, 2, 3, 4, 5];
        let mut rng = fastrand::Rng::with_seed(42);

        let result = Swap::new().mutate(&genes, &mut rng);
        let (a, b) = result.swapped();

        assert_eq!(result.child[a], genes[b]);
        assert_eq!(result.child[b], genes[a]);
    }

    #[test]
    fn preserves_length() {
        let genes = vec![1, 2, 3, 4, 5];
        let mut rng = fastrand::Rng::with_seed(42);

        let result = Swap::new().mutate(&genes, &mut rng);

        assert_eq!(result.child.len(), genes.len());
    }

    #[test]
    fn indices_are_different() {
        let genes = vec![1, 2, 3, 4, 5];
        let mut rng = fastrand::Rng::with_seed(42);

        for _ in 0..100 {
            let result = Swap::new().mutate(&genes, &mut rng);
            let (a, b) = result.swapped();
            assert_ne!(a, b);
        }
    }

    #[test]
    #[should_panic(expected = "swap requires at least 2 genes")]
    fn panics_with_single_element() {
        let genes = vec![1];
        let mut rng = fastrand::Rng::with_seed(42);

        Swap::new().mutate(&genes, &mut rng);
    }

    #[test]
    fn preserves_all_elements() {
        let genes = vec![1, 2, 3, 4, 5];
        let mut rng = fastrand::Rng::with_seed(42);

        let result = Swap::new().mutate(&genes, &mut rng);
        let mut sorted = result.into_child();
        sorted.sort();

        assert_eq!(sorted, vec![1, 2, 3, 4, 5]);
    }
}
