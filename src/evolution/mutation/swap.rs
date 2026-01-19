use super::RunMutation;

/// Result of a swap mutation.
struct SwapResult<T> {
    child: Vec<T>,
    swapped: (usize, usize),
}

/// Swap mutation: exchanges two random elements.
#[derive(Debug, Default, Clone, Copy)]
pub struct Swap;

impl Swap {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    fn mutate_internal<T: Clone>(&self, genes: &[T], rng: &mut fastrand::Rng) -> SwapResult<T> {
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

impl RunMutation for Swap {
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

        let internal = Swap.mutate_internal(&genes, &mut fastrand::Rng::with_seed(42));
        let public = Swap.mutate(&genes, &mut fastrand::Rng::with_seed(42));

        assert_eq!(public, internal.child);
    }

    #[test]
    fn swaps_two_elements() {
        let genes = vec![1, 2, 3, 4, 5];
        let mut rng = fastrand::Rng::with_seed(42);

        let result = Swap.mutate_internal(&genes, &mut rng);
        let (a, b) = result.swapped;

        assert_eq!(result.child[a], genes[b]);
        assert_eq!(result.child[b], genes[a]);
    }

    #[test]
    fn preserves_length() {
        let genes = vec![1, 2, 3, 4, 5];
        let mut rng = fastrand::Rng::with_seed(42);

        let result = Swap.mutate_internal(&genes, &mut rng);

        assert_eq!(result.child.len(), genes.len());
    }

    #[test]
    fn indices_are_different() {
        let genes = vec![1, 2, 3, 4, 5];
        let mut rng = fastrand::Rng::with_seed(42);

        for _ in 0..100 {
            let result = Swap.mutate_internal(&genes, &mut rng);
            let (a, b) = result.swapped;
            assert_ne!(a, b);
        }
    }

    #[test]
    #[should_panic(expected = "swap requires at least 2 genes")]
    fn panics_with_single_element() {
        let genes = vec![1];
        let mut rng = fastrand::Rng::with_seed(42);

        Swap.mutate_internal(&genes, &mut rng);
    }

    #[test]
    fn preserves_all_elements() {
        let genes = vec![1, 2, 3, 4, 5];
        let mut rng = fastrand::Rng::with_seed(42);

        let result = Swap.mutate_internal(&genes, &mut rng);
        let mut sorted = result.child;
        sorted.sort();

        assert_eq!(sorted, vec![1, 2, 3, 4, 5]);
    }
}
