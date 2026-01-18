/// Result of an inversion mutation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InversionResult<T> {
    child: Vec<T>,
    segment: (usize, usize),
}

impl<T> InversionResult<T> {
    #[must_use]
    pub fn into_child(self) -> Vec<T> {
        self.child
    }

    /// Returns (start, end) indices of the inverted segment (both inclusive).
    #[cfg(test)]
    fn segment(&self) -> (usize, usize) {
        self.segment
    }
}

/// Inversion mutation: reverses a random segment of the sequence.
#[derive(Debug, Default, Clone, Copy)]
pub struct Inversion;

impl Inversion {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// # Panics
    ///
    /// Panics if genes has fewer than 2 elements.
    pub fn mutate<T: Clone>(&self, genes: &[T], rng: &mut fastrand::Rng) -> InversionResult<T> {
        assert!(genes.len() >= 2, "inversion requires at least 2 genes");

        let a = rng.usize(..genes.len());
        let b = rng.usize(..genes.len() - 1);
        let b = if b >= a { b + 1 } else { b };

        let (start, end) = if a < b { (a, b) } else { (b, a) };

        let mut child = genes.to_vec();
        child[start..=end].reverse();

        InversionResult {
            child,
            segment: (start, end),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reverses_segment_and_preserves_rest() {
        let genes = vec![1, 2, 3, 4, 5];
        let mut rng = fastrand::Rng::with_seed(42);

        for _ in 0..100 {
            let result = Inversion::new().mutate(&genes, &mut rng);
            let (start, end) = result.segment();

            // Elements before segment unchanged
            for i in 0..start {
                assert_eq!(result.child[i], genes[i]);
            }

            // Segment is reversed
            for i in start..=end {
                assert_eq!(result.child[i], genes[end - (i - start)]);
            }

            // Elements after segment unchanged
            for i in (end + 1)..genes.len() {
                assert_eq!(result.child[i], genes[i]);
            }
        }
    }

    #[test]
    fn preserves_length() {
        let genes = vec![1, 2, 3, 4, 5];
        let mut rng = fastrand::Rng::with_seed(42);

        let result = Inversion::new().mutate(&genes, &mut rng);

        assert_eq!(result.child.len(), genes.len());
    }

    #[test]
    fn start_less_than_end() {
        let genes = vec![1, 2, 3, 4, 5];
        let mut rng = fastrand::Rng::with_seed(42);

        for _ in 0..100 {
            let result = Inversion::new().mutate(&genes, &mut rng);
            let (start, end) = result.segment();
            assert!(start < end);
        }
    }

    #[test]
    #[should_panic(expected = "inversion requires at least 2 genes")]
    fn panics_with_single_element() {
        let genes = vec![1];
        let mut rng = fastrand::Rng::with_seed(42);

        Inversion::new().mutate(&genes, &mut rng);
    }

    #[test]
    fn preserves_all_elements() {
        let genes = vec![1, 2, 3, 4, 5];
        let mut rng = fastrand::Rng::with_seed(42);

        let result = Inversion::new().mutate(&genes, &mut rng);
        let mut sorted = result.into_child();
        sorted.sort();

        assert_eq!(sorted, vec![1, 2, 3, 4, 5]);
    }
}
