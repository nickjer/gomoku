use super::RunMutation;

/// Result of an inversion mutation.
struct InversionResult<T> {
    child: Vec<T>,
    segment: (usize, usize),
}

/// Inversion mutation: reverses a random segment of the sequence.
#[derive(Debug, Default, Clone, Copy)]
pub struct Inversion;

impl Inversion {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    fn mutate_internal<T: Clone>(
        &self,
        genes: &[T],
        rng: &mut fastrand::Rng,
    ) -> InversionResult<T> {
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

impl RunMutation for Inversion {
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

        let internal = Inversion.mutate_internal(&genes, &mut fastrand::Rng::with_seed(42));
        let public = Inversion.mutate(&genes, &mut fastrand::Rng::with_seed(42));

        assert_eq!(public, internal.child);
    }

    #[test]
    fn reverses_segment_and_preserves_rest() {
        let genes = vec![1, 2, 3, 4, 5];
        let mut rng = fastrand::Rng::with_seed(42);

        for _ in 0..100 {
            let result = Inversion.mutate_internal(&genes, &mut rng);
            let (start, end) = result.segment;

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

        let result = Inversion.mutate_internal(&genes, &mut rng);

        assert_eq!(result.child.len(), genes.len());
    }

    #[test]
    fn start_less_than_end() {
        let genes = vec![1, 2, 3, 4, 5];
        let mut rng = fastrand::Rng::with_seed(42);

        for _ in 0..100 {
            let result = Inversion.mutate_internal(&genes, &mut rng);
            let (start, end) = result.segment;
            assert!(start < end);
        }
    }

    #[test]
    #[should_panic(expected = "inversion requires at least 2 genes")]
    fn panics_with_single_element() {
        let genes = vec![1];
        let mut rng = fastrand::Rng::with_seed(42);

        Inversion.mutate_internal(&genes, &mut rng);
    }

    #[test]
    fn preserves_all_elements() {
        let genes = vec![1, 2, 3, 4, 5];
        let mut rng = fastrand::Rng::with_seed(42);

        let result = Inversion.mutate_internal(&genes, &mut rng);
        let mut sorted = result.child;
        sorted.sort();

        assert_eq!(sorted, vec![1, 2, 3, 4, 5]);
    }
}
