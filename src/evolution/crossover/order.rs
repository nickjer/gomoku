use std::collections::HashSet;
use std::hash::Hash;

/// Result of an order crossover.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrderResult<T> {
    child: Vec<T>,
    segment: (usize, usize),
}

impl<T> OrderResult<T> {
    #[must_use]
    pub fn into_child(self) -> Vec<T> {
        self.child
    }

    /// Returns (start, end) indices of the copied segment (both inclusive).
    #[cfg(test)]
    fn segment(&self) -> (usize, usize) {
        self.segment
    }
}

/// Order crossover (OX): copies a segment from parent1, fills remaining positions
/// with elements from parent2 in order, skipping those already in the segment.
#[derive(Debug, Default, Clone, Copy)]
pub struct Order;

impl Order {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// # Panics
    ///
    /// Panics if parent2 doesn't contain all elements from parent1.
    pub fn crossover<T: Clone + Eq + Hash>(
        &self,
        parent1: &[T],
        parent2: &[T],
        rng: &mut fastrand::Rng,
    ) -> OrderResult<T> {
        let size = parent1.len();

        let a = rng.usize(..size);
        let b = rng.usize(..size);
        let (start, end) = if a <= b { (a, b) } else { (b, a) };

        let segment: HashSet<_> = parent1[start..=end].iter().collect();
        let mut remaining = parent2.iter().filter(|x| !segment.contains(x));

        let child: Vec<T> = (0..size)
            .map(|i| {
                if i >= start && i <= end {
                    parent1[i].clone()
                } else {
                    remaining.next().expect("parent2 missing elements").clone()
                }
            })
            .collect();

        OrderResult {
            child,
            segment: (start, end),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn child_has_same_size() {
        let parent1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let parent2 = vec![8, 7, 6, 5, 4, 3, 2, 1];
        let mut rng = fastrand::Rng::with_seed(42);

        let result = Order::new().crossover(&parent1, &parent2, &mut rng);

        assert_eq!(result.child.len(), parent1.len());
    }

    #[test]
    fn child_is_valid_permutation() {
        let parent1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let parent2 = vec![8, 7, 6, 5, 4, 3, 2, 1];
        // Seed 13 produces segment (1, 5) - 5 elements
        let mut rng = fastrand::Rng::with_seed(13);

        let result = Order::new().crossover(&parent1, &parent2, &mut rng);
        let mut sorted = result.into_child();
        sorted.sort();

        assert_eq!(sorted, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn segment_from_parent1_preserved() {
        let parent1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let parent2 = vec![8, 7, 6, 5, 4, 3, 2, 1];
        // Seed 13 produces segment (1, 5) - 5 elements
        let mut rng = fastrand::Rng::with_seed(13);

        let result = Order::new().crossover(&parent1, &parent2, &mut rng);
        let (start, end) = result.segment();

        assert!(end - start >= 2, "segment too small: ({start}, {end})");
        assert_eq!(&result.child[start..=end], &parent1[start..=end]);
    }

    #[test]
    fn maintains_parent2_order_outside_segment() {
        let parent1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let parent2 = vec![8, 7, 6, 5, 4, 3, 2, 1];
        // Seed 13 produces segment (1, 5) - leaves positions 0, 6, 7 outside
        let mut rng = fastrand::Rng::with_seed(13);

        let result = Order::new().crossover(&parent1, &parent2, &mut rng);
        let (start, end) = result.segment();

        assert!(end - start >= 2, "segment too small: ({start}, {end})");

        let segment_values: HashSet<_> = parent1[start..=end].iter().collect();

        let child_outside: Vec<_> = result
            .child
            .iter()
            .enumerate()
            .filter(|(i, _)| *i < start || *i > end)
            .map(|(_, v)| v)
            .collect();

        let parent2_filtered: Vec<_> = parent2
            .iter()
            .filter(|x| !segment_values.contains(x))
            .collect();

        assert_eq!(child_outside, parent2_filtered);
    }

    #[test]
    fn start_less_than_or_equal_to_end() {
        let parent1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let parent2 = vec![8, 7, 6, 5, 4, 3, 2, 1];
        let mut rng = fastrand::Rng::with_seed(42);

        for _ in 0..100 {
            let result = Order::new().crossover(&parent1, &parent2, &mut rng);
            let (start, end) = result.segment();
            assert!(start <= end);
        }
    }

    #[test]
    fn works_with_size_one() {
        let parent1 = vec![1];
        let parent2 = vec![1];
        let mut rng = fastrand::Rng::with_seed(42);

        let result = Order::new().crossover(&parent1, &parent2, &mut rng);

        assert_eq!(result.child, vec![1]);
        assert_eq!(result.segment(), (0, 0));
    }

    #[test]
    fn deterministic_with_same_seed() {
        let parent1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let parent2 = vec![8, 7, 6, 5, 4, 3, 2, 1];

        let result1 = Order::new().crossover(&parent1, &parent2, &mut fastrand::Rng::with_seed(42));
        let result2 = Order::new().crossover(&parent1, &parent2, &mut fastrand::Rng::with_seed(42));

        assert_eq!(result1.child, result2.child);
    }

    #[test]
    fn different_seeds_produce_different_results() {
        let parent1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let parent2 = vec![8, 7, 6, 5, 4, 3, 2, 1];

        let results: Vec<_> = (0..20)
            .map(|seed| {
                Order::new()
                    .crossover(&parent1, &parent2, &mut fastrand::Rng::with_seed(seed))
                    .into_child()
            })
            .collect();

        let unique: HashSet<_> = results.iter().collect();
        assert!(unique.len() > 1);
    }
}
