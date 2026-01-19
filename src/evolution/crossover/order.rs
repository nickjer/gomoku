use std::collections::HashSet;
use std::hash::Hash;

use super::RunCrossover;

/// Order crossover (OX): copies a segment from parent1, fills remaining positions
/// with elements from parent2 in order, skipping those already in the segment.
#[derive(Debug, Default, Clone, Copy)]
pub struct Order;

impl Order {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    fn crossover_internal<T: Clone + Eq + Hash>(
        &self,
        parent1: &[T],
        parent2: &[T],
        rng: &mut fastrand::Rng,
    ) -> (Vec<T>, (usize, usize)) {
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

        (child, (start, end))
    }
}

impl RunCrossover for Order {
    fn crossover<T: Clone + Eq + Hash>(
        &self,
        parent1: &[T],
        parent2: &[T],
        rng: &mut fastrand::Rng,
    ) -> Vec<T> {
        self.crossover_internal(parent1, parent2, rng).0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crossover_returns_child_from_internal() {
        let parent1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let parent2 = vec![8, 7, 6, 5, 4, 3, 2, 1];

        let (child, _) =
            Order.crossover_internal(&parent1, &parent2, &mut fastrand::Rng::with_seed(42));
        let public = Order.crossover(&parent1, &parent2, &mut fastrand::Rng::with_seed(42));

        assert_eq!(public, child);
    }

    #[test]
    fn child_has_same_size() {
        let parent1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let parent2 = vec![8, 7, 6, 5, 4, 3, 2, 1];
        let mut rng = fastrand::Rng::with_seed(42);

        let (child, _) = Order.crossover_internal(&parent1, &parent2, &mut rng);

        assert_eq!(child.len(), parent1.len());
    }

    #[test]
    fn child_is_valid_permutation() {
        let parent1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let parent2 = vec![8, 7, 6, 5, 4, 3, 2, 1];
        // Seed 13 produces segment (1, 5) - 5 elements
        let mut rng = fastrand::Rng::with_seed(13);

        let (mut child, _) = Order.crossover_internal(&parent1, &parent2, &mut rng);
        child.sort();

        assert_eq!(child, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn segment_from_parent1_preserved() {
        let parent1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let parent2 = vec![8, 7, 6, 5, 4, 3, 2, 1];
        // Seed 13 produces segment (1, 5) - 5 elements
        let mut rng = fastrand::Rng::with_seed(13);

        let (child, (start, end)) = Order.crossover_internal(&parent1, &parent2, &mut rng);

        assert!(end - start >= 2, "segment too small: ({start}, {end})");
        assert_eq!(&child[start..=end], &parent1[start..=end]);
    }

    #[test]
    fn maintains_parent2_order_outside_segment() {
        let parent1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let parent2 = vec![8, 7, 6, 5, 4, 3, 2, 1];
        // Seed 13 produces segment (1, 5) - leaves positions 0, 6, 7 outside
        let mut rng = fastrand::Rng::with_seed(13);

        let (child, (start, end)) = Order.crossover_internal(&parent1, &parent2, &mut rng);

        assert!(end - start >= 2, "segment too small: ({start}, {end})");

        let segment_values: HashSet<_> = parent1[start..=end].iter().collect();

        let child_outside: Vec<_> = child
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
            let (_, (start, end)) = Order.crossover_internal(&parent1, &parent2, &mut rng);
            assert!(start <= end);
        }
    }

    #[test]
    fn works_with_size_one() {
        let parent1 = vec![1];
        let parent2 = vec![1];
        let mut rng = fastrand::Rng::with_seed(42);

        let (child, segment) = Order.crossover_internal(&parent1, &parent2, &mut rng);

        assert_eq!(child, vec![1]);
        assert_eq!(segment, (0, 0));
    }

    #[test]
    fn deterministic_with_same_seed() {
        let parent1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let parent2 = vec![8, 7, 6, 5, 4, 3, 2, 1];

        let (child1, _) =
            Order.crossover_internal(&parent1, &parent2, &mut fastrand::Rng::with_seed(42));
        let (child2, _) =
            Order.crossover_internal(&parent1, &parent2, &mut fastrand::Rng::with_seed(42));

        assert_eq!(child1, child2);
    }

    #[test]
    fn different_seeds_produce_different_results() {
        let parent1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let parent2 = vec![8, 7, 6, 5, 4, 3, 2, 1];

        let results: Vec<_> = (0..20)
            .map(|seed| {
                Order
                    .crossover_internal(&parent1, &parent2, &mut fastrand::Rng::with_seed(seed))
                    .0
            })
            .collect();

        let unique: HashSet<_> = results.iter().collect();
        assert!(unique.len() > 1);
    }
}
