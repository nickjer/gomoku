use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use super::RunCrossover;

/// Result of a PMX crossover.
struct PmxResult<T> {
    child: Vec<T>,
    segment: (usize, usize),
}

/// Partially Mapped Crossover (PMX): copies a segment from parent1, then fills
/// remaining positions using a mapping to resolve conflicts.
#[derive(Debug, Default, Clone, Copy)]
pub struct Pmx;

impl Pmx {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    fn crossover_internal<T: Clone + Eq + Hash>(
        &self,
        parent1: &[T],
        parent2: &[T],
        rng: &mut fastrand::Rng,
    ) -> PmxResult<T> {
        let size = parent1.len();

        let a = rng.usize(..size);
        let b = rng.usize(..size);
        let (start, end) = if a <= b { (a, b) } else { (b, a) };

        let mapping: HashMap<_, _> = (start..=end).map(|i| (&parent1[i], &parent2[i])).collect();

        let segment_values: HashSet<_> = parent1[start..=end].iter().collect();

        let child: Vec<T> = (0..size)
            .map(|i| {
                if i >= start && i <= end {
                    parent1[i].clone()
                } else {
                    let mut value = &parent2[i];
                    while segment_values.contains(value) {
                        value = mapping[value];
                    }
                    value.clone()
                }
            })
            .collect();

        PmxResult {
            child,
            segment: (start, end),
        }
    }
}

impl RunCrossover for Pmx {
    fn crossover<T: Clone + Eq + Hash>(
        &self,
        parent1: &[T],
        parent2: &[T],
        rng: &mut fastrand::Rng,
    ) -> Vec<T> {
        self.crossover_internal(parent1, parent2, rng).child
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crossover_returns_child_from_internal() {
        let parent1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let parent2 = vec![8, 7, 6, 5, 4, 3, 2, 1];

        let internal =
            Pmx.crossover_internal(&parent1, &parent2, &mut fastrand::Rng::with_seed(42));
        let public = Pmx.crossover(&parent1, &parent2, &mut fastrand::Rng::with_seed(42));

        assert_eq!(public, internal.child);
    }

    #[test]
    fn child_has_same_size() {
        let parent1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let parent2 = vec![8, 7, 6, 5, 4, 3, 2, 1];
        let mut rng = fastrand::Rng::with_seed(42);

        let result = Pmx.crossover_internal(&parent1, &parent2, &mut rng);

        assert_eq!(result.child.len(), parent1.len());
    }

    #[test]
    fn child_is_valid_permutation() {
        let parent1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let parent2 = vec![8, 7, 6, 5, 4, 3, 2, 1];
        // Seed 13 produces segment (1, 5) - 5 elements
        let mut rng = fastrand::Rng::with_seed(13);

        let result = Pmx.crossover_internal(&parent1, &parent2, &mut rng);
        let mut sorted = result.child;
        sorted.sort();

        assert_eq!(sorted, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn segment_from_parent1_preserved() {
        let parent1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let parent2 = vec![8, 7, 6, 5, 4, 3, 2, 1];
        // Seed 13 produces segment (1, 5) - 5 elements
        let mut rng = fastrand::Rng::with_seed(13);

        let result = Pmx.crossover_internal(&parent1, &parent2, &mut rng);
        let (start, end) = result.segment;

        assert!(end - start >= 2, "segment too small: ({start}, {end})");
        assert_eq!(&result.child[start..=end], &parent1[start..=end]);
    }

    #[test]
    fn positions_without_conflict_use_parent2_directly() {
        let parent1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let parent2 = vec![8, 7, 6, 5, 4, 3, 2, 1];
        // Seed 13 produces segment (1, 5) - 5 elements
        let mut rng = fastrand::Rng::with_seed(13);

        let result = Pmx.crossover_internal(&parent1, &parent2, &mut rng);
        let (start, end) = result.segment;

        assert!(end - start >= 2, "segment too small: ({start}, {end})");

        let segment_values: HashSet<_> = parent1[start..=end].iter().collect();

        let mut tested_count = 0;
        for i in (0..start).chain(end + 1..parent1.len()) {
            if !segment_values.contains(&parent2[i]) {
                assert_eq!(
                    result.child[i], parent2[i],
                    "position {i} should use parent2 directly"
                );
                tested_count += 1;
            }
        }
        assert!(tested_count > 0, "no positions without conflict to test");
    }

    #[test]
    fn start_less_than_or_equal_to_end() {
        let parent1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let parent2 = vec![8, 7, 6, 5, 4, 3, 2, 1];
        let mut rng = fastrand::Rng::with_seed(42);

        for _ in 0..100 {
            let result = Pmx.crossover_internal(&parent1, &parent2, &mut rng);
            let (start, end) = result.segment;
            assert!(start <= end);
        }
    }

    #[test]
    fn works_with_size_one() {
        let parent1 = vec![1];
        let parent2 = vec![1];
        let mut rng = fastrand::Rng::with_seed(42);

        let result = Pmx.crossover_internal(&parent1, &parent2, &mut rng);

        assert_eq!(result.child, vec![1]);
        assert_eq!(result.segment, (0, 0));
    }

    #[test]
    fn deterministic_with_same_seed() {
        let parent1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let parent2 = vec![8, 7, 6, 5, 4, 3, 2, 1];

        let result1 = Pmx.crossover_internal(&parent1, &parent2, &mut fastrand::Rng::with_seed(42));
        let result2 = Pmx.crossover_internal(&parent1, &parent2, &mut fastrand::Rng::with_seed(42));

        assert_eq!(result1.child, result2.child);
    }

    #[test]
    fn different_seeds_produce_different_results() {
        let parent1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let parent2 = vec![8, 7, 6, 5, 4, 3, 2, 1];

        let results: Vec<_> = (0..20)
            .map(|seed| {
                Pmx.crossover_internal(&parent1, &parent2, &mut fastrand::Rng::with_seed(seed))
                    .child
            })
            .collect();

        let unique: HashSet<_> = results.iter().collect();
        assert!(unique.len() > 1);
    }

    #[test]
    fn mapping_resolves_conflicts() {
        let parent1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let parent2 = vec![3, 4, 5, 6, 7, 8, 1, 2];

        for seed in 0..100 {
            let result =
                Pmx.crossover_internal(&parent1, &parent2, &mut fastrand::Rng::with_seed(seed));
            let mut sorted = result.child;
            sorted.sort();
            assert_eq!(
                sorted,
                vec![1, 2, 3, 4, 5, 6, 7, 8],
                "seed {seed} produced invalid permutation"
            );
        }
    }
}
