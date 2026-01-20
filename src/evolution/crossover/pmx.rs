use std::hash::Hash;

use rapidhash::{RapidHashMap, RapidHashSet};

use super::RunCrossover;

/// Partially Mapped Crossover (PMX): copies a segment from parent1, then fills
/// remaining positions using a mapping to resolve conflicts.
#[derive(Debug, Default, Clone, Copy)]
pub struct Pmx;

impl Pmx {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    #[allow(clippy::unused_self)]
    fn crossover_internal<T: Clone + Eq + Hash>(
        self,
        parent1: &[T],
        parent2: &[T],
        rng: &mut fastrand::Rng,
    ) -> (Vec<T>, (usize, usize)) {
        let size = parent1.len();

        let a = rng.usize(..size);
        let b = rng.usize(..size);
        let (start, end) = if a <= b { (a, b) } else { (b, a) };

        let mapping: RapidHashMap<_, _> =
            (start..=end).map(|i| (&parent1[i], &parent2[i])).collect();

        let segment_values: RapidHashSet<_> = parent1[start..=end].iter().collect();

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

        (child, (start, end))
    }
}

impl RunCrossover for Pmx {
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
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn crossover_returns_child_from_internal() {
        let parent1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let parent2 = vec![8, 7, 6, 5, 4, 3, 2, 1];

        let (child, _) =
            Pmx.crossover_internal(&parent1, &parent2, &mut fastrand::Rng::with_seed(42));
        let public = Pmx.crossover(&parent1, &parent2, &mut fastrand::Rng::with_seed(42));

        assert_eq!(public, child);
    }

    #[test]
    fn child_has_same_size() {
        let parent1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let parent2 = vec![8, 7, 6, 5, 4, 3, 2, 1];
        let mut rng = fastrand::Rng::with_seed(42);

        let (child, _) = Pmx.crossover_internal(&parent1, &parent2, &mut rng);

        assert_eq!(child.len(), parent1.len());
    }

    #[test]
    fn child_is_valid_permutation() {
        let parent1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let parent2 = vec![8, 7, 6, 5, 4, 3, 2, 1];
        // Seed 13 produces segment (1, 5) - 5 elements
        let mut rng = fastrand::Rng::with_seed(13);

        let (mut child, _) = Pmx.crossover_internal(&parent1, &parent2, &mut rng);
        child.sort();

        assert_eq!(child, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn segment_from_parent1_preserved() {
        let parent1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let parent2 = vec![8, 7, 6, 5, 4, 3, 2, 1];
        // Seed 13 produces segment (1, 5) - 5 elements
        let mut rng = fastrand::Rng::with_seed(13);

        let (child, (start, end)) = Pmx.crossover_internal(&parent1, &parent2, &mut rng);

        assert!(end - start >= 2, "segment too small: ({start}, {end})");
        assert_eq!(&child[start..=end], &parent1[start..=end]);
    }

    #[test]
    fn positions_without_conflict_use_parent2_directly() {
        let parent1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let parent2 = vec![8, 7, 6, 5, 4, 3, 2, 1];
        // Seed 13 produces segment (1, 5) - 5 elements
        let mut rng = fastrand::Rng::with_seed(13);

        let (child, (start, end)) = Pmx.crossover_internal(&parent1, &parent2, &mut rng);

        assert!(end - start >= 2, "segment too small: ({start}, {end})");

        let segment_values: HashSet<_> = parent1[start..=end].iter().collect();

        let mut tested_count = 0;
        for i in (0..start).chain(end + 1..parent1.len()) {
            if !segment_values.contains(&parent2[i]) {
                assert_eq!(
                    child[i], parent2[i],
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
            let (_, (start, end)) = Pmx.crossover_internal(&parent1, &parent2, &mut rng);
            assert!(start <= end);
        }
    }

    #[test]
    fn works_with_size_one() {
        let parent1 = vec![1];
        let parent2 = vec![1];
        let mut rng = fastrand::Rng::with_seed(42);

        let (child, segment) = Pmx.crossover_internal(&parent1, &parent2, &mut rng);

        assert_eq!(child, vec![1]);
        assert_eq!(segment, (0, 0));
    }

    #[test]
    fn deterministic_with_same_seed() {
        let parent1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let parent2 = vec![8, 7, 6, 5, 4, 3, 2, 1];

        let (child1, _) =
            Pmx.crossover_internal(&parent1, &parent2, &mut fastrand::Rng::with_seed(42));
        let (child2, _) =
            Pmx.crossover_internal(&parent1, &parent2, &mut fastrand::Rng::with_seed(42));

        assert_eq!(child1, child2);
    }

    #[test]
    fn different_seeds_produce_different_results() {
        let parent1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let parent2 = vec![8, 7, 6, 5, 4, 3, 2, 1];

        let results: Vec<_> = (0..20)
            .map(|seed| {
                Pmx.crossover_internal(&parent1, &parent2, &mut fastrand::Rng::with_seed(seed))
                    .0
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
            let (mut child, _) =
                Pmx.crossover_internal(&parent1, &parent2, &mut fastrand::Rng::with_seed(seed));
            child.sort();
            assert_eq!(
                child,
                vec![1, 2, 3, 4, 5, 6, 7, 8],
                "seed {seed} produced invalid permutation"
            );
        }
    }
}
