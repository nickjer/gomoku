use fixedbitset::FixedBitSet;

use crate::gene::Gene;

/// Performs order crossover (OX) on two parent gene sequences.
///
/// Copies a random segment from parent1, then fills remaining positions
/// with elements from parent2 in order, skipping those already in the segment.
#[must_use]
pub fn order_crossover(parent1: &[Gene], parent2: &[Gene], rng: &mut fastrand::Rng) -> Vec<Gene> {
    crossover_internal(parent1, parent2, rng).0
}

fn crossover_internal(
    parent1: &[Gene],
    parent2: &[Gene],
    rng: &mut fastrand::Rng,
) -> (Vec<Gene>, (usize, usize)) {
    let size = parent1.len();

    let a = rng.usize(..size);
    let b = rng.usize(..size);
    let (start, end) = if a <= b { (a, b) } else { (b, a) };

    let mut segment = FixedBitSet::with_capacity(size);
    for &gene in &parent1[start..=end] {
        segment.insert(gene.index());
    }

    let mut remaining = parent2.iter().filter(|g| !segment.contains(g.index()));

    let child: Vec<Gene> = (0..size)
        .map(|i| {
            if i >= start && i <= end {
                parent1[i]
            } else {
                *remaining.next().expect("parent2 missing elements")
            }
        })
        .collect();

    (child, (start, end))
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    fn genes(values: &[usize]) -> Vec<Gene> {
        values.iter().copied().map(Gene::new).collect()
    }

    #[test]
    fn child_has_same_size() {
        let parent1 = genes(&[0, 1, 2, 3, 4, 5, 6, 7]);
        let parent2 = genes(&[7, 6, 5, 4, 3, 2, 1, 0]);
        let mut rng = fastrand::Rng::with_seed(42);

        let (child, _) = crossover_internal(&parent1, &parent2, &mut rng);

        assert_eq!(child.len(), parent1.len());
    }

    #[test]
    fn child_is_valid_permutation() {
        let parent1 = genes(&[0, 1, 2, 3, 4, 5, 6, 7]);
        let parent2 = genes(&[7, 6, 5, 4, 3, 2, 1, 0]);
        // Seed 13 produces segment (1, 5) - 5 elements
        let mut rng = fastrand::Rng::with_seed(13);

        let (mut child, _) = crossover_internal(&parent1, &parent2, &mut rng);
        child.sort();

        assert_eq!(child, genes(&[0, 1, 2, 3, 4, 5, 6, 7]));
    }

    #[test]
    fn segment_from_parent1_preserved() {
        let parent1 = genes(&[0, 1, 2, 3, 4, 5, 6, 7]);
        let parent2 = genes(&[7, 6, 5, 4, 3, 2, 1, 0]);
        // Seed 13 produces segment (1, 5) - 5 elements
        let mut rng = fastrand::Rng::with_seed(13);

        let (child, (start, end)) = crossover_internal(&parent1, &parent2, &mut rng);

        assert!(end - start >= 2, "segment too small: ({start}, {end})");
        assert_eq!(&child[start..=end], &parent1[start..=end]);
    }

    #[test]
    fn maintains_parent2_order_outside_segment() {
        let parent1 = genes(&[0, 1, 2, 3, 4, 5, 6, 7]);
        let parent2 = genes(&[7, 6, 5, 4, 3, 2, 1, 0]);
        // Seed 13 produces segment (1, 5) - leaves positions 0, 6, 7 outside
        let mut rng = fastrand::Rng::with_seed(13);

        let (child, (start, end)) = crossover_internal(&parent1, &parent2, &mut rng);

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
        let parent1 = genes(&[0, 1, 2, 3, 4, 5, 6, 7]);
        let parent2 = genes(&[7, 6, 5, 4, 3, 2, 1, 0]);
        let mut rng = fastrand::Rng::with_seed(42);

        for _ in 0..100 {
            let (_, (start, end)) = crossover_internal(&parent1, &parent2, &mut rng);
            assert!(start <= end);
        }
    }

    #[test]
    fn works_with_size_one() {
        let parent1 = genes(&[0]);
        let parent2 = genes(&[0]);
        let mut rng = fastrand::Rng::with_seed(42);

        let (child, segment) = crossover_internal(&parent1, &parent2, &mut rng);

        assert_eq!(child, genes(&[0]));
        assert_eq!(segment, (0, 0));
    }

    #[test]
    fn deterministic_with_same_seed() {
        let parent1 = genes(&[0, 1, 2, 3, 4, 5, 6, 7]);
        let parent2 = genes(&[7, 6, 5, 4, 3, 2, 1, 0]);

        let (child1, _) = crossover_internal(&parent1, &parent2, &mut fastrand::Rng::with_seed(42));
        let (child2, _) = crossover_internal(&parent1, &parent2, &mut fastrand::Rng::with_seed(42));

        assert_eq!(child1, child2);
    }

    #[test]
    fn different_seeds_produce_different_results() {
        let parent1 = genes(&[0, 1, 2, 3, 4, 5, 6, 7]);
        let parent2 = genes(&[7, 6, 5, 4, 3, 2, 1, 0]);

        let results: Vec<_> = (0..20)
            .map(|seed| {
                crossover_internal(&parent1, &parent2, &mut fastrand::Rng::with_seed(seed)).0
            })
            .collect();

        let unique: HashSet<_> = results.iter().collect();
        assert!(unique.len() > 1);
    }
}
