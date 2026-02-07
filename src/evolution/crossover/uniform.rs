/// Performs uniform crossover: for each position, randomly picks from either parent.
///
/// # Panics
///
/// Panics if the parents have different lengths.
#[must_use]
pub fn uniform_crossover<T: Copy>(parent1: &[T], parent2: &[T], rng: &mut fastrand::Rng) -> Vec<T> {
    assert_eq!(
        parent1.len(),
        parent2.len(),
        "parents must have same length"
    );
    parent1
        .iter()
        .zip(parent2)
        .map(|(&val1, &val2)| if rng.bool() { val1 } else { val2 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn child_has_same_length() {
        let parent1 = vec![1.0, 2.0, 3.0, 4.0];
        let parent2 = vec![5.0, 6.0, 7.0, 8.0];
        let mut rng = fastrand::Rng::with_seed(42);

        let child = uniform_crossover(&parent1, &parent2, &mut rng);

        assert_eq!(child.len(), parent1.len());
    }

    #[test]
    fn child_contains_only_parent_values() {
        let parent1 = vec![1.0, 2.0, 3.0, 4.0];
        let parent2 = vec![5.0, 6.0, 7.0, 8.0];
        let mut rng = fastrand::Rng::with_seed(42);

        let child = uniform_crossover(&parent1, &parent2, &mut rng);

        for (i, &val) in child.iter().enumerate() {
            assert!(val == parent1[i] || val == parent2[i]);
        }
    }

    #[test]
    fn deterministic_with_same_seed() {
        let parent1 = vec![1.0, 2.0, 3.0, 4.0];
        let parent2 = vec![5.0, 6.0, 7.0, 8.0];

        let child1 = uniform_crossover(&parent1, &parent2, &mut fastrand::Rng::with_seed(42));
        let child2 = uniform_crossover(&parent1, &parent2, &mut fastrand::Rng::with_seed(42));

        assert_eq!(child1, child2);
    }

    #[test]
    fn mixes_both_parents() {
        let parent1 = vec![1.0_f32; 100];
        let parent2 = vec![2.0_f32; 100];
        let mut rng = fastrand::Rng::with_seed(42);

        let child = uniform_crossover(&parent1, &parent2, &mut rng);

        let from_parent1 = child.iter().filter(|&&v| v == 1.0).count();
        let from_parent2 = child.iter().filter(|&&v| v == 2.0).count();

        assert!(from_parent1 > 0, "should have some values from parent1");
        assert!(from_parent2 > 0, "should have some values from parent2");
    }

    #[test]
    fn works_with_integers() {
        let parent1 = vec![1, 2, 3, 4];
        let parent2 = vec![5, 6, 7, 8];
        let mut rng = fastrand::Rng::with_seed(42);

        let child = uniform_crossover(&parent1, &parent2, &mut rng);

        assert_eq!(child.len(), 4);
        for (i, &val) in child.iter().enumerate() {
            assert!(val == parent1[i] || val == parent2[i]);
        }
    }

    #[test]
    #[should_panic(expected = "parents must have same length")]
    fn panics_with_different_lengths() {
        let parent1 = vec![1.0, 2.0, 3.0];
        let parent2 = vec![4.0, 5.0];
        let mut rng = fastrand::Rng::with_seed(42);

        let _ = uniform_crossover(&parent1, &parent2, &mut rng);
    }
}
