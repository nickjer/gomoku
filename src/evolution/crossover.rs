use super::genes::EvolvableGenes;

/// How two parents are combined into a child.
#[derive(Debug, Default, Clone, Copy)]
pub enum Crossover {
    /// Each number is taken from one parent or the other at random.
    #[default]
    Uniform,
}

impl Crossover {
    /// Builds a child from two parents of the same shape.
    ///
    /// # Panics
    ///
    /// Panics if the parents have different shapes.
    #[must_use]
    pub fn apply<G: EvolvableGenes>(self, parent1: &G, parent2: &G, rng: &mut fastrand::Rng) -> G {
        assert_eq!(
            parent1.gene_groups().count(),
            parent2.gene_groups().count(),
            "parents must have the same shape"
        );

        let mut child = parent1.clone();
        for (child_group, parent2_group) in child.gene_groups_mut().zip(parent2.gene_groups()) {
            assert_eq!(
                child_group.len(),
                parent2_group.len(),
                "parents must have the same shape"
            );
            match self {
                Self::Uniform => {
                    for (child_value, &parent2_value) in child_group.iter_mut().zip(parent2_group) {
                        *child_value = if rng.bool() {
                            *child_value
                        } else {
                            parent2_value
                        };
                    }
                }
            }
        }
        child
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::TestGenes;

    #[test]
    fn child_has_parent_shape() {
        let parent1 = TestGenes(vec![vec![1.0, 2.0], vec![3.0, 4.0, 5.0]]);
        let parent2 = TestGenes(vec![vec![6.0, 7.0], vec![8.0, 9.0, 10.0]]);
        let mut rng = fastrand::Rng::with_seed(42);

        let child = Crossover::Uniform.apply(&parent1, &parent2, &mut rng);

        assert_eq!(child.0.len(), 2);
        assert_eq!(child.0[0].len(), 2);
        assert_eq!(child.0[1].len(), 3);
    }

    #[test]
    fn child_contains_only_parent_values() {
        let parent1 = TestGenes(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let parent2 = TestGenes(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
        let mut rng = fastrand::Rng::with_seed(42);

        let child = Crossover::Uniform.apply(&parent1, &parent2, &mut rng);

        for (group_idx, child_group) in child.0.iter().enumerate() {
            for (value_idx, &child_value) in child_group.iter().enumerate() {
                assert!(
                    child_value == parent1.0[group_idx][value_idx]
                        || child_value == parent2.0[group_idx][value_idx]
                );
            }
        }
    }

    #[test]
    fn deterministic_with_same_seed() {
        let parent1 = TestGenes(vec![vec![1.0, 2.0, 3.0, 4.0]]);
        let parent2 = TestGenes(vec![vec![5.0, 6.0, 7.0, 8.0]]);

        let child1 =
            Crossover::Uniform.apply(&parent1, &parent2, &mut fastrand::Rng::with_seed(42));
        let child2 =
            Crossover::Uniform.apply(&parent1, &parent2, &mut fastrand::Rng::with_seed(42));

        assert_eq!(child1, child2);
    }

    #[test]
    fn mixes_both_parents() {
        let parent1 = TestGenes(vec![vec![1.0; 100]]);
        let parent2 = TestGenes(vec![vec![2.0; 100]]);
        let mut rng = fastrand::Rng::with_seed(42);

        let child = Crossover::Uniform.apply(&parent1, &parent2, &mut rng);

        let from_parent1 = child.0[0].iter().filter(|&&value| value == 1.0).count();
        let from_parent2 = child.0[0].iter().filter(|&&value| value == 2.0).count();
        assert!(from_parent1 > 0, "should have some values from parent1");
        assert!(from_parent2 > 0, "should have some values from parent2");
    }

    #[test]
    #[should_panic(expected = "parents must have the same shape")]
    fn panics_when_group_counts_differ() {
        let parent1 = TestGenes(vec![vec![1.0], vec![2.0]]);
        let parent2 = TestGenes(vec![vec![3.0]]);
        let mut rng = fastrand::Rng::with_seed(42);

        let _ = Crossover::Uniform.apply(&parent1, &parent2, &mut rng);
    }

    #[test]
    #[should_panic(expected = "parents must have the same shape")]
    fn panics_when_group_lengths_differ() {
        let parent1 = TestGenes(vec![vec![1.0, 2.0, 3.0]]);
        let parent2 = TestGenes(vec![vec![4.0, 5.0]]);
        let mut rng = fastrand::Rng::with_seed(42);

        let _ = Crossover::Uniform.apply(&parent1, &parent2, &mut rng);
    }
}
