use fastrand_contrib::RngExt;

use super::genes::EvolvableGenes;

/// Largest magnitude a number may reach through mutation.
const VALUE_LIMIT: f32 = 10.0;

/// How a child's numbers are nudged after crossover.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mutation {
    /// Adds random noise of size `sigma` to every number.
    Gaussian { sigma: f32 },
}

impl Mutation {
    /// Changes the genes in place.
    pub fn apply<G: EvolvableGenes>(self, genes: &mut G, rng: &mut fastrand::Rng) {
        match self {
            Self::Gaussian { sigma } => {
                for gene_value in genes.gene_groups_mut().flatten() {
                    *gene_value =
                        (*gene_value + rng.f32_normal(0.0, sigma)).clamp(-VALUE_LIMIT, VALUE_LIMIT);
                }
            }
        }
    }
}

impl Default for Mutation {
    fn default() -> Self {
        Self::Gaussian { sigma: 0.01 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::TestGenes;

    /// Mutates one group of `values` with a fresh RNG at `seed`.
    fn mutated(values: Vec<f32>, sigma: f32, seed: u64) -> Vec<f32> {
        let mut genes = TestGenes(vec![values]);
        Mutation::Gaussian { sigma }.apply(&mut genes, &mut fastrand::Rng::with_seed(seed));
        genes.0.remove(0)
    }

    #[test]
    fn keeps_the_shape() {
        let mut genes = TestGenes(vec![vec![1.0, 2.0], vec![3.0, 4.0, 5.0]]);

        Mutation::Gaussian { sigma: 1.0 }.apply(&mut genes, &mut fastrand::Rng::with_seed(42));

        assert_eq!(genes.0.len(), 2);
        assert_eq!(genes.0[0].len(), 2);
        assert_eq!(genes.0[1].len(), 3);
    }

    #[test]
    fn changes_every_group() {
        let mut genes = TestGenes(vec![vec![0.0; 10], vec![0.0; 10]]);

        Mutation::Gaussian { sigma: 1.0 }.apply(&mut genes, &mut fastrand::Rng::with_seed(42));

        for gene_group in &genes.0 {
            assert!(
                gene_group.iter().any(|&gene_value| gene_value != 0.0),
                "group should change"
            );
        }
    }

    #[test]
    fn values_unchanged_with_zero_sigma() {
        let values = vec![0.1, 0.2, 0.3, 0.4];

        let result = mutated(values.clone(), 0.0, 42);

        assert_eq!(result, values);
    }

    #[test]
    fn deterministic_with_same_seed() {
        let result1 = mutated(vec![1.0, 2.0, 3.0, 4.0], 0.1, 42);
        let result2 = mutated(vec![1.0, 2.0, 3.0, 4.0], 0.1, 42);

        assert_eq!(result1, result2);
    }

    #[test]
    fn mutations_centered_around_original() {
        let result = mutated(vec![0.0; 1000], 1.0, 42);

        let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
        assert!(mean.abs() < 0.1, "mean should be close to 0, got {mean}");
    }

    #[test]
    fn values_clamped_to_limit() {
        let result = mutated(vec![9.99, -9.99], 100.0, 42);

        for &gene_value in &result {
            assert!(
                (-VALUE_LIMIT..=VALUE_LIMIT).contains(&gene_value),
                "value {gene_value} is outside the limit"
            );
        }
    }

    #[test]
    fn larger_sigma_produces_larger_changes() {
        let small_sigma = mutated(vec![0.0; 1000], 0.1, 42);
        let large_sigma = mutated(vec![0.0; 1000], 1.0, 42);

        let small_variance: f32 = small_sigma
            .iter()
            .map(|gene_value| gene_value * gene_value)
            .sum::<f32>()
            / small_sigma.len() as f32;
        let large_variance: f32 = large_sigma
            .iter()
            .map(|gene_value| gene_value * gene_value)
            .sum::<f32>()
            / large_sigma.len() as f32;

        assert!(
            large_variance > small_variance,
            "larger sigma should produce larger variance"
        );
    }

    #[test]
    fn default_is_small_gaussian() {
        assert_eq!(Mutation::default(), Mutation::Gaussian { sigma: 0.01 });
    }
}
