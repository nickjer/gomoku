use fastrand_contrib::RngExt;

/// Performs Gaussian mutation: adds N(0, sigma) noise to all weights.
#[must_use]
pub fn gaussian_mutate(weights: &[f32], sigma: f32, rng: &mut fastrand::Rng) -> Vec<f32> {
    weights
        .iter()
        .map(|&weight| (weight + rng.f32_normal(0.0, sigma)).clamp(-10.0, 10.0))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn result_has_same_length() {
        let weights = vec![1.0, 2.0, 3.0, 4.0];
        let mut rng = fastrand::Rng::with_seed(42);

        let result = gaussian_mutate(&weights, 0.1, &mut rng);

        assert_eq!(result.len(), weights.len());
    }

    #[test]
    fn values_change_with_nonzero_sigma() {
        let weights = vec![0.0; 100];
        let mut rng = fastrand::Rng::with_seed(42);

        let result = gaussian_mutate(&weights, 1.0, &mut rng);

        let changed = result.iter().filter(|&&val| val != 0.0).count();
        assert!(changed > 0, "some values should change");
    }

    #[test]
    fn values_unchanged_with_zero_sigma() {
        let weights = vec![0.1, 0.2, 0.3, 0.4];
        let mut rng = fastrand::Rng::with_seed(42);

        let result = gaussian_mutate(&weights, 0.0, &mut rng);

        assert_eq!(result, weights);
    }

    #[test]
    fn deterministic_with_same_seed() {
        let weights = vec![1.0, 2.0, 3.0, 4.0];

        let result1 = gaussian_mutate(&weights, 0.1, &mut fastrand::Rng::with_seed(42));
        let result2 = gaussian_mutate(&weights, 0.1, &mut fastrand::Rng::with_seed(42));

        assert_eq!(result1, result2);
    }

    #[test]
    fn mutations_centered_around_original() {
        let weights = vec![0.0; 1000];
        let mut rng = fastrand::Rng::with_seed(42);

        let result = gaussian_mutate(&weights, 1.0, &mut rng);

        let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
        assert!(mean.abs() < 0.1, "mean should be close to 0, got {mean}");
    }

    #[test]
    fn values_clamped_to_range() {
        let weights = vec![9.99, -9.99];
        let mut rng = fastrand::Rng::with_seed(42);

        let result = gaussian_mutate(&weights, 100.0, &mut rng);

        for &value in &result {
            assert!(
                (-10.0..=10.0).contains(&value),
                "value {value} is outside [-10, 10]"
            );
        }
    }

    #[test]
    fn larger_sigma_produces_larger_changes() {
        let weights = vec![0.0; 1000];

        let small_sigma = gaussian_mutate(&weights, 0.1, &mut fastrand::Rng::with_seed(42));
        let large_sigma = gaussian_mutate(&weights, 1.0, &mut fastrand::Rng::with_seed(42));

        let small_variance: f32 =
            small_sigma.iter().map(|val| val * val).sum::<f32>() / small_sigma.len() as f32;
        let large_variance: f32 =
            large_sigma.iter().map(|val| val * val).sum::<f32>() / large_sigma.len() as f32;

        assert!(
            large_variance > small_variance,
            "larger sigma should produce larger variance"
        );
    }
}
