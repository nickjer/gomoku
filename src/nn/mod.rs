mod board_symmetry;
mod cluster_expansion;
mod position_selection;
mod stone_channels;

pub use board_symmetry::BoardSymmetry;
pub use cluster_expansion::{CLUSTER_COUNT, CLUSTER_NAMES, NEIGHBOR_OFFSETS, cluster_sums};
pub use position_selection::highest_scored_empty_position;
pub use stone_channels::{STONE_CHANNELS, board_to_stone_channels};

/// Standard deviation for random initial weights: `√(2 / inputs_per_output)`.
///
/// This is He initialization, which keeps activations from vanishing or
/// exploding through layers that zero their negatives.
pub fn initial_weight_spread(inputs_per_output: usize) -> f32 {
    // Safely downcast to u16 (valid up to 65,535), then losslessly convert to f32.
    let inputs_u16 = u16::try_from(inputs_per_output).expect("inputs_per_output exceeds u16::MAX");
    (2.0 / f32::from(inputs_u16)).sqrt()
}

/// Replaces every negative value with zero, in place (the `ReLU` activation).
pub fn zero_negatives_inplace(data: &mut [f32]) {
    for val in data.iter_mut() {
        *val = val.max(0.0);
    }
}

/// Formats summary statistics for a slice of f32 values.
///
/// Output: `[count]: min=X, max=X, mean=X, std=X`
pub fn format_slice_stats(values: &[f32]) -> String {
    let count = values.len();
    if count == 0 {
        return "[0]: (empty)".to_string();
    }

    let min = values.iter().copied().reduce(f32::min).unwrap_or(0.0);
    let max = values.iter().copied().reduce(f32::max).unwrap_or(0.0);
    let sum: f32 = values.iter().sum();
    let count_f32 = f32::from(u16::try_from(count).expect("slice length exceeds u16::MAX"));
    let mean = sum / count_f32;
    let variance: f32 = values.iter().map(|&val| (val - mean).powi(2)).sum::<f32>() / count_f32;
    let std = variance.sqrt();

    format!("[{count}]: min={min:.4}, max={max:.4}, mean={mean:.4}, std={std:.4}")
}

#[cfg(test)]
mod zero_negatives_tests {
    use super::*;

    #[test]
    fn positive_values_unchanged() {
        let mut data = vec![1.0, 2.5, 0.001];
        zero_negatives_inplace(&mut data);
        assert_eq!(data, vec![1.0, 2.5, 0.001]);
    }

    #[test]
    fn negative_values_become_zero() {
        let mut data = vec![-1.0, -0.001, -100.0];
        zero_negatives_inplace(&mut data);
        assert_eq!(data, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn zero_unchanged() {
        let mut data = vec![0.0];
        zero_negatives_inplace(&mut data);
        assert_eq!(data, vec![0.0]);
    }

    #[test]
    fn mixed_values() {
        let mut data = vec![-2.0, 0.0, 3.0, -0.5, 1.0];
        zero_negatives_inplace(&mut data);
        assert_eq!(data, vec![0.0, 0.0, 3.0, 0.0, 1.0]);
    }

    #[test]
    fn empty_slice() {
        let mut data: Vec<f32> = vec![];
        zero_negatives_inplace(&mut data);
        assert!(data.is_empty());
    }
}
