mod board_symmetry;
mod cluster_expansion;
mod layer;
mod neighborhood_encoder;
mod neural_network;
mod neural_network_strategy;
mod no_neighbors;
mod square3x3;
mod stone_channels;

pub use neural_network_strategy::{ClusterSmall, ClusterTiny, ConvSmall, ConvTiny};

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
mod tests {
    use super::*;

    #[test]
    fn format_slice_stats_reports_count_and_spread() {
        assert_eq!(
            format_slice_stats(&[1.0, 3.0]),
            "[2]: min=1.0000, max=3.0000, mean=2.0000, std=1.0000"
        );
    }

    #[test]
    fn format_slice_stats_handles_an_empty_slice() {
        assert_eq!(format_slice_stats(&[]), "[0]: (empty)");
    }
}
