mod encoding;
mod select;

pub use encoding::{INPUT_CHANNELS, encode_board};
pub use select::select_best_position;

/// Computes He initialization standard deviation: `√(2/n_in)`.
pub fn he_std(n_in: usize) -> f32 {
    // Safely downcast to u16 (valid up to 65,535), then losslessly convert to f32.
    let n_in_u16 = u16::try_from(n_in).expect("n_in exceeds u16::MAX");
    (2.0 / f32::from(n_in_u16)).sqrt()
}

/// Applies `ReLU` activation in-place: `x = max(0, x)`.
pub fn relu_inplace(data: &mut [f32]) {
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
