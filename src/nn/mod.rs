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
