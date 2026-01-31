use crate::offset::Offset;
use crate::position_id::PositionId;
use crate::position_map::{PositionSlice, PositionSliceMut};

/// 2D convolution with same-padding (zero-padding, output size = input size).
///
/// # Layout
/// - `input`: `[IN_C * PositionId::COUNT]` - channels concatenated
/// - `weights`: `[OUT_C * IN_C * K * K]` - `out_ch` major, then `in_ch`, then kernel
/// - `bias`: `[OUT_C]`
/// - Returns: `[OUT_C * PositionId::COUNT]`
///
/// # Panics
///
/// Panics if input slices have incorrect lengths.
pub fn conv2d<const IN_C: usize, const OUT_C: usize, const K: usize>(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
) -> Vec<f32> {
    assert_eq!(
        input.len(),
        IN_C * PositionId::COUNT,
        "input length mismatch"
    );
    assert_eq!(
        weights.len(),
        OUT_C * IN_C * K * K,
        "weights length mismatch"
    );
    assert_eq!(bias.len(), OUT_C, "bias length mismatch");

    let pad = isize::try_from(K / 2).expect("kernel size too large");
    let mut output = vec![0.0f32; OUT_C * PositionId::COUNT];

    for out_ch in 0..OUT_C {
        let out_channel_data = &mut output[out_ch * PositionId::COUNT..][..PositionId::COUNT];
        let mut out_channel = PositionSliceMut::new(out_channel_data);

        for pos in PositionId::iter() {
            let mut sum = bias[out_ch];

            for in_ch in 0..IN_C {
                let in_channel =
                    PositionSlice::new(&input[in_ch * PositionId::COUNT..][..PositionId::COUNT]);

                for kr in 0..K {
                    for kc in 0..K {
                        let row_offset = isize::try_from(kr).expect("kernel size too large") - pad;
                        let col_offset = isize::try_from(kc).expect("kernel size too large") - pad;
                        let offset = Offset::new(row_offset, col_offset);

                        if let Some(in_pos) = pos.from_offset(offset) {
                            let w_idx = out_ch * IN_C * K * K + in_ch * K * K + kr * K + kc;
                            sum += in_channel[in_pos] * weights[w_idx];
                        }
                    }
                }
            }

            out_channel[pos] = sum;
        }
    }

    output
}

/// Applies `ReLU` activation in-place: `x = max(0, x)`.
fn relu_inplace(data: &mut [f32]) {
    for val in data.iter_mut() {
        *val = val.max(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create a single-channel input with one non-zero value
    fn single_value_input<const C: usize>(pos: PositionId, value: f32) -> Vec<f32> {
        let mut input = vec![0.0f32; C * PositionId::COUNT];
        input[usize::from(pos)] = value;
        input
    }

    // Helper to get value at position from multi-channel output
    fn output_at<const C: usize>(output: &[f32], channel: usize, pos: PositionId) -> f32 {
        output[channel * PositionId::COUNT + usize::from(pos)]
    }

    mod conv2d_tests {
        use super::*;
        use crate::position::Position;

        fn pos(row: usize, col: usize) -> PositionId {
            PositionId::from_position(Position::new(row, col))
        }

        #[test]
        fn output_has_correct_shape() {
            let input = vec![0.0f32; 2 * PositionId::COUNT];
            let weights = vec![0.0f32; 4 * 2 * 3 * 3]; // 4 out, 2 in, 3x3 kernel
            let bias = vec![0.0f32; 4];

            let output = conv2d::<2, 4, 3>(&input, &weights, &bias);

            assert_eq!(output.len(), 4 * PositionId::COUNT);
        }

        #[test]
        fn bias_only_produces_constant_output() {
            let input = vec![0.0f32; 1 * PositionId::COUNT];
            let weights = vec![0.0f32; 2 * 1 * 3 * 3];
            let bias = vec![1.5, -0.5];

            let output = conv2d::<1, 2, 3>(&input, &weights, &bias);

            // Channel 0 should all be 1.5, channel 1 should all be -0.5
            for pos in PositionId::iter() {
                assert_eq!(output_at::<2>(&output, 0, pos), 1.5);
                assert_eq!(output_at::<2>(&output, 1, pos), -0.5);
            }
        }

        #[test]
        fn identity_kernel_copies_input() {
            // 3x3 kernel with center=1, rest=0 acts as identity
            let center = PositionId::center();
            let input = single_value_input::<1>(center, 7.0);

            let mut weights = vec![0.0f32; 1 * 1 * 3 * 3];
            weights[4] = 1.0; // Center of 3x3 kernel (index 4)
            let bias = vec![0.0f32; 1];

            let output = conv2d::<1, 1, 3>(&input, &weights, &bias);

            assert_eq!(output_at::<1>(&output, 0, center), 7.0);
            // Other positions should be 0
            assert_eq!(output_at::<1>(&output, 0, pos(0, 0)), 0.0);
        }

        #[test]
        fn shift_kernel_moves_value() {
            // Kernel with only top-left (0,0) = 1 shifts input down-right
            // When kernel[0,0]=1, output[r,c] = input[r-1,c-1]
            let input_pos = pos(5, 5);
            let input = single_value_input::<1>(input_pos, 3.0);

            let mut weights = vec![0.0f32; 1 * 1 * 3 * 3];
            weights[0] = 1.0; // Top-left of kernel
            let bias = vec![0.0f32; 1];

            let output = conv2d::<1, 1, 3>(&input, &weights, &bias);

            // Value should appear at (6, 6) - shifted down and right
            let output_pos = pos(6, 6);
            assert_eq!(output_at::<1>(&output, 0, output_pos), 3.0);
            assert_eq!(output_at::<1>(&output, 0, input_pos), 0.0);
        }

        #[test]
        fn summing_kernel_sums_neighbors() {
            // 3x3 kernel of all 1s sums the 3x3 neighborhood
            // Place a 2x2 block of 1s at center, expect various sums
            let mut input = vec![0.0f32; 1 * PositionId::COUNT];
            let positions = [pos(7, 7), pos(7, 8), pos(8, 7), pos(8, 8)];
            for &p in &positions {
                input[usize::from(p)] = 1.0;
            }

            let weights = vec![1.0f32; 1 * 1 * 3 * 3]; // All 1s
            let bias = vec![0.0f32; 1];

            let output = conv2d::<1, 1, 3>(&input, &weights, &bias);

            // Center of the 2x2 block should see all 4 values
            // But actually the kernel centered at (7,7) sees positions (6,6) to (8,8)
            // which contains positions (7,7), (7,8), (8,7), (8,8) = 4 ones
            assert_eq!(output_at::<1>(&output, 0, pos(7, 7)), 4.0);

            // Corner position (6,6) kernel window (5,5)-(7,7) contains only (7,7) = 1
            assert_eq!(output_at::<1>(&output, 0, pos(6, 6)), 1.0);

            // Position (9,9) kernel window (8,8)-(10,10) contains only (8,8) = 1
            assert_eq!(output_at::<1>(&output, 0, pos(9, 9)), 1.0);
        }

        #[test]
        fn multiple_input_channels_are_summed() {
            // Two input channels, each with a value at center
            let center = PositionId::center();
            let mut input = vec![0.0f32; 2 * PositionId::COUNT];
            input[usize::from(center)] = 2.0; // Channel 0
            input[PositionId::COUNT + usize::from(center)] = 3.0; // Channel 1

            // Identity kernel for both channels (center=1)
            let mut weights = vec![0.0f32; 1 * 2 * 3 * 3];
            weights[4] = 1.0; // Channel 0 kernel center
            weights[9 + 4] = 1.0; // Channel 1 kernel center
            let bias = vec![0.0f32; 1];

            let output = conv2d::<2, 1, 3>(&input, &weights, &bias);

            // Output should be sum: 2.0 + 3.0 = 5.0
            assert_eq!(output_at::<1>(&output, 0, center), 5.0);
        }

        #[test]
        fn edge_position_uses_zero_padding() {
            // Place value at corner, use summing kernel
            let corner = pos(0, 0);
            let input = single_value_input::<1>(corner, 9.0);

            let weights = vec![1.0f32; 1 * 1 * 3 * 3];
            let bias = vec![0.0f32; 1];

            let output = conv2d::<1, 1, 3>(&input, &weights, &bias);

            // Kernel centered at (0,0) only sees (0,0) itself (others are padding)
            assert_eq!(output_at::<1>(&output, 0, corner), 9.0);

            // Kernel centered at (1,1) sees (0,0) through (2,2), includes corner
            assert_eq!(output_at::<1>(&output, 0, pos(1, 1)), 9.0);

            // Kernel centered at (0,1) sees (-1,0) through (1,2), includes corner
            assert_eq!(output_at::<1>(&output, 0, pos(0, 1)), 9.0);
        }

        #[test]
        fn weighted_kernel_applies_correctly() {
            // Hand-calculated: 3x3 kernel on known input
            // Input at center = 2.0, at (7,8) = 3.0
            let mut input = vec![0.0f32; 1 * PositionId::COUNT];
            input[usize::from(pos(7, 7))] = 2.0;
            input[usize::from(pos(7, 8))] = 3.0;

            // Kernel: center=2, right-of-center=3
            // Layout: [0,1,2,3,4,5,6,7,8] = row-major
            // Center is index 4, right-of-center is index 5
            let mut weights = vec![0.0f32; 1 * 1 * 3 * 3];
            weights[4] = 2.0; // center
            weights[5] = 3.0; // right of center
            let bias = vec![1.0f32; 1];

            let output = conv2d::<1, 1, 3>(&input, &weights, &bias);

            // At (7,7): sees input(7,7)=2 at kernel-center, input(7,8)=3 at kernel-right
            // sum = bias + 2*2 + 3*3 = 1 + 4 + 9 = 14
            assert_eq!(output_at::<1>(&output, 0, pos(7, 7)), 14.0);

            // At (7,8): sees input(7,8)=3 at kernel-center, input(7,7)=2 at kernel-left (idx 3)
            // But kernel[3] = 0, so only kernel-center contributes
            // sum = bias + 2*3 = 1 + 6 = 7
            assert_eq!(output_at::<1>(&output, 0, pos(7, 8)), 7.0);
        }
    }

    mod relu_tests {
        use super::*;

        #[test]
        fn positive_values_unchanged() {
            let mut data = vec![1.0, 2.5, 0.001];

            relu_inplace(&mut data);

            assert_eq!(data, vec![1.0, 2.5, 0.001]);
        }

        #[test]
        fn negative_values_become_zero() {
            let mut data = vec![-1.0, -0.001, -100.0];

            relu_inplace(&mut data);

            assert_eq!(data, vec![0.0, 0.0, 0.0]);
        }

        #[test]
        fn zero_unchanged() {
            let mut data = vec![0.0];

            relu_inplace(&mut data);

            assert_eq!(data, vec![0.0]);
        }

        #[test]
        fn mixed_values() {
            let mut data = vec![-2.0, 0.0, 3.0, -0.5, 1.0];

            relu_inplace(&mut data);

            assert_eq!(data, vec![0.0, 0.0, 3.0, 0.0, 1.0]);
        }
    }
}
