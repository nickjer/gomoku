use crate::offset::Offset;
use crate::position_id::PositionId;
use crate::position_map::{PositionSlice, PositionSliceMut};

/// Gather im2col-style patches for convolution.
///
/// Transforms input from `[IN_C][225]` to patches `[IN_C * K * K][225]`.
/// Each slice of 225 elements contains the values for one kernel position across all positions.
/// Out-of-bounds positions are zero-padded.
///
/// Layout `[patch_size][225]` allows using `PositionSliceMut` for clean indexing.
#[inline]
fn gather_patches<const IN_C: usize, const K: usize>(input: &[f32], patches: &mut [f32]) {
    let pad = isize::try_from(K / 2).expect("kernel size too large");

    for in_ch in 0..IN_C {
        let in_channel =
            PositionSlice::new(&input[in_ch * PositionId::COUNT..][..PositionId::COUNT]);

        for kr in 0..K {
            for kc in 0..K {
                let k_idx = in_ch * K * K + kr * K + kc;
                let mut out_slice = PositionSliceMut::new(
                    &mut patches[k_idx * PositionId::COUNT..][..PositionId::COUNT],
                );

                let row_offset = isize::try_from(kr).expect("kernel size too large") - pad;
                let col_offset = isize::try_from(kc).expect("kernel size too large") - pad;
                let offset = Offset::new(row_offset, col_offset);

                for pos in PositionId::iter() {
                    out_slice[pos] = pos
                        .from_offset(offset)
                        .map_or(0.0, |neighbor| in_channel[neighbor]);
                }
            }
        }
    }
}

/// Convolution using pre-gathered im2col patches via rank-1 updates.
///
/// - `patches`: `[patch_size][225]` - channel-major patches
/// - `weights`: `[patch_size][OUT_C]` - weights in im2col layout
/// - `bias`: `[OUT_C]`
/// - Returns: `[OUT_C][225]`
#[inline]
fn conv2d_from_patches<const OUT_C: usize>(
    patches: &[f32],
    weights: &[f32],
    bias: &[f32],
) -> Vec<f32> {
    let mut output = vec![0.0f32; OUT_C * PositionId::COUNT];

    // Initialize output with bias
    for (out_ch, out_chunk) in output.chunks_exact_mut(PositionId::COUNT).enumerate() {
        out_chunk.fill(bias[out_ch]);
    }

    // Accumulate via rank-1 updates: output += outer(patches[k], weights[k])
    // Loop order k -> out_ch -> pos keeps patches[k] in cache across all out_ch
    for (k, p_chunk) in patches.chunks_exact(PositionId::COUNT).enumerate() {
        let p = PositionSlice::new(p_chunk);
        let w = &weights[k * OUT_C..][..OUT_C];

        for out_ch in 0..OUT_C {
            let mut out_slice = PositionSliceMut::new(
                &mut output[out_ch * PositionId::COUNT..][..PositionId::COUNT],
            );
            let w_val = w[out_ch];
            for pos in PositionId::iter() {
                out_slice[pos] += p[pos] * w_val;
            }
        }
    }

    output
}

/// im2col-based convolution.
///
/// Weights must be in im2col layout `[IN_C * K * K][OUT_C]`, not standard `[OUT_C][IN_C * K * K]`.
/// This avoids runtime transposition.
///
/// # Panics
///
/// Panics if input slices have incorrect lengths.
pub fn conv2d_im2col<const IN_C: usize, const OUT_C: usize, const K: usize>(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    patch_buffer: &mut [f32],
) -> Vec<f32> {
    let patch_size = IN_C * K * K;

    assert_eq!(
        input.len(),
        IN_C * PositionId::COUNT,
        "input length mismatch: expected {}, got {}",
        IN_C * PositionId::COUNT,
        input.len()
    );
    assert_eq!(
        weights.len(),
        patch_size * OUT_C,
        "weights length mismatch: expected {}, got {}",
        patch_size * OUT_C,
        weights.len()
    );
    assert_eq!(
        bias.len(),
        OUT_C,
        "bias length mismatch: expected {OUT_C}, got {}",
        bias.len()
    );
    assert!(
        patch_buffer.len() >= patch_size * PositionId::COUNT,
        "patch_buffer too small: expected at least {}, got {}",
        patch_size * PositionId::COUNT,
        patch_buffer.len()
    );

    let patches = &mut patch_buffer[..patch_size * PositionId::COUNT];

    // Gather patches: [IN_C][225] -> [patch_size][225]
    gather_patches::<IN_C, K>(input, patches);

    // Convolve using rank-1 updates (weights already in [patch_size][OUT_C] layout)
    conv2d_from_patches::<OUT_C>(patches, weights, bias)
}

/// Applies `ReLU` activation in-place: `x = max(0, x)`.
pub fn relu_inplace(data: &mut [f32]) {
    for val in data.iter_mut() {
        *val = val.max(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;

    fn pos(row: usize, col: usize) -> PositionId {
        PositionId::from_position(Position::new(row, col))
    }

    /// Helper to run conv2d_im2col with automatic patch buffer allocation.
    fn conv<const IN_C: usize, const OUT_C: usize, const K: usize>(
        input: &[f32],
        weights: &[f32],
        bias: &[f32],
    ) -> Vec<f32> {
        let patch_buffer_size = PositionId::COUNT * IN_C * K * K;
        let mut patch_buffer = vec![0.0f32; patch_buffer_size];
        conv2d_im2col::<IN_C, OUT_C, K>(input, weights, bias, &mut patch_buffer)
    }

    /// Creates input with a single non-zero value at the given position in channel 0.
    fn single_value_input<const C: usize>(pos: PositionId, value: f32) -> Vec<f32> {
        let mut input = vec![0.0f32; C * PositionId::COUNT];
        input[usize::from(pos)] = value;
        input
    }

    /// Gets the output value at a specific channel and position.
    fn output_at<const C: usize>(output: &[f32], channel: usize, pos: PositionId) -> f32 {
        output[channel * PositionId::COUNT + usize::from(pos)]
    }

    mod conv2d_im2col_tests {
        use super::*;

        /// Computes weight index in im2col layout `[patch_size][OUT_C]`.
        /// - `in_ch`: input channel (0 to IN_C-1)
        /// - `kr`, `kc`: kernel row and column (0 to K-1)
        /// - `out_ch`: output channel (0 to OUT_C-1)
        const fn weight_index<const IN_C: usize, const OUT_C: usize, const K: usize>(
            in_ch: usize,
            kr: usize,
            kc: usize,
            out_ch: usize,
        ) -> usize {
            let patch_idx = in_ch * K * K + kr * K + kc;
            patch_idx * OUT_C + out_ch
        }

        /// Shorthand for center kernel position (kr=K/2, kc=K/2).
        const fn center_weight_index<const IN_C: usize, const OUT_C: usize, const K: usize>(
            in_ch: usize,
            out_ch: usize,
        ) -> usize {
            weight_index::<IN_C, OUT_C, K>(in_ch, K / 2, K / 2, out_ch)
        }

        #[test]
        fn output_has_correct_shape_3x3_kernel() {
            let input = vec![0.0f32; 2 * PositionId::COUNT];
            let weights = vec![0.0f32; 4 * 2 * 3 * 3]; // 4 out, 2 in, 3x3 kernel
            let bias = vec![0.0f32; 4];

            let output = conv::<2, 4, 3>(&input, &weights, &bias);

            // Output should have OUT_C * 225 elements
            assert_eq!(output.len(), 4 * PositionId::COUNT);
        }

        #[test]
        fn output_has_correct_shape_1x1_kernel() {
            let input = vec![0.0f32; 8 * PositionId::COUNT];
            let weights = vec![0.0f32; 1 * 8 * 1 * 1]; // 1 out, 8 in, 1x1 kernel
            let bias = vec![0.0f32; 1];

            let output = conv::<8, 1, 1>(&input, &weights, &bias);

            // 1x1 kernel used for final layer: C -> 1 channel
            assert_eq!(output.len(), PositionId::COUNT);
        }

        #[test]
        fn bias_only_produces_constant_output() {
            let input = vec![0.0f32; 1 * PositionId::COUNT];
            let weights = vec![0.0f32; 2 * 1 * 3 * 3];
            let bias = vec![1.5, -0.5];

            let output = conv::<1, 2, 3>(&input, &weights, &bias);

            // With zero input and zero weights, output is just bias at every position
            for pos in PositionId::iter() {
                assert_eq!(output_at::<2>(&output, 0, pos), 1.5);
                assert_eq!(output_at::<2>(&output, 1, pos), -0.5);
            }
        }

        #[test]
        fn identity_kernel_copies_input() {
            // 3x3 kernel with center=1, rest=0 acts as identity filter
            let center = PositionId::center();
            let input = single_value_input::<1>(center, 7.0);

            let mut weights = vec![0.0f32; 1 * 1 * 3 * 3];
            weights[4] = 1.0; // Center of 3x3 kernel (index 4 in row-major)
            let bias = vec![0.0f32; 1];

            let output = conv::<1, 1, 3>(&input, &weights, &bias);

            // Identity kernel preserves the input value at center
            assert_eq!(output_at::<1>(&output, 0, center), 7.0);
            // Other positions should be 0 (no input there)
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

            let output = conv::<1, 1, 3>(&input, &weights, &bias);

            // Value should appear at (6, 6) - shifted down and right by 1
            let output_pos = pos(6, 6);
            assert_eq!(output_at::<1>(&output, 0, output_pos), 3.0);
            // Original position should be 0 (shifted away)
            assert_eq!(output_at::<1>(&output, 0, input_pos), 0.0);
        }

        #[test]
        fn summing_kernel_sums_neighbors() {
            // 3x3 kernel of all 1s sums the 3x3 neighborhood
            let mut input = vec![0.0f32; 1 * PositionId::COUNT];
            // Place a 2x2 block of 1s
            let positions = [pos(7, 7), pos(7, 8), pos(8, 7), pos(8, 8)];
            for &p in &positions {
                input[usize::from(p)] = 1.0;
            }

            let weights = vec![1.0f32; 1 * 1 * 3 * 3]; // All 1s
            let bias = vec![0.0f32; 1];

            let output = conv::<1, 1, 3>(&input, &weights, &bias);

            // Kernel centered at (7,7) sees window (6,6)-(8,8), contains all 4 ones
            assert_eq!(output_at::<1>(&output, 0, pos(7, 7)), 4.0);

            // Kernel centered at (6,6) sees window (5,5)-(7,7), contains only (7,7)
            assert_eq!(output_at::<1>(&output, 0, pos(6, 6)), 1.0);

            // Kernel centered at (9,9) sees window (8,8)-(10,10), contains only (8,8)
            assert_eq!(output_at::<1>(&output, 0, pos(9, 9)), 1.0);
        }

        #[test]
        fn multiple_input_channels_are_summed() {
            // Convolution sums contributions from all input channels
            let center = PositionId::center();
            let mut input = vec![0.0f32; 2 * PositionId::COUNT];
            input[usize::from(center)] = 2.0; // Channel 0
            input[PositionId::COUNT + usize::from(center)] = 3.0; // Channel 1

            // Identity kernel for both channels (center=1)
            let mut weights = vec![0.0f32; 1 * 2 * 3 * 3];
            weights[4] = 1.0; // Channel 0 kernel center
            weights[9 + 4] = 1.0; // Channel 1 kernel center
            let bias = vec![0.0f32; 1];

            let output = conv::<2, 1, 3>(&input, &weights, &bias);

            // Output is sum of both channels: 2.0 + 3.0 = 5.0
            assert_eq!(output_at::<1>(&output, 0, center), 5.0);
        }

        #[test]
        fn edge_position_uses_zero_padding() {
            // Out-of-bounds positions are treated as zero (same-padding)
            let corner = pos(0, 0);
            let input = single_value_input::<1>(corner, 9.0);

            let weights = vec![1.0f32; 1 * 1 * 3 * 3]; // Summing kernel
            let bias = vec![0.0f32; 1];

            let output = conv::<1, 1, 3>(&input, &weights, &bias);

            // Kernel centered at (0,0): 5 of 9 positions are padding (zero)
            // Only (0,0), (0,1), (1,0), (1,1) are valid, but only (0,0) has value
            assert_eq!(output_at::<1>(&output, 0, corner), 9.0);

            // Kernel centered at (1,1) sees full 3x3 window including corner
            assert_eq!(output_at::<1>(&output, 0, pos(1, 1)), 9.0);

            // Kernel centered at (0,1) sees (0,0) in its left column
            assert_eq!(output_at::<1>(&output, 0, pos(0, 1)), 9.0);
        }

        #[test]
        fn weighted_kernel_applies_correctly() {
            // Hand-calculated example with specific weights
            let mut input = vec![0.0f32; 1 * PositionId::COUNT];
            input[usize::from(pos(7, 7))] = 2.0;
            input[usize::from(pos(7, 8))] = 3.0;

            // Kernel layout (row-major): [0,1,2,3,4,5,6,7,8]
            // Position 4 = center, position 5 = right of center
            let mut weights = vec![0.0f32; 1 * 1 * 3 * 3];
            weights[4] = 2.0; // center weight
            weights[5] = 3.0; // right-of-center weight
            let bias = vec![1.0f32; 1];

            let output = conv::<1, 1, 3>(&input, &weights, &bias);

            // At (7,7): kernel center sees 2.0, kernel right sees 3.0
            // sum = bias + center*2.0 + right*3.0 = 1 + 2*2 + 3*3 = 14
            assert_eq!(output_at::<1>(&output, 0, pos(7, 7)), 14.0);

            // At (7,8): kernel center sees 3.0, kernel left (idx 3) sees 2.0
            // But weights[3] = 0, so only center contributes
            // sum = bias + center*3.0 = 1 + 2*3 = 7
            assert_eq!(output_at::<1>(&output, 0, pos(7, 8)), 7.0);
        }

        #[test]
        fn one_by_one_kernel_acts_as_pointwise() {
            // 1x1 convolution is a learned linear combination at each position
            let center = PositionId::center();
            let mut input = vec![0.0f32; 2 * PositionId::COUNT];
            input[usize::from(center)] = 3.0; // Channel 0
            input[PositionId::COUNT + usize::from(center)] = 4.0; // Channel 1

            // 1x1 conv: output = w0*in0 + w1*in1 + bias
            let weights = vec![2.0, 0.5]; // [OUT_C * IN_C * 1 * 1]
            let bias = vec![1.0];

            let output = conv::<2, 1, 1>(&input, &weights, &bias);

            // At center: 2.0 * 3.0 + 0.5 * 4.0 + 1.0 = 6 + 2 + 1 = 9
            assert_eq!(output_at::<1>(&output, 0, center), 9.0);
            // Other positions have zero input: 2.0 * 0 + 0.5 * 0 + 1.0 = 1
            assert_eq!(output_at::<1>(&output, 0, pos(0, 0)), 1.0);
        }

        #[test]
        fn multiple_output_channels() {
            // Each output channel has its own set of weights
            let center = PositionId::center();
            let input = single_value_input::<1>(center, 5.0);

            // Weights in im2col layout [patch_size][OUT_C] = [9][2]
            let mut weights = vec![0.0f32; 2 * 1 * 3 * 3];
            weights[center_weight_index::<1, 2, 3>(0, 0)] = 1.0; // Out 0: 1x input
            weights[center_weight_index::<1, 2, 3>(0, 1)] = 2.0; // Out 1: 2x input
            let bias = vec![0.0, 10.0];

            let output = conv::<1, 2, 3>(&input, &weights, &bias);

            // Channel 0: 1.0 * 5.0 + 0.0 = 5.0
            assert_eq!(output_at::<2>(&output, 0, center), 5.0);
            // Channel 1: 2.0 * 5.0 + 10.0 = 20.0
            assert_eq!(output_at::<2>(&output, 1, center), 20.0);
        }

        #[test]
        fn different_input_output_channels() {
            // Test IN_C != OUT_C (e.g., first conv: 2 -> C channels)
            let center = PositionId::center();
            let mut input = vec![0.0f32; 2 * PositionId::COUNT];
            input[usize::from(center)] = 1.0; // Channel 0
            input[PositionId::COUNT + usize::from(center)] = 2.0; // Channel 1

            // Weights in im2col layout [patch_size][OUT_C] = [18][3]
            let mut weights = vec![0.0f32; 3 * 2 * 3 * 3];
            // Out 0: 1*in0 + 1*in1 (sum both)
            weights[center_weight_index::<2, 3, 3>(0, 0)] = 1.0; // in_ch=0, out_ch=0
            weights[center_weight_index::<2, 3, 3>(1, 0)] = 1.0; // in_ch=1, out_ch=0
            // Out 1: 2*in0 + 0*in1 (only first channel, doubled)
            weights[center_weight_index::<2, 3, 3>(0, 1)] = 2.0; // in_ch=0, out_ch=1
            // Out 2: 0*in0 + 3*in1 (only second channel, tripled)
            weights[center_weight_index::<2, 3, 3>(1, 2)] = 3.0; // in_ch=1, out_ch=2
            let bias = vec![0.0, 0.0, 0.0];

            let output = conv::<2, 3, 3>(&input, &weights, &bias);

            assert_eq!(output_at::<3>(&output, 0, center), 3.0); // 1*1 + 1*2 = 3
            assert_eq!(output_at::<3>(&output, 1, center), 2.0); // 2*1 + 0*2 = 2
            assert_eq!(output_at::<3>(&output, 2, center), 6.0); // 0*1 + 3*2 = 6
        }

        #[test]
        fn all_corners_use_zero_padding() {
            // Verify zero-padding works at all four corners
            let mut input = vec![0.0f32; PositionId::COUNT];
            input[usize::from(pos(0, 0))] = 1.0;
            input[usize::from(pos(0, 14))] = 1.0;
            input[usize::from(pos(14, 0))] = 1.0;
            input[usize::from(pos(14, 14))] = 1.0;

            let weights = vec![1.0f32; 9]; // Summing kernel
            let bias = vec![0.0f32];

            let output = conv::<1, 1, 3>(&input, &weights, &bias);

            // Each corner kernel sees only itself (other positions are padding)
            assert_eq!(output_at::<1>(&output, 0, pos(0, 0)), 1.0);
            assert_eq!(output_at::<1>(&output, 0, pos(0, 14)), 1.0);
            assert_eq!(output_at::<1>(&output, 0, pos(14, 0)), 1.0);
            assert_eq!(output_at::<1>(&output, 0, pos(14, 14)), 1.0);
        }

        #[test]
        fn patch_buffer_reuse_does_not_leak_state() {
            // Ensure running multiple convolutions with same buffer works correctly
            let patch_buffer_size = PositionId::COUNT * 2 * 3 * 3;
            let mut patch_buffer = vec![99.0f32; patch_buffer_size]; // Non-zero initial

            let input1 = single_value_input::<2>(PositionId::center(), 5.0);
            let mut weights = vec![0.0f32; 1 * 2 * 3 * 3];
            weights[4] = 1.0;
            let bias = vec![0.0];

            // First convolution with non-zero input
            let output1 = conv2d_im2col::<2, 1, 3>(&input1, &weights, &bias, &mut patch_buffer);

            // Second convolution with zero input (buffer still has old data)
            let input2 = vec![0.0f32; 2 * PositionId::COUNT];
            let output2 = conv2d_im2col::<2, 1, 3>(&input2, &weights, &bias, &mut patch_buffer);

            // First output should have the value from input1
            assert_eq!(output_at::<1>(&output1, 0, PositionId::center()), 5.0);
            // Second output should be zero (not contaminated by first run)
            assert_eq!(output_at::<1>(&output2, 0, PositionId::center()), 0.0);
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

        #[test]
        fn empty_slice() {
            let mut data: Vec<f32> = vec![];
            relu_inplace(&mut data);
            assert!(data.is_empty());
        }
    }
}
