use crate::offset::Offset;
use crate::position_id::PositionId;
use crate::position_map::{PositionMap, PositionMapView, PositionMapViewMut};

/// Gathers zero-padded input neighborhoods into the workspace buffer.
///
/// For each board position, collects the K×K neighborhood across all input channels
/// into a contiguous slice. Out-of-bounds positions are zero-padded.
///
/// The workspace stride must equal `input.stride() * K * K`.
#[allow(clippy::inline_always)]
#[inline(always)]
fn gather_workspace<const K: usize>(
    input: PositionMapView<'_, f32>,
    mut workspace: PositionMapViewMut<'_, f32>,
) {
    let pad = isize::try_from(K / 2).expect("kernel size too large");

    for pos in PositionId::iter() {
        let padded_input = workspace.get_mut(pos);
        padded_input.fill(0.0);

        for kr in 0..K {
            for kc in 0..K {
                let row_offset = isize::try_from(kr).expect("kernel size too large") - pad;
                let col_offset = isize::try_from(kc).expect("kernel size too large") - pad;
                let offset = Offset::new(row_offset, col_offset);

                if let Some(neighbor) = pos.from_offset(offset) {
                    let channels = input.get(neighbor);
                    for (in_ch, &val) in channels.iter().enumerate() {
                        padded_input[in_ch * K * K + kr * K + kc] = val;
                    }
                }
            }
        }
    }
}

/// Convolution using pre-gathered workspace data.
///
/// - `workspace`: per-position padded input neighborhoods
/// - `weights`: `[workspace.stride()][OUT_C]` layout
/// - `bias`: `[OUT_C]`
#[allow(clippy::inline_always)]
#[inline(always)]
fn conv2d_from_workspace<const OUT_C: usize>(
    workspace: PositionMapView<'_, f32>,
    weights: &[f32],
    bias: &[f32],
) -> PositionMap<f32> {
    let stride = workspace.stride();
    let mut output = PositionMap::new(0.0, OUT_C);

    for pos in PositionId::iter() {
        let out = output.get_mut(pos);
        let input = workspace.get(pos);
        for k in 0..stride {
            let val = input[k];
            let w = &weights[k * OUT_C..][..OUT_C];
            for out_ch in 0..OUT_C {
                out[out_ch] += val * w[out_ch];
            }
        }
        for out_ch in 0..OUT_C {
            out[out_ch] += bias[out_ch];
        }
    }
    output
}

/// Convolution with workspace-based neighborhood gathering.
///
/// Weights must be in `[IN_C * K * K][OUT_C]` layout, where `IN_C` is `input.stride()`.
///
/// # Panics
///
/// Panics if weight, bias, or workspace slices have incorrect lengths.
pub fn conv2d<const OUT_C: usize, const K: usize>(
    input: PositionMapView<'_, f32>,
    weights: &[f32],
    bias: &[f32],
    workspace: &mut [f32],
) -> PositionMap<f32> {
    let stride = input.stride() * K * K;

    assert_eq!(
        weights.len(),
        stride * OUT_C,
        "weights length mismatch: expected {}, got {}",
        stride * OUT_C,
        weights.len()
    );
    assert_eq!(
        bias.len(),
        OUT_C,
        "bias length mismatch: expected {OUT_C}, got {}",
        bias.len()
    );
    assert!(
        workspace.len() >= stride * PositionId::COUNT,
        "workspace too small: expected at least {}, got {}",
        stride * PositionId::COUNT,
        workspace.len()
    );

    let workspace_mut = PositionMapViewMut::new(workspace, stride);
    gather_workspace::<K>(input, workspace_mut);

    let workspace_view = PositionMapView::new(workspace, stride);
    conv2d_from_workspace::<OUT_C>(workspace_view, weights, bias)
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

    /// Helper to run conv2d with automatic workspace allocation.
    fn conv<const IN_C: usize, const OUT_C: usize, const K: usize>(
        input: &[f32],
        weights: &[f32],
        bias: &[f32],
    ) -> PositionMap<f32> {
        let workspace_size = PositionId::COUNT * IN_C * K * K;
        let mut workspace = vec![0.0f32; workspace_size];
        let input_view = PositionMapView::new(input, IN_C);
        conv2d::<OUT_C, K>(input_view, weights, bias, &mut workspace)
    }

    /// Creates input with a single non-zero value at the given position in channel 0.
    fn single_value_input<const C: usize>(position: PositionId, value: f32) -> Vec<f32> {
        let mut input = PositionMap::new(0.0, C);
        input.get_mut(position)[0] = value;
        input.into_vec()
    }

    mod conv2d_tests {
        use super::*;

        /// Computes weight index in `[stride][OUT_C]` layout.
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
            let weights = vec![0.0f32; 4 * 2 * 3 * 3];
            let bias = vec![0.0f32; 4];

            let output = conv::<2, 4, 3>(&input, &weights, &bias);

            assert_eq!(output.as_slice().len(), 4 * PositionId::COUNT);
        }

        #[test]
        fn output_has_correct_shape_1x1_kernel() {
            let input = vec![0.0f32; 8 * PositionId::COUNT];
            let weights = vec![0.0f32; 1 * 8 * 1 * 1];
            let bias = vec![0.0f32; 1];

            let output = conv::<8, 1, 1>(&input, &weights, &bias);

            assert_eq!(output.as_slice().len(), PositionId::COUNT);
        }

        #[test]
        fn bias_only_produces_constant_output() {
            let input = vec![0.0f32; 1 * PositionId::COUNT];
            let weights = vec![0.0f32; 2 * 1 * 3 * 3];
            let bias = vec![1.5, -0.5];

            let output = conv::<1, 2, 3>(&input, &weights, &bias);

            for pos in PositionId::iter() {
                assert_eq!(output.get(pos), &[1.5, -0.5]);
            }
        }

        #[test]
        fn identity_kernel_copies_input() {
            let center = PositionId::center();
            let input = single_value_input::<1>(center, 7.0);

            let mut weights = vec![0.0f32; 1 * 1 * 3 * 3];
            weights[4] = 1.0;
            let bias = vec![0.0f32; 1];

            let output = conv::<1, 1, 3>(&input, &weights, &bias);

            assert_eq!(output.get(center), &[7.0]);
            assert_eq!(output.get(pos(0, 0)), &[0.0]);
        }

        #[test]
        fn shift_kernel_moves_value() {
            let input_pos = pos(5, 5);
            let input = single_value_input::<1>(input_pos, 3.0);

            let mut weights = vec![0.0f32; 1 * 1 * 3 * 3];
            weights[0] = 1.0;
            let bias = vec![0.0f32; 1];

            let output = conv::<1, 1, 3>(&input, &weights, &bias);

            let output_pos = pos(6, 6);
            assert_eq!(output.get(output_pos), &[3.0]);
            assert_eq!(output.get(input_pos), &[0.0]);
        }

        #[test]
        fn summing_kernel_sums_neighbors() {
            let mut input = PositionMap::new(0.0f32, 1);
            let positions = [pos(7, 7), pos(7, 8), pos(8, 7), pos(8, 8)];
            for &p in &positions {
                input.get_mut(p)[0] = 1.0;
            }

            let weights = vec![1.0f32; 1 * 1 * 3 * 3];
            let bias = vec![0.0f32; 1];

            let output = conv::<1, 1, 3>(input.as_slice(), &weights, &bias);

            assert_eq!(output.get(pos(7, 7)), &[4.0]);
            assert_eq!(output.get(pos(6, 6)), &[1.0]);
            assert_eq!(output.get(pos(9, 9)), &[1.0]);
        }

        #[test]
        fn multiple_input_channels_are_summed() {
            let center = PositionId::center();
            let mut input = PositionMap::new(0.0f32, 2);
            input.get_mut(center).copy_from_slice(&[2.0, 3.0]);

            let mut weights = vec![0.0f32; 1 * 2 * 3 * 3];
            weights[4] = 1.0;
            weights[9 + 4] = 1.0;
            let bias = vec![0.0f32; 1];

            let output = conv::<2, 1, 3>(input.as_slice(), &weights, &bias);

            assert_eq!(output.get(center), &[5.0]);
        }

        #[test]
        fn edge_position_uses_zero_padding() {
            let corner = pos(0, 0);
            let input = single_value_input::<1>(corner, 9.0);

            let weights = vec![1.0f32; 1 * 1 * 3 * 3];
            let bias = vec![0.0f32; 1];

            let output = conv::<1, 1, 3>(&input, &weights, &bias);

            assert_eq!(output.get(corner), &[9.0]);
            assert_eq!(output.get(pos(1, 1)), &[9.0]);
            assert_eq!(output.get(pos(0, 1)), &[9.0]);
        }

        #[test]
        fn weighted_kernel_applies_correctly() {
            let mut input = PositionMap::new(0.0f32, 1);
            input.get_mut(pos(7, 7))[0] = 2.0;
            input.get_mut(pos(7, 8))[0] = 3.0;

            let mut weights = vec![0.0f32; 1 * 1 * 3 * 3];
            weights[4] = 2.0;
            weights[5] = 3.0;
            let bias = vec![1.0f32; 1];

            let output = conv::<1, 1, 3>(input.as_slice(), &weights, &bias);

            assert_eq!(output.get(pos(7, 7)), &[14.0]);
            assert_eq!(output.get(pos(7, 8)), &[7.0]);
        }

        #[test]
        fn one_by_one_kernel_acts_as_pointwise() {
            let center = PositionId::center();
            let mut input = PositionMap::new(0.0f32, 2);
            input.get_mut(center).copy_from_slice(&[3.0, 4.0]);

            let weights = vec![2.0, 0.5];
            let bias = vec![1.0];

            let output = conv::<2, 1, 1>(input.as_slice(), &weights, &bias);

            assert_eq!(output.get(center), &[9.0]);
            assert_eq!(output.get(pos(0, 0)), &[1.0]);
        }

        #[test]
        fn multiple_output_channels() {
            let center = PositionId::center();
            let input = single_value_input::<1>(center, 5.0);

            let mut weights = vec![0.0f32; 2 * 1 * 3 * 3];
            weights[center_weight_index::<1, 2, 3>(0, 0)] = 1.0;
            weights[center_weight_index::<1, 2, 3>(0, 1)] = 2.0;
            let bias = vec![0.0, 10.0];

            let output = conv::<1, 2, 3>(&input, &weights, &bias);

            assert_eq!(output.get(center), &[5.0, 20.0]);
        }

        #[test]
        fn different_input_output_channels() {
            let center = PositionId::center();
            let mut input = PositionMap::new(0.0f32, 2);
            input.get_mut(center).copy_from_slice(&[1.0, 2.0]);

            let mut weights = vec![0.0f32; 3 * 2 * 3 * 3];
            weights[center_weight_index::<2, 3, 3>(0, 0)] = 1.0;
            weights[center_weight_index::<2, 3, 3>(1, 0)] = 1.0;
            weights[center_weight_index::<2, 3, 3>(0, 1)] = 2.0;
            weights[center_weight_index::<2, 3, 3>(1, 2)] = 3.0;
            let bias = vec![0.0, 0.0, 0.0];

            let output = conv::<2, 3, 3>(input.as_slice(), &weights, &bias);

            assert_eq!(output.get(center), &[3.0, 2.0, 6.0]);
        }

        #[test]
        fn all_corners_use_zero_padding() {
            let mut input = PositionMap::new(0.0f32, 1);
            input.get_mut(pos(0, 0))[0] = 1.0;
            input.get_mut(pos(0, 14))[0] = 1.0;
            input.get_mut(pos(14, 0))[0] = 1.0;
            input.get_mut(pos(14, 14))[0] = 1.0;

            let weights = vec![1.0f32; 9];
            let bias = vec![0.0f32];

            let output = conv::<1, 1, 3>(input.as_slice(), &weights, &bias);

            assert_eq!(output.get(pos(0, 0)), &[1.0]);
            assert_eq!(output.get(pos(0, 14)), &[1.0]);
            assert_eq!(output.get(pos(14, 0)), &[1.0]);
            assert_eq!(output.get(pos(14, 14)), &[1.0]);
        }

        #[test]
        fn workspace_reuse_does_not_leak_state() {
            let workspace_size = PositionId::COUNT * 2 * 3 * 3;
            let mut workspace = vec![99.0f32; workspace_size];

            let input1 = single_value_input::<2>(PositionId::center(), 5.0);
            let mut weights = vec![0.0f32; 1 * 2 * 3 * 3];
            weights[4] = 1.0;
            let bias = vec![0.0];

            let input1_view = PositionMapView::new(&input1, 2);
            let output1 = conv2d::<1, 3>(input1_view, &weights, &bias, &mut workspace);

            let input2 = vec![0.0f32; 2 * PositionId::COUNT];
            let input2_view = PositionMapView::new(&input2, 2);
            let output2 = conv2d::<1, 3>(input2_view, &weights, &bias, &mut workspace);

            assert_eq!(output1.get(PositionId::center()), &[5.0]);
            assert_eq!(output2.get(PositionId::center()), &[0.0]);
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
