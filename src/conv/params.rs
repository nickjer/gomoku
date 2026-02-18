use fastrand_contrib::RngExt;
use serde::{Deserialize, Serialize};

use crate::evolution::crossover::uniform_crossover;
use crate::evolution::mutation::gaussian_mutate;
use crate::nn::he_std;
use crate::offset::Offset;
use crate::position_map::PositionMap;

/// A single convolutional layer's parameters (weights + bias).
///
/// Owns its data with compile-time dimensions so they cannot be mismatched.
/// Validated at construction time.
///
/// # Type Parameters
/// - `IN_C`: Number of input channels
/// - `OUT_C`: Number of output channels
/// - `K`: Kernel size (e.g., 3 for 3×3 kernels)
///
/// # Weight Layout
/// Weights are in `[IN_C * K * K][OUT_C]` layout.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConvParams<const IN_C: usize, const OUT_C: usize, const K: usize> {
    weights: Vec<f32>,
    bias: Vec<f32>,
}

impl<const IN_C: usize, const OUT_C: usize, const K: usize> ConvParams<IN_C, OUT_C, K> {
    /// Workspace stride: number of elements per position in the gathered neighborhood.
    const STRIDE: usize = IN_C
        .checked_mul(K)
        .expect("STRIDE overflow")
        .checked_mul(K)
        .expect("STRIDE overflow");

    /// Expected weights length: `IN_C * K * K * OUT_C`.
    const EXPECTED_WEIGHTS: usize = Self::STRIDE
        .checked_mul(OUT_C)
        .expect("EXPECTED_WEIGHTS overflow");

    /// Creates a new `ConvParams` with validated dimensions.
    ///
    /// # Panics
    ///
    /// Panics if `weights.len() != IN_C * K * K * OUT_C` or `bias.len() != OUT_C`.
    #[must_use]
    pub fn new(weights: Vec<f32>, bias: Vec<f32>) -> Self {
        assert_eq!(
            weights.len(),
            Self::EXPECTED_WEIGHTS,
            "weights length mismatch: expected {}, got {}",
            Self::EXPECTED_WEIGHTS,
            weights.len()
        );
        assert_eq!(
            bias.len(),
            OUT_C,
            "bias length mismatch: expected {OUT_C}, got {}",
            bias.len()
        );
        Self { weights, bias }
    }

    /// Creates parameters initialized using He initialization.
    ///
    /// Uses `N(0, √(2/n_in))` which is optimal for `ReLU` networks.
    /// Biases are initialized to zero.
    #[must_use]
    pub fn random(rng: &mut fastrand::Rng) -> Self {
        let std = he_std(Self::STRIDE);
        let weights: Vec<f32> = (0..Self::EXPECTED_WEIGHTS)
            .map(|_| rng.f32_normal(0.0, std))
            .collect();
        let bias = vec![0.0; OUT_C];
        Self::new(weights, bias)
    }

    /// Returns the weight slice in `[IN_C * K * K][OUT_C]` layout.
    #[must_use]
    pub fn weights(&self) -> &[f32] {
        assert!(self.weights.len() == Self::EXPECTED_WEIGHTS);
        &self.weights
    }

    /// Returns the bias slice of length `OUT_C`.
    #[must_use]
    pub fn bias(&self) -> &[f32] {
        assert!(self.bias.len() == OUT_C);
        &self.bias
    }

    /// Applies a 2D convolution to the input using workspace-based neighborhood gathering.
    ///
    /// All dimensions (`IN_C`, `OUT_C`, `K`) are encoded in `Self`, ensuring separate
    /// monomorphizations for each layer configuration.
    #[must_use]
    pub fn conv2d(&self, input: &PositionMap<f32>) -> PositionMap<f32> {
        let workspace = Self::gather_workspace(input);
        Self::conv2d_from_workspace(self.weights(), self.bias(), &workspace)
    }

    /// Performs uniform crossover with another set of parameters.
    #[must_use]
    pub fn uniform_crossover(&self, other: &Self, rng: &mut fastrand::Rng) -> Self {
        Self {
            weights: uniform_crossover(&self.weights, &other.weights, rng),
            bias: uniform_crossover(&self.bias, &other.bias, rng),
        }
    }

    /// Performs Gaussian mutation on these parameters.
    #[must_use]
    pub fn gaussian_mutate(&self, sigma: f32, rng: &mut fastrand::Rng) -> Self {
        Self {
            weights: gaussian_mutate(&self.weights, sigma, rng),
            bias: gaussian_mutate(&self.bias, sigma, rng),
        }
    }

    /// Gathers zero-padded input neighborhoods into the workspace buffer.
    ///
    /// For each board position, collects the K×K neighborhood across all input channels
    /// into a contiguous slice. Out-of-bounds positions are zero-padded.
    fn gather_workspace(input: &PositionMap<f32>) -> PositionMap<f32> {
        let half_kernel = isize::try_from(K / 2).expect("kernel size too large");

        PositionMap::from_fn(0.0, Self::STRIDE, |pos, neighborhood| {
            for kernel_row in 0..K {
                for kernel_col in 0..K {
                    let row_offset =
                        isize::try_from(kernel_row).expect("kernel size too large") - half_kernel;
                    let col_offset =
                        isize::try_from(kernel_col).expect("kernel size too large") - half_kernel;
                    let offset = Offset::new(row_offset, col_offset);

                    if let Some(neighbor) = pos.offset(offset) {
                        let channels = input.get(neighbor);
                        for (&channel_val, kernel_plane) in
                            channels.iter().zip(neighborhood.chunks_exact_mut(K * K))
                        {
                            kernel_plane[kernel_row * K + kernel_col] = channel_val;
                        }
                    }
                }
            }
        })
    }

    /// Convolution using pre-gathered workspace data.
    ///
    /// This is an associated function (no `&self`) so that `weights` and `bias` arrive
    /// as independent `&[f32]` parameters. All three const generics (`IN_C`, `OUT_C`, `K`)
    /// are used: the stride is computed as `IN_C * K * K`, producing distinct
    /// monomorphizations per layer configuration.
    #[must_use]
    fn conv2d_from_workspace(
        weights: &[f32],
        bias: &[f32],
        workspace: &PositionMap<f32>,
    ) -> PositionMap<f32> {
        PositionMap::from_fn(0.0, OUT_C, |pos, output_channels| {
            output_channels.copy_from_slice(bias);
            let neighborhood = workspace.get(pos);
            for (&input_val, weight_row) in neighborhood.iter().zip(weights.chunks_exact(OUT_C)) {
                for (out_ch, &weight) in output_channels.iter_mut().zip(weight_row) {
                    *out_ch += input_val * weight;
                }
            }
        })
    }
}

impl<const IN_C: usize, const OUT_C: usize, const K: usize> std::fmt::Display
    for ConvParams<IN_C, OUT_C, K>
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::nn::format_slice_stats;

        writeln!(formatter, "Conv {IN_C} -> {OUT_C}, {K}x{K}")?;
        writeln!(
            formatter,
            "  Weights {}",
            format_slice_stats(self.weights())
        )?;
        write!(formatter, "  Bias    {}", format_slice_stats(self.bias()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;
    use crate::position_id::PositionId;
    use crate::position_map::PositionMap;

    fn pos(row: usize, col: usize) -> PositionId {
        PositionId::from_position(Position::new(row, col))
    }

    /// Helper to run conv2d.
    fn conv<const IN_C: usize, const OUT_C: usize, const K: usize>(
        input: &PositionMap<f32>,
        params: &ConvParams<IN_C, OUT_C, K>,
    ) -> PositionMap<f32> {
        params.conv2d(input)
    }

    /// Creates input with a single non-zero value at the given position in channel 0.
    fn single_value_input(channels: usize, position: PositionId, value: f32) -> PositionMap<f32> {
        let mut input = PositionMap::new(0.0, channels);
        input.get_mut(position)[0] = value;
        input
    }

    #[test]
    fn valid_params_construction() {
        let weights = vec![0.0f32; 2 * 3 * 3 * 4]; // in=2, k=3, out=4
        let bias = vec![0.0f32; 4];

        let params = ConvParams::<2, 4, 3>::new(weights, bias);

        assert_eq!(params.weights().len(), 72);
        assert_eq!(params.bias().len(), 4);
    }

    #[test]
    fn one_by_one_kernel_params() {
        let weights = vec![0.0f32; 8]; // in=8, k=1, out=1
        let bias = vec![0.0f32; 1];

        let params = ConvParams::<8, 1, 1>::new(weights, bias);

        assert_eq!(params.weights().len(), 8);
        assert_eq!(params.bias().len(), 1);
    }

    #[test]
    #[should_panic(expected = "weights length mismatch")]
    fn rejects_wrong_weights_length() {
        let weights = vec![0.0f32; 10]; // wrong
        let bias = vec![0.0f32; 4];
        let _ = ConvParams::<2, 4, 3>::new(weights, bias);
    }

    #[test]
    #[should_panic(expected = "bias length mismatch")]
    fn rejects_wrong_bias_length() {
        let weights = vec![0.0f32; 2 * 3 * 3 * 4];
        let bias = vec![0.0f32; 3]; // wrong
        let _ = ConvParams::<2, 4, 3>::new(weights, bias);
    }

    #[test]
    fn random_creates_correct_lengths() {
        let mut rng = fastrand::Rng::with_seed(42);
        let params = ConvParams::<2, 4, 3>::random(&mut rng);

        assert_eq!(params.weights().len(), 2 * 3 * 3 * 4);
        assert_eq!(params.bias().len(), 4);
    }

    #[test]
    fn random_biases_initialized_to_zero() {
        let mut rng = fastrand::Rng::with_seed(42);
        let params = ConvParams::<2, 8, 3>::random(&mut rng);

        assert!(params.bias().iter().all(|&b| b == 0.0));
    }

    #[test]
    fn he_initialization_has_reasonable_distribution() {
        let mut rng = fastrand::Rng::with_seed(42);
        let params = ConvParams::<2, 8, 3>::random(&mut rng);

        let weights = params.weights();
        let mean: f32 = weights.iter().sum::<f32>() / weights.len() as f32;
        let variance: f32 =
            weights.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / weights.len() as f32;
        let std_dev = variance.sqrt();

        // Expected std: sqrt(2 / (2 * 9)) = sqrt(1/9) ≈ 0.333
        let expected_std = he_std(2 * 3 * 3);
        assert!(
            (std_dev - expected_std).abs() < 0.1,
            "std_dev {std_dev} not close to expected {expected_std}"
        );
    }

    #[test]
    fn uniform_crossover_produces_valid_child() {
        let mut rng = fastrand::Rng::with_seed(42);
        let parent1 = ConvParams::<2, 4, 3>::random(&mut rng);
        let parent2 = ConvParams::<2, 4, 3>::random(&mut rng);

        let child = parent1.uniform_crossover(&parent2, &mut rng);

        assert_eq!(child.weights().len(), parent1.weights().len());
        assert_eq!(child.bias().len(), parent1.bias().len());
        for (i, &val) in child.weights().iter().enumerate() {
            assert!(val == parent1.weights()[i] || val == parent2.weights()[i]);
        }
    }

    #[test]
    fn gaussian_mutate_changes_values() {
        let params = ConvParams::<2, 4, 3>::new(vec![0.0f32; 2 * 3 * 3 * 4], vec![0.0f32; 4]);
        let mut rng = fastrand::Rng::with_seed(42);

        let result = params.gaussian_mutate(1.0, &mut rng);

        let changed = result.weights().iter().filter(|&&val| val != 0.0).count();
        assert!(changed > 0, "some weights should change");
    }

    mod conv2d_tests {
        use super::*;

        /// Computes weight index in `[stride][out_channels]` layout.
        const fn weight_index(
            in_ch: usize,
            kr: usize,
            kc: usize,
            out_ch: usize,
            kernel_size: usize,
            out_channels: usize,
        ) -> usize {
            let patch_idx = in_ch * kernel_size * kernel_size + kr * kernel_size + kc;
            patch_idx * out_channels + out_ch
        }

        /// Shorthand for center kernel position (kr=K/2, kc=K/2).
        const fn center_weight_index(
            in_ch: usize,
            out_ch: usize,
            kernel_size: usize,
            out_channels: usize,
        ) -> usize {
            weight_index(
                in_ch,
                kernel_size / 2,
                kernel_size / 2,
                out_ch,
                kernel_size,
                out_channels,
            )
        }

        #[test]
        fn output_has_correct_shape_3x3_kernel() {
            let input = PositionMap::new(0.0f32, 2);
            let weights = vec![0.0f32; 4 * 2 * 3 * 3];
            let bias = vec![0.0f32; 4];
            let params = ConvParams::<2, 4, 3>::new(weights, bias);

            let output = conv(&input, &params);

            assert_eq!(output.stride(), 4);
        }

        #[test]
        fn output_has_correct_shape_1x1_kernel() {
            let input = PositionMap::new(0.0f32, 8);
            let weights = vec![0.0f32; 1 * 8 * 1 * 1];
            let bias = vec![0.0f32; 1];
            let params = ConvParams::<8, 1, 1>::new(weights, bias);

            let output = conv(&input, &params);

            assert_eq!(output.stride(), 1);
        }

        #[test]
        fn bias_only_produces_constant_output() {
            let input = PositionMap::new(0.0f32, 1);
            let weights = vec![0.0f32; 2 * 1 * 3 * 3];
            let bias = vec![1.5, -0.5];
            let params = ConvParams::<1, 2, 3>::new(weights, bias);

            let output = conv(&input, &params);

            for pos in PositionId::iter() {
                assert_eq!(output.get(pos), &[1.5, -0.5]);
            }
        }

        #[test]
        fn identity_kernel_copies_input() {
            let center = PositionId::center();
            let input = single_value_input(1, center, 7.0);

            let mut weights = vec![0.0f32; 1 * 1 * 3 * 3];
            weights[4] = 1.0;
            let bias = vec![0.0f32; 1];
            let params = ConvParams::<1, 1, 3>::new(weights, bias);

            let output = conv(&input, &params);

            assert_eq!(output.get(center), &[7.0]);
            assert_eq!(output.get(pos(0, 0)), &[0.0]);
        }

        #[test]
        fn shift_kernel_moves_value() {
            let input_pos = pos(5, 5);
            let input = single_value_input(1, input_pos, 3.0);

            let mut weights = vec![0.0f32; 1 * 1 * 3 * 3];
            weights[0] = 1.0;
            let bias = vec![0.0f32; 1];
            let params = ConvParams::<1, 1, 3>::new(weights, bias);

            let output = conv(&input, &params);

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
            let params = ConvParams::<1, 1, 3>::new(weights, bias);

            let output = conv(&input, &params);

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
            let params = ConvParams::<2, 1, 3>::new(weights, bias);

            let output = conv(&input, &params);

            assert_eq!(output.get(center), &[5.0]);
        }

        #[test]
        fn edge_position_uses_zero_padding() {
            let corner = pos(0, 0);
            let input = single_value_input(1, corner, 9.0);

            let weights = vec![1.0f32; 1 * 1 * 3 * 3];
            let bias = vec![0.0f32; 1];
            let params = ConvParams::<1, 1, 3>::new(weights, bias);

            let output = conv(&input, &params);

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
            let params = ConvParams::<1, 1, 3>::new(weights, bias);

            let output = conv(&input, &params);

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
            let params = ConvParams::<2, 1, 1>::new(weights, bias);

            let output = conv(&input, &params);

            assert_eq!(output.get(center), &[9.0]);
            assert_eq!(output.get(pos(0, 0)), &[1.0]);
        }

        #[test]
        fn multiple_output_channels() {
            let center = PositionId::center();
            let input = single_value_input(1, center, 5.0);

            let mut weights = vec![0.0f32; 2 * 1 * 3 * 3];
            weights[center_weight_index(0, 0, 3, 2)] = 1.0;
            weights[center_weight_index(0, 1, 3, 2)] = 2.0;
            let bias = vec![0.0, 10.0];
            let params = ConvParams::<1, 2, 3>::new(weights, bias);

            let output = conv(&input, &params);

            assert_eq!(output.get(center), &[5.0, 20.0]);
        }

        #[test]
        fn different_input_output_channels() {
            let center = PositionId::center();
            let mut input = PositionMap::new(0.0f32, 2);
            input.get_mut(center).copy_from_slice(&[1.0, 2.0]);

            let mut weights = vec![0.0f32; 3 * 2 * 3 * 3];
            weights[center_weight_index(0, 0, 3, 3)] = 1.0;
            weights[center_weight_index(1, 0, 3, 3)] = 1.0;
            weights[center_weight_index(0, 1, 3, 3)] = 2.0;
            weights[center_weight_index(1, 2, 3, 3)] = 3.0;
            let bias = vec![0.0, 0.0, 0.0];
            let params = ConvParams::<2, 3, 3>::new(weights, bias);

            let output = conv(&input, &params);

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
            let params = ConvParams::<1, 1, 3>::new(weights, bias);

            let output = conv(&input, &params);

            assert_eq!(output.get(pos(0, 0)), &[1.0]);
            assert_eq!(output.get(pos(0, 14)), &[1.0]);
            assert_eq!(output.get(pos(14, 0)), &[1.0]);
            assert_eq!(output.get(pos(14, 14)), &[1.0]);
        }

        #[test]
        fn successive_calls_produce_independent_results() {
            let input1 = single_value_input(2, PositionId::center(), 5.0);
            let mut weights = vec![0.0f32; 1 * 2 * 3 * 3];
            weights[4] = 1.0;
            let bias = vec![0.0];
            let params = ConvParams::<2, 1, 3>::new(weights, bias);

            let output1 = params.conv2d(&input1);

            let input2 = PositionMap::new(0.0f32, 2);
            let output2 = params.conv2d(&input2);

            assert_eq!(output1.get(PositionId::center()), &[5.0]);
            assert_eq!(output2.get(PositionId::center()), &[0.0]);
        }
    }
}
