use fastrand_contrib::RngExt;
use serde::{Deserialize, Serialize};

use crate::evolution::crossover::uniform_crossover;
use crate::evolution::mutation::gaussian_mutate;
use crate::nn::he_std;
use crate::position_map::PositionMap;

use super::features::{NEIGHBOR_OFFSETS, compute_features};

/// A single cluster layer's parameters (weights + bias).
///
/// Uses D8-equivariant polynomial features instead of raw convolution.
/// The first `F` features (out of 9 total) are used per layer, enabling
/// F-truncation: F=9 for spatial layers, F=1 for pointwise output.
///
/// # Type Parameters
/// - `IN_C`: Number of input channels
/// - `OUT_C`: Number of output channels
/// - `F`: Number of features per channel (1..=9)
///
/// # Weight Layout
/// Weights are in `[IN_C * F][OUT_C]` layout.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClusterParams<const IN_C: usize, const OUT_C: usize, const F: usize> {
    weights: Vec<f32>,
    bias: Vec<f32>,
}

impl<const IN_C: usize, const OUT_C: usize, const F: usize> ClusterParams<IN_C, OUT_C, F> {
    /// Workspace stride: number of elements per position in the gathered features.
    const STRIDE: usize = IN_C.checked_mul(F).expect("STRIDE overflow");

    /// Expected weights length: `IN_C * F * OUT_C`.
    const EXPECTED_WEIGHTS: usize = Self::STRIDE
        .checked_mul(OUT_C)
        .expect("EXPECTED_WEIGHTS overflow");

    /// Creates a new `ClusterParams` with validated dimensions.
    ///
    /// # Panics
    ///
    /// Panics if `weights.len() != IN_C * F * OUT_C` or `bias.len() != OUT_C`.
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

    /// Returns the weight slice in `[IN_C * F][OUT_C]` layout.
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

    /// Applies the cluster layer to the input: gather features, then dot product.
    ///
    /// All dimensions (`IN_C`, `OUT_C`, `F`) are encoded in `Self`, ensuring separate
    /// monomorphizations per layer configuration.
    #[must_use]
    pub fn cluster2d(&self, input: &PositionMap<f32>) -> PositionMap<f32> {
        let workspace = Self::gather_features(input);
        Self::cluster2d_from_workspace(self.weights(), self.bias(), &workspace)
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

    /// Gathers D8-equivariant polynomial features into the workspace buffer.
    ///
    /// For each board position and input channel, fetches the center + 8 neighbors
    /// (zero-padded at edges), computes all 9 polynomial features, then stores
    /// the first `F` into the workspace (F-truncation).
    ///
    /// When `F == 1` (pointwise), only the center value is needed — neighbor
    /// gathering and polynomial computation are skipped entirely.
    fn gather_features(input: &PositionMap<f32>) -> PositionMap<f32> {
        if F == 1 {
            // Pointwise: feature 0 is just the center value, no neighbors needed.
            return PositionMap::from_fn(0.0, Self::STRIDE, |pos, workspace| {
                workspace.copy_from_slice(input.get(pos));
            });
        }
        PositionMap::from_fn(0.0, Self::STRIDE, |pos, workspace| {
            let mut neighbors = [[0.0f32; IN_C]; 9];
            neighbors[0].copy_from_slice(input.get(pos));
            for (idx, offset) in NEIGHBOR_OFFSETS.iter().enumerate() {
                if let Some(neighbor) = pos.offset(*offset) {
                    neighbors[idx + 1].copy_from_slice(input.get(neighbor));
                }
            }
            for channel in 0..IN_C {
                let raw: [f32; 9] = std::array::from_fn(|i| neighbors[i][channel]);
                let features = compute_features(&raw);
                workspace[channel * F..][..F].copy_from_slice(&features[..F]);
            }
        })
    }

    /// Dot product of workspace features against weights + bias.
    ///
    /// This is an associated function (no `&self`) so that `weights` and `bias` arrive
    /// as independent `&[f32]` parameters — LLVM's alias analysis can then prove
    /// no aliasing with the output buffer, enabling auto-vectorization.
    #[must_use]
    fn cluster2d_from_workspace(
        weights: &[f32],
        bias: &[f32],
        workspace: &PositionMap<f32>,
    ) -> PositionMap<f32> {
        PositionMap::from_fn(0.0, OUT_C, |pos, output_channels| {
            output_channels.copy_from_slice(bias);
            let features = workspace.get(pos);
            assert!(features.len() == Self::STRIDE);

            for (&feature_val, weight_row) in features.iter().zip(weights.chunks_exact(OUT_C)) {
                for (out_ch, &weight) in output_channels.iter_mut().zip(weight_row) {
                    *out_ch += feature_val * weight;
                }
            }
        })
    }
}

impl<const IN_C: usize, const OUT_C: usize, const F: usize> std::fmt::Display
    for ClusterParams<IN_C, OUT_C, F>
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use super::features::FEATURE_NAMES;
        use crate::nn::format_slice_stats;

        writeln!(formatter, "Cluster {IN_C} -> {OUT_C}, F={F}")?;

        let weights = self.weights();
        // Weights layout: [IN_C * F][OUT_C]. Gather per-feature stats across all
        // input channels: feature f lives at rows ch*F + f for ch in 0..IN_C.
        for feature_idx in 0..F {
            let per_feature: Vec<f32> = (0..IN_C)
                .flat_map(|ch| {
                    let row = ch * F + feature_idx;
                    let start = row * OUT_C;
                    weights[start..start + OUT_C].iter().copied()
                })
                .collect();
            let name = FEATURE_NAMES.get(feature_idx).unwrap_or(&"?");
            writeln!(
                formatter,
                "  F{feature_idx} ({name:>9}) {}",
                format_slice_stats(&per_feature)
            )?;
        }

        write!(
            formatter,
            "  Bias            {}",
            format_slice_stats(self.bias())
        )
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

    #[test]
    fn valid_params_construction() {
        // in=2, f=9, out=4 → weights: 2*9*4 = 72, bias: 4
        let weights = vec![0.0f32; 2 * 9 * 4];
        let bias = vec![0.0f32; 4];

        let params = ClusterParams::<2, 4, 9>::new(weights, bias);

        assert_eq!(params.weights().len(), 72);
        assert_eq!(params.bias().len(), 4);
    }

    #[test]
    fn pointwise_params_construction() {
        // in=8, f=1, out=1 → weights: 8*1*1 = 8, bias: 1
        let weights = vec![0.0f32; 8];
        let bias = vec![0.0f32; 1];

        let params = ClusterParams::<8, 1, 1>::new(weights, bias);

        assert_eq!(params.weights().len(), 8);
        assert_eq!(params.bias().len(), 1);
    }

    #[test]
    #[should_panic(expected = "weights length mismatch")]
    fn rejects_wrong_weights_length() {
        let weights = vec![0.0f32; 10]; // wrong
        let bias = vec![0.0f32; 4];
        let _ = ClusterParams::<2, 4, 9>::new(weights, bias);
    }

    #[test]
    #[should_panic(expected = "bias length mismatch")]
    fn rejects_wrong_bias_length() {
        let weights = vec![0.0f32; 2 * 9 * 4];
        let bias = vec![0.0f32; 3]; // wrong
        let _ = ClusterParams::<2, 4, 9>::new(weights, bias);
    }

    #[test]
    fn random_creates_correct_lengths() {
        let mut rng = fastrand::Rng::with_seed(42);
        let params = ClusterParams::<2, 4, 9>::random(&mut rng);

        assert_eq!(params.weights().len(), 2 * 9 * 4);
        assert_eq!(params.bias().len(), 4);
    }

    #[test]
    fn random_biases_initialized_to_zero() {
        let mut rng = fastrand::Rng::with_seed(42);
        let params = ClusterParams::<2, 8, 9>::random(&mut rng);

        assert!(params.bias().iter().all(|&bias| bias == 0.0));
    }

    #[test]
    fn he_initialization_has_reasonable_distribution() {
        let mut rng = fastrand::Rng::with_seed(42);
        let params = ClusterParams::<2, 8, 9>::random(&mut rng);

        let weights = params.weights();
        let mean: f32 = weights.iter().sum::<f32>() / weights.len() as f32;
        let variance: f32 =
            weights.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / weights.len() as f32;
        let std_dev = variance.sqrt();

        // Expected std: sqrt(2 / (2 * 9)) = sqrt(1/9) ≈ 0.333
        let expected_std = he_std(2 * 9);
        assert!(
            (std_dev - expected_std).abs() < 0.1,
            "std_dev {std_dev} not close to expected {expected_std}"
        );
    }

    #[test]
    fn uniform_crossover_produces_valid_child() {
        let mut rng = fastrand::Rng::with_seed(42);
        let parent1 = ClusterParams::<2, 4, 9>::random(&mut rng);
        let parent2 = ClusterParams::<2, 4, 9>::random(&mut rng);

        let child = parent1.uniform_crossover(&parent2, &mut rng);

        assert_eq!(child.weights().len(), parent1.weights().len());
        assert_eq!(child.bias().len(), parent1.bias().len());
        for (idx, &val) in child.weights().iter().enumerate() {
            assert!(val == parent1.weights()[idx] || val == parent2.weights()[idx]);
        }
    }

    #[test]
    fn gaussian_mutate_changes_values() {
        let params = ClusterParams::<2, 4, 9>::new(vec![0.0f32; 2 * 9 * 4], vec![0.0f32; 4]);
        let mut rng = fastrand::Rng::with_seed(42);

        let result = params.gaussian_mutate(1.0, &mut rng);

        let changed = result.weights().iter().filter(|&&val| val != 0.0).count();
        assert!(changed > 0, "some weights should change");
    }

    #[test]
    fn bias_only_produces_constant_output() {
        let input = PositionMap::new(0.0f32, 1);
        let weights = vec![0.0f32; 1 * 9 * 2];
        let bias = vec![1.5, -0.5];
        let params = ClusterParams::<1, 2, 9>::new(weights, bias);

        let output = params.cluster2d(&input);

        for pos in PositionId::iter() {
            assert_eq!(output.get(pos), &[1.5, -0.5]);
        }
    }

    #[test]
    fn center_feature_passes_through_with_f1() {
        // F=1 means only center feature (feature[0] = raw center value)
        let center = PositionId::center();
        let mut input = PositionMap::new(0.0f32, 1);
        input.get_mut(center)[0] = 7.0;

        // Weight=1.0 for the single feature, bias=0.0
        let weights = vec![1.0f32];
        let bias = vec![0.0f32];
        let params = ClusterParams::<1, 1, 1>::new(weights, bias);

        let output = params.cluster2d(&input);

        assert_eq!(output.get(center), &[7.0]);
        assert_eq!(output.get(pos(0, 0)), &[0.0]);
    }

    #[test]
    fn edge_positions_use_zero_padding() {
        let corner = pos(0, 0);
        let mut input = PositionMap::new(0.0f32, 1);
        input.get_mut(corner)[0] = 1.0;

        // Use F=3 (center + ortho + diag) with weight=1 for each feature
        let weights = vec![1.0f32; 3];
        let bias = vec![0.0f32];
        let params = ClusterParams::<1, 1, 3>::new(weights, bias);

        let output = params.cluster2d(&input);

        // Corner: center=1.0, ortho=0 (neighbors out of bounds), diag=0
        // Dot product: 1.0*1.0 + 0.0*1.0 + 0.0*1.0 = 1.0
        assert_eq!(output.get(corner), &[1.0]);
    }

    #[test]
    fn multiple_input_channels_are_combined() {
        let center = PositionId::center();
        let mut input = PositionMap::new(0.0f32, 2);
        input.get_mut(center).copy_from_slice(&[2.0, 3.0]);

        // F=1 (pointwise): stride = 2*1 = 2
        // Weights in [STRIDE][OUT_C] layout, OUT_C=1: [w_ch0_f0, w_ch1_f0]
        let weights = vec![1.0, 1.0];
        let bias = vec![0.0];
        let params = ClusterParams::<2, 1, 1>::new(weights, bias);

        let output = params.cluster2d(&input);

        // center features: ch0=2.0, ch1=3.0. dot product = 2+3 = 5
        assert_eq!(output.get(center), &[5.0]);
    }

    #[test]
    fn multiple_output_channels() {
        let center = PositionId::center();
        let mut input = PositionMap::new(0.0f32, 1);
        input.get_mut(center)[0] = 5.0;

        // F=1, IN_C=1, OUT_C=2: stride=1, weights len = 1*2 = 2
        let weights = vec![1.0, 2.0]; // [feature0→out0, feature0→out1]
        let bias = vec![0.0, 10.0];
        let params = ClusterParams::<1, 2, 1>::new(weights, bias);

        let output = params.cluster2d(&input);

        assert_eq!(output.get(center), &[5.0, 20.0]);
    }

    #[test]
    fn successive_calls_produce_independent_results() {
        let mut input1 = PositionMap::new(0.0f32, 2);
        input1.get_mut(PositionId::center())[0] = 5.0;

        let weights = vec![1.0; 2 * 9 * 1];
        let bias = vec![0.0];
        let params = ClusterParams::<2, 1, 9>::new(weights, bias);

        let output1 = params.cluster2d(&input1);

        let input2 = PositionMap::new(0.0f32, 2);
        let output2 = params.cluster2d(&input2);

        // First call should have non-zero at center, second should be zero
        assert_ne!(output1.get(PositionId::center()), &[0.0]);
        assert_eq!(output2.get(PositionId::center()), &[0.0]);
    }
}
