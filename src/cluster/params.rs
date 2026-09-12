use fastrand_contrib::RngExt;
use serde::{Deserialize, Serialize};

use crate::evolution::crossover::uniform_crossover;
use crate::evolution::mutation::gaussian_mutate;
use crate::nn::initial_weight_spread;
use crate::position_id::PositionId;
use crate::position_map::PositionMap;

use crate::nn::{NEIGHBOR_OFFSETS, cluster_sums};

/// A single cluster layer's parameters (weights + bias).
///
/// Uses sums over D8-equivalent neighbor clusters instead of raw convolution.
/// The first `CLUSTERS` clusters (out of 9 total) are used per layer,
/// truncating the expansion: CLUSTERS=9 for spatial layers, CLUSTERS=1 for pointwise output.
///
/// # Type Parameters
/// - `IN_CHANNELS`: Number of input channels
/// - `OUT_CHANNELS`: Number of output channels
/// - `CLUSTERS`: Number of clusters per channel (1..=9)
///
/// # Weight Layout
/// Weights are in `[IN_CHANNELS * CLUSTERS][OUT_CHANNELS]` layout.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClusterParams<const IN_CHANNELS: usize, const OUT_CHANNELS: usize, const CLUSTERS: usize>
{
    weights: Vec<f32>,
    bias: Vec<f32>,
}

impl<const IN_CHANNELS: usize, const OUT_CHANNELS: usize, const CLUSTERS: usize>
    ClusterParams<IN_CHANNELS, OUT_CHANNELS, CLUSTERS>
{
    /// Workspace stride: number of elements per position in the gathered clusters.
    const INPUTS_PER_POSITION: usize = IN_CHANNELS
        .checked_mul(CLUSTERS)
        .expect("INPUTS_PER_POSITION overflow");

    /// Expected weights length: `IN_CHANNELS * CLUSTERS * OUT_CHANNELS`.
    const WEIGHT_COUNT: usize = Self::INPUTS_PER_POSITION
        .checked_mul(OUT_CHANNELS)
        .expect("WEIGHT_COUNT overflow");

    /// Creates a new `ClusterParams` with validated dimensions.
    ///
    /// # Panics
    ///
    /// Panics if `weights.len() != IN_CHANNELS * CLUSTERS * OUT_CHANNELS` or `bias.len() != OUT_CHANNELS`.
    #[must_use]
    pub fn new(weights: Vec<f32>, bias: Vec<f32>) -> Self {
        assert_eq!(
            weights.len(),
            Self::WEIGHT_COUNT,
            "weights length mismatch: expected {}, got {}",
            Self::WEIGHT_COUNT,
            weights.len()
        );
        assert_eq!(
            bias.len(),
            OUT_CHANNELS,
            "bias length mismatch: expected {OUT_CHANNELS}, got {}",
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
        let std = initial_weight_spread(Self::INPUTS_PER_POSITION);
        let weights: Vec<f32> = (0..Self::WEIGHT_COUNT)
            .map(|_| rng.f32_normal(0.0, std))
            .collect();
        let bias = vec![0.0; OUT_CHANNELS];
        Self::new(weights, bias)
    }

    /// Returns the weight slice in `[IN_CHANNELS * CLUSTERS][OUT_CHANNELS]` layout.
    #[must_use]
    pub fn weights(&self) -> &[f32] {
        assert_eq!(self.weights.len(), Self::WEIGHT_COUNT);
        &self.weights
    }

    /// Returns the bias as a fixed-size array of `OUT_CHANNELS` values.
    #[must_use]
    pub fn bias(&self) -> &[f32; OUT_CHANNELS] {
        self.bias
            .as_slice()
            .try_into()
            .expect("bias length must equal OUT_CHANNELS")
    }

    /// Applies the cluster layer to the input: gather clusters, then dot product.
    ///
    /// All dimensions (`IN_CHANNELS`, `OUT_CHANNELS`, `CLUSTERS`) are encoded in `Self`, ensuring separate
    /// monomorphizations per layer configuration.
    #[must_use]
    pub fn cluster2d(
        &self,
        input: &PositionMap<f32, IN_CHANNELS>,
    ) -> PositionMap<f32, OUT_CHANNELS> {
        let workspace = Self::gather_clusters(input);
        Self::weighted_sums(
            self.weights(),
            self.bias(),
            workspace.iter().map(|clusters| clusters.as_flattened()),
        )
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

    /// Gathers the first `CLUSTERS` cluster sums per input
    /// channel for each position (truncating the expansion). Neighbors outside the board
    /// are zero-padded.
    fn gather_clusters(
        input: &PositionMap<f32, IN_CHANNELS>,
    ) -> PositionMap<[f32; CLUSTERS], IN_CHANNELS> {
        let mut workspace = PositionMap::new([0.0; CLUSTERS]);

        for (pos, clusters) in PositionId::iter().zip(workspace.iter_mut()) {
            // Center first, then the 8 neighbors; off-board neighbors stay zero.
            let mut neighbors = [[0.0; IN_CHANNELS]; 9];
            neighbors[0] = *input.get(pos);
            for (slot, offset) in neighbors[1..].iter_mut().zip(NEIGHBOR_OFFSETS) {
                if let Some(neighbor) = pos.offset(offset) {
                    *slot = *input.get(neighbor);
                }
            }
            for (channel, channel_clusters) in clusters.iter_mut().enumerate() {
                let raw: [f32; 9] = std::array::from_fn(|i| neighbors[i][channel]);
                channel_clusters.copy_from_slice(&cluster_sums(&raw)[..CLUSTERS]);
            }
        }
        workspace
    }

    /// Dot product of each position's `INPUTS_PER_POSITION` gathered values against weights + bias.
    /// `positions` must yield one slice per position in [`PositionId::iter`] order.
    ///
    /// This is an associated function (no `&self`) so that `weights` and `bias` arrive
    /// as independent parameters. LLVM's alias analysis can then prove they do not
    /// alias the freshly allocated output, enabling auto-vectorization.
    #[must_use]
    fn weighted_sums<'a>(
        weights: &[f32],
        bias: &[f32; OUT_CHANNELS],
        positions: impl ExactSizeIterator<Item = &'a [f32]>,
    ) -> PositionMap<f32, OUT_CHANNELS> {
        let (weight_rows, remainder) = weights.as_chunks::<OUT_CHANNELS>();
        assert!(remainder.is_empty());
        assert_eq!(weight_rows.len(), Self::INPUTS_PER_POSITION);
        assert_eq!(positions.len(), PositionId::COUNT);

        let mut output = PositionMap::new(0.0);
        for (output_channels, clusters) in output.iter_mut().zip(positions) {
            assert_eq!(clusters.len(), Self::INPUTS_PER_POSITION);
            *output_channels = *bias;
            for (&cluster_val, weight_row) in clusters.iter().zip(weight_rows) {
                for (out_ch, &weight) in output_channels.iter_mut().zip(weight_row) {
                    *out_ch += cluster_val * weight;
                }
            }
        }
        output
    }
}

/// Pointwise layers (`CLUSTERS == 1`): each position's `IN_CHANNELS` input values are
/// already exactly what gathering would produce, so the dot product can read
/// the input directly.
impl<const IN_CHANNELS: usize, const OUT_CHANNELS: usize>
    ClusterParams<IN_CHANNELS, OUT_CHANNELS, 1>
{
    /// Applies the layer without gathering. Same result as `cluster2d`.
    #[must_use]
    pub fn pointwise2d(
        &self,
        input: &PositionMap<f32, IN_CHANNELS>,
    ) -> PositionMap<f32, OUT_CHANNELS> {
        Self::weighted_sums(
            self.weights(),
            self.bias(),
            input.iter().map(<[f32; IN_CHANNELS]>::as_slice),
        )
    }
}

impl<const IN_CHANNELS: usize, const OUT_CHANNELS: usize, const CLUSTERS: usize> std::fmt::Display
    for ClusterParams<IN_CHANNELS, OUT_CHANNELS, CLUSTERS>
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::nn::CLUSTER_NAMES;
        use crate::nn::format_slice_stats;

        writeln!(
            formatter,
            "Cluster {IN_CHANNELS} -> {OUT_CHANNELS}, {CLUSTERS} clusters"
        )?;

        let weights = self.weights();
        // Weights layout: [IN_CHANNELS * CLUSTERS][OUT_CHANNELS]. Gather per-cluster stats across all
        // input channels: cluster idx lives at rows ch*CLUSTERS + idx for ch in 0..IN_CHANNELS.
        for cluster_idx in 0..CLUSTERS {
            let per_cluster: Vec<f32> = (0..IN_CHANNELS)
                .flat_map(|ch| {
                    let row = ch * CLUSTERS + cluster_idx;
                    let start = row * OUT_CHANNELS;
                    weights[start..start + OUT_CHANNELS].iter().copied()
                })
                .collect();
            let name = CLUSTER_NAMES.get(cluster_idx).unwrap_or(&"?");
            writeln!(
                formatter,
                "  C{cluster_idx} ({name:>9}) {}",
                format_slice_stats(&per_cluster)
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
        let expected_std = initial_weight_spread(2 * 9);
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
        let input = PositionMap::<f32, 1>::new(0.0);
        let weights = vec![0.0f32; 1 * 9 * 2];
        let bias = vec![1.5, -0.5];
        let params = ClusterParams::<1, 2, 9>::new(weights, bias);

        let output = params.cluster2d(&input);

        for pos in PositionId::iter() {
            assert_eq!(output.get(pos), &[1.5, -0.5]);
        }
    }

    #[test]
    fn center_cluster_passes_through_with_one_cluster() {
        // CLUSTERS=1 means only center cluster (cluster[0] = raw center value)
        let center = PositionId::center();
        let mut input = PositionMap::<f32, 1>::new(0.0);
        *input.get_mut(center) = [7.0];

        // Weight=1.0 for the single cluster, bias=0.0
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
        let mut input = PositionMap::<f32, 1>::new(0.0);
        *input.get_mut(corner) = [1.0];

        // Use CLUSTERS=3 (center + ortho + diag) with weight=1 for each cluster
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
        let mut input = PositionMap::<f32, 2>::new(0.0);
        *input.get_mut(center) = [2.0, 3.0];

        // CLUSTERS=1 (pointwise): stride = 2*1 = 2
        // Weights in [INPUTS_PER_POSITION][OUT_CHANNELS] layout, OUT_CHANNELS=1: [w_ch0_f0, w_ch1_f0]
        let weights = vec![1.0, 1.0];
        let bias = vec![0.0];
        let params = ClusterParams::<2, 1, 1>::new(weights, bias);

        let output = params.cluster2d(&input);

        // center clusters: ch0=2.0, ch1=3.0. dot product = 2+3 = 5
        assert_eq!(output.get(center), &[5.0]);
    }

    #[test]
    fn multiple_output_channels() {
        let center = PositionId::center();
        let mut input = PositionMap::<f32, 1>::new(0.0);
        *input.get_mut(center) = [5.0];

        // CLUSTERS=1, IN_CHANNELS=1, OUT_CHANNELS=2: stride=1, weights len = 1*2 = 2
        let weights = vec![1.0, 2.0]; // [cluster0→out0, cluster0→out1]
        let bias = vec![0.0, 10.0];
        let params = ClusterParams::<1, 2, 1>::new(weights, bias);

        let output = params.cluster2d(&input);

        assert_eq!(output.get(center), &[5.0, 20.0]);
    }

    #[test]
    fn successive_calls_produce_independent_results() {
        let mut input1 = PositionMap::<f32, 2>::new(0.0);
        input1.get_mut(PositionId::center())[0] = 5.0;

        let weights = vec![1.0; 2 * 9 * 1];
        let bias = vec![0.0];
        let params = ClusterParams::<2, 1, 9>::new(weights, bias);

        let output1 = params.cluster2d(&input1);

        let input2 = PositionMap::<f32, 2>::new(0.0);
        let output2 = params.cluster2d(&input2);

        // First call should have non-zero at center, second should be zero
        assert_ne!(output1.get(PositionId::center()), &[0.0]);
        assert_eq!(output2.get(PositionId::center()), &[0.0]);
    }

    #[test]
    fn display_shows_dimensions_and_per_cluster_stats() {
        let params = ClusterParams::<2, 4, 9>::new(vec![0.5f32; 2 * 9 * 4], vec![0.0f32; 4]);

        let rendered = params.to_string();

        assert!(
            rendered.starts_with("Cluster 2 -> 4, 9 clusters\n"),
            "{rendered}"
        );
        // Each cluster row aggregates IN_CHANNELS * OUT_CHANNELS = 8 weights.
        assert!(rendered.contains("C0 (   Center) [8]:"), "{rendered}");
        assert!(rendered.contains("C8 ( Diag-180) [8]:"), "{rendered}");
        assert!(rendered.contains("Bias            [4]:"), "{rendered}");
    }

    // Same `<2, 4, 9>` instantiation as above: `cargo llvm-cov` reports
    // coverage per single best instantiation.

    #[test]
    fn display_propagates_header_write_error() {
        use std::fmt::Write as _;
        let params = ClusterParams::<2, 4, 9>::new(vec![0.0f32; 2 * 9 * 4], vec![0.0f32; 4]);
        let mut sink = crate::test_utils::LimitedWriter::with_budget(0);

        assert!(write!(sink, "{params}").is_err());
    }

    #[test]
    fn display_propagates_cluster_row_write_error() {
        use std::fmt::Write as _;
        let params = ClusterParams::<2, 4, 9>::new(vec![0.0f32; 2 * 9 * 4], vec![0.0f32; 4]);
        // Exactly enough budget for the header line, so the first cluster row fails.
        let header = "Cluster 2 -> 4, 9 clusters\n";
        let mut sink = crate::test_utils::LimitedWriter::with_budget(header.len());

        assert!(write!(sink, "{params}").is_err());
    }

    #[test]
    fn pointwise2d_matches_cluster2d_for_f1() {
        let mut rng = fastrand::Rng::with_seed(7);
        let params = ClusterParams::<3, 2, 1>::random(&mut rng);
        let mut input = PositionMap::<f32, 3>::new(0.0);
        for channels in input.iter_mut() {
            for value in channels {
                *value = rng.f32();
            }
        }

        let via_gather = params.cluster2d(&input);
        let pointwise = params.pointwise2d(&input);

        for pos in PositionId::iter() {
            assert_eq!(pointwise.get(pos), via_gather.get(pos));
        }
    }

    #[test]
    fn pointwise2d_computes_weighted_sum_plus_bias() {
        let center = PositionId::center();
        let mut input = PositionMap::<f32, 2>::new(0.0);
        *input.get_mut(center) = [3.0, 4.0];

        // [INPUTS_PER_POSITION=2][OUT_CHANNELS=1]: w_ch0 = 2.0, w_ch1 = 0.5; bias = 1.0
        let params = ClusterParams::<2, 1, 1>::new(vec![2.0, 0.5], vec![1.0]);

        let output = params.pointwise2d(&input);

        assert_eq!(output.get(center), &[9.0]);
        assert_eq!(output.get(pos(0, 0)), &[1.0]);
    }
}
