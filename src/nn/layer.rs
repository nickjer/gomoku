use std::fmt;
use std::marker::PhantomData;

use fastrand_contrib::RngExt;
use serde::{Deserialize, Serialize};

use crate::evolution::genes::EvolvableGenes;
use crate::position_id::PositionId;
use crate::position_map::PositionMap;

use super::format_slice_stats;
use super::neighborhood_encoder::{FlatValues, NeighborhoodEncoder};

/// One step of the network.
///
/// For every position it reads the neighborhood through its encoder,
/// multiplies each value by a learned weight, adds them up, and adds a bias,
/// producing `OUT_CHANNELS` new values for that position. Weights are stored
/// one row per input value so the sum can walk inputs and weights side by side.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Layer<Encoder, const IN_CHANNELS: usize, const OUT_CHANNELS: usize> {
    weights: Vec<f32>,
    bias: Vec<f32>,
    // The encoder is a type, not data, so there is nothing to store.
    #[serde(skip)]
    encoder: PhantomData<Encoder>,
}

impl<Encoder: NeighborhoodEncoder, const IN_CHANNELS: usize, const OUT_CHANNELS: usize>
    Layer<Encoder, IN_CHANNELS, OUT_CHANNELS>
{
    /// How many numbers the encoder produces for one position.
    const INPUTS_PER_POSITION: usize = IN_CHANNELS
        .checked_mul(<Encoder::Neighborhood as FlatValues>::COUNT)
        .expect("INPUTS_PER_POSITION overflow");

    /// One weight per input number per output channel.
    const WEIGHT_COUNT: usize = Self::INPUTS_PER_POSITION
        .checked_mul(OUT_CHANNELS)
        .expect("WEIGHT_COUNT overflow");

    /// Creates a layer from its weights and biases.
    ///
    /// # Panics
    ///
    /// Panics if either length does not match the layer's dimensions.
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
        Self {
            weights,
            bias,
            encoder: PhantomData,
        }
    }

    /// Creates a layer with random weights and zero biases.
    #[must_use]
    pub fn random(rng: &mut fastrand::Rng) -> Self {
        let spread = initial_weight_spread(Self::INPUTS_PER_POSITION);
        let weights = (0..Self::WEIGHT_COUNT)
            .map(|_| rng.f32_normal(0.0, spread))
            .collect();
        Self::new(weights, vec![0.0; OUT_CHANNELS])
    }

    /// The weights, one row of `OUT_CHANNELS` per input number.
    ///
    /// The length check lets the compiler treat the slice as fixed-size, which
    /// makes the loops over it faster.
    #[must_use]
    pub fn weights(&self) -> &[f32] {
        assert_eq!(self.weights.len(), Self::WEIGHT_COUNT);
        &self.weights
    }

    /// The bias added to each output channel.
    #[must_use]
    pub fn bias(&self) -> &[f32; OUT_CHANNELS] {
        self.bias
            .as_slice()
            .try_into()
            .expect("bias length must equal OUT_CHANNELS")
    }

    /// Runs the layer over a whole board.
    ///
    /// Takes the input by value so an encoder that reads the position alone
    /// can pass it straight through without copying.
    #[must_use]
    pub fn apply(&self, input: PositionMap<f32, IN_CHANNELS>) -> PositionMap<f32, OUT_CHANNELS> {
        let encoded = Encoder::encode(input);
        Self::weighted_sums(
            self.weights(),
            self.bias(),
            encoded
                .iter()
                .map(|neighborhoods| Encoder::Neighborhood::as_flat(neighborhoods)),
        )
    }

    /// Multiplies each position's inputs by the weights and adds the bias.
    /// `positions` yields one list of inputs per position, in board order.
    ///
    /// Takes the weights and bias as plain arguments rather than reading them
    /// from `self` so the compiler can see they never overlap the output and
    /// can vectorize the loop.
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
        for (output_channels, inputs) in output.iter_mut().zip(positions) {
            assert_eq!(inputs.len(), Self::INPUTS_PER_POSITION);
            *output_channels = *bias;
            for (&input, weight_row) in inputs.iter().zip(weight_rows) {
                for (out_ch, &weight) in output_channels.iter_mut().zip(weight_row) {
                    *out_ch += input * weight;
                }
            }
        }
        output
    }
}

impl<Encoder: NeighborhoodEncoder, const IN_CHANNELS: usize, const OUT_CHANNELS: usize> fmt::Display
    for Layer<Encoder, IN_CHANNELS, OUT_CHANNELS>
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            formatter,
            "{IN_CHANNELS} -> {OUT_CHANNELS} channels, {}",
            Encoder::default()
        )?;
        Encoder::fmt_weight_stats::<IN_CHANNELS, OUT_CHANNELS>(formatter, self.weights())?;
        write!(formatter, "  Bias    {}", format_slice_stats(self.bias()))
    }
}

impl<Encoder: NeighborhoodEncoder, const IN_CHANNELS: usize, const OUT_CHANNELS: usize>
    EvolvableGenes for Layer<Encoder, IN_CHANNELS, OUT_CHANNELS>
{
    fn gene_groups(&self) -> impl Iterator<Item = &[f32]> {
        [self.weights.as_slice(), self.bias.as_slice()].into_iter()
    }

    fn gene_groups_mut(&mut self) -> impl Iterator<Item = &mut [f32]> {
        [self.weights.as_mut_slice(), self.bias.as_mut_slice()].into_iter()
    }
}

/// How widely to scatter random starting weights: `√(2 / inputs)`.
///
/// Fewer inputs per output means each weight must be larger to matter. This
/// keeps values from fading out or blowing up as they pass through layers
/// that zero their negatives.
fn initial_weight_spread(inputs: usize) -> f32 {
    // Safely downcast to u16 (valid up to 65,535), then losslessly convert to f32.
    let inputs_u16 = u16::try_from(inputs).expect("inputs exceeds u16::MAX");
    (2.0 / f32::from(inputs_u16)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::no_neighbors::NoNeighbors;
    use crate::position::Position;
    use crate::test_utils::LimitedWriter;

    /// Two channels read without neighbors into four outputs: 8 weights.
    type TestLayer = Layer<NoNeighbors, 2, 4>;

    fn pos(row: usize, col: usize) -> PositionId {
        PositionId::from_position(Position::new(row, col))
    }

    #[test]
    fn new_accepts_matching_lengths() {
        let layer = TestLayer::new(vec![0.0; 8], vec![0.0; 4]);

        assert_eq!(layer.weights().len(), 8);
        assert_eq!(layer.bias().len(), 4);
    }

    #[test]
    #[should_panic(expected = "weights length mismatch")]
    fn new_rejects_wrong_weights_length() {
        let _ = TestLayer::new(vec![0.0; 10], vec![0.0; 4]);
    }

    #[test]
    #[should_panic(expected = "bias length mismatch")]
    fn new_rejects_wrong_bias_length() {
        let _ = TestLayer::new(vec![0.0; 8], vec![0.0; 3]);
    }

    #[test]
    fn random_creates_correct_lengths_and_zero_biases() {
        let mut rng = fastrand::Rng::with_seed(42);
        let layer = TestLayer::random(&mut rng);

        assert_eq!(layer.weights().len(), 8);
        assert!(layer.bias().iter().all(|&bias| bias == 0.0));
    }

    #[test]
    fn random_weights_have_the_initial_spread() {
        let mut rng = fastrand::Rng::with_seed(42);
        let layer = Layer::<NoNeighbors, 18, 64>::random(&mut rng);

        let weights = layer.weights();
        let count = f32::from(u16::try_from(weights.len()).unwrap());
        let mean: f32 = weights.iter().sum::<f32>() / count;
        let variance: f32 = weights.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / count;
        let std_dev = variance.sqrt();

        // sqrt(2 / 18) ≈ 0.333
        let expected = initial_weight_spread(18);
        assert!(
            (std_dev - expected).abs() < 0.05,
            "std_dev {std_dev} not close to expected {expected}"
        );
    }

    #[test]
    fn initial_weight_spread_shrinks_with_more_inputs() {
        assert!((initial_weight_spread(2) - 1.0).abs() < f32::EPSILON);
        assert!((initial_weight_spread(8) - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn gene_groups_are_weights_then_bias() {
        let layer = TestLayer::new(vec![0.5; 8], vec![1.5; 4]);

        let gene_groups: Vec<&[f32]> = layer.gene_groups().collect();

        assert_eq!(gene_groups, [&[0.5; 8][..], &[1.5; 4][..]]);
    }

    #[test]
    fn gene_groups_mut_changes_weights_and_bias() {
        let mut layer = TestLayer::new(vec![0.0; 8], vec![0.0; 4]);

        for (gene_group, fill_value) in layer.gene_groups_mut().zip([1.0, 2.0]) {
            gene_group.fill(fill_value);
        }

        assert_eq!(layer.weights(), &[1.0; 8]);
        assert_eq!(layer.bias(), &[2.0; 4]);
    }

    #[test]
    fn bias_only_produces_constant_output() {
        let layer = Layer::<NoNeighbors, 1, 2>::new(vec![0.0; 2], vec![1.5, -0.5]);

        let output = layer.apply(PositionMap::new(0.0));

        for pos in PositionId::iter() {
            assert_eq!(output.get(pos), &[1.5, -0.5]);
        }
    }

    #[test]
    fn apply_computes_weighted_sum_plus_bias() {
        let center = PositionId::center();
        let mut input = PositionMap::<f32, 2>::new(0.0);
        *input.get_mut(center) = [3.0, 4.0];
        // Weights per input channel: 2.0 and 0.5; bias 1.0.
        let layer = Layer::<NoNeighbors, 2, 1>::new(vec![2.0, 0.5], vec![1.0]);

        let output = layer.apply(input);

        assert_eq!(output.get(center), &[9.0]);
        assert_eq!(output.get(pos(0, 0)), &[1.0]);
    }

    #[test]
    fn apply_fans_out_to_multiple_output_channels() {
        let center = PositionId::center();
        let mut input = PositionMap::<f32, 1>::new(0.0);
        *input.get_mut(center) = [5.0];
        let layer = Layer::<NoNeighbors, 1, 2>::new(vec![1.0, 2.0], vec![0.0, 10.0]);

        let output = layer.apply(input);

        assert_eq!(output.get(center), &[5.0, 20.0]);
    }

    #[test]
    fn display_shows_channels_encoder_and_stats() {
        let layer = TestLayer::new(vec![0.5; 8], vec![0.0; 4]);

        let rendered = layer.to_string();

        assert!(
            rendered.starts_with("2 -> 4 channels, no neighbors\n"),
            "{rendered}"
        );
        assert!(rendered.contains("Weights [8]:"), "{rendered}");
        assert!(rendered.contains("Bias    [4]:"), "{rendered}");
    }

    // The error-path tests reuse `TestLayer`: `cargo llvm-cov` reports coverage
    // per single best instantiation.

    #[test]
    fn display_propagates_header_write_error() {
        use std::fmt::Write as _;
        let layer = TestLayer::new(vec![0.0; 8], vec![0.0; 4]);
        let mut sink = LimitedWriter::with_budget(0);

        assert!(write!(sink, "{layer}").is_err());
    }

    #[test]
    fn display_propagates_weights_write_error() {
        use std::fmt::Write as _;
        let layer = TestLayer::new(vec![0.0; 8], vec![0.0; 4]);
        // Exactly enough budget for the header line, so the weights line fails.
        let mut sink = LimitedWriter::with_budget("2 -> 4 channels, no neighbors\n".len());

        assert!(write!(sink, "{layer}").is_err());
    }
}
