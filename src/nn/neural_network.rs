use std::fmt;

use serde::{Deserialize, Serialize};

use crate::evolution::genes::EvolvableGenes;
use crate::position_map::PositionMap;

use super::layer::Layer;
use super::neighborhood_encoder::NeighborhoodEncoder;
use super::no_neighbors::NoNeighbors;
use super::stone_channels::STONE_CHANNELS;

/// A neural network that gives every position on the board a score.
///
/// The board layer reads the stones, the middle layers refine what the
/// previous layer found, and the scoring layer turns each position's channels
/// into a single number. Negative values are zeroed between layers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NeuralNetwork<Encoder, const CHANNELS: usize, const LAYERS: usize> {
    board_layer: Layer<Encoder, { STONE_CHANNELS }, CHANNELS>,
    middle_layers: Vec<Layer<Encoder, CHANNELS, CHANNELS>>,
    scoring_layer: Layer<NoNeighbors, CHANNELS, 1>,
}

impl<Encoder: NeighborhoodEncoder, const CHANNELS: usize, const LAYERS: usize>
    NeuralNetwork<Encoder, CHANNELS, LAYERS>
{
    /// Creates a network with random weights and zero biases.
    ///
    /// A network with zero layers will not compile.
    #[must_use]
    pub fn random(rng: &mut fastrand::Rng) -> Self {
        const { assert!(LAYERS >= 1, "a network needs at least one layer") }

        Self {
            board_layer: Layer::random(rng),
            middle_layers: (1..LAYERS).map(|_| Layer::random(rng)).collect(),
            scoring_layer: Layer::random(rng),
        }
    }

    /// Scores every position on the board from its stone channels.
    #[must_use]
    pub fn score_positions(
        &self,
        stone_channels: PositionMap<f32, STONE_CHANNELS>,
    ) -> PositionMap<f32, 1> {
        let mut activations = self.board_layer.apply(stone_channels);
        zero_negatives_inplace(activations.as_flattened_mut());

        for middle in &self.middle_layers {
            activations = middle.apply(activations);
            zero_negatives_inplace(activations.as_flattened_mut());
        }

        self.scoring_layer.apply(activations)
    }

    /// Total number of weights and biases.
    fn parameter_count(&self) -> usize {
        self.gene_groups().map(<[f32]>::len).sum()
    }
}

impl<Encoder: NeighborhoodEncoder, const CHANNELS: usize, const LAYERS: usize> fmt::Display
    for NeuralNetwork<Encoder, CHANNELS, LAYERS>
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            formatter,
            "Neural network: {}, {CHANNELS} channels, {LAYERS} layers",
            Encoder::default()
        )?;
        writeln!(formatter)?;
        writeln!(formatter, "Layer 1 (board): {}", self.board_layer)?;
        for (idx, layer) in self.middle_layers.iter().enumerate() {
            writeln!(formatter, "Layer {} (middle): {layer}", idx + 2)?;
        }
        let scoring_idx = 2 + self.middle_layers.len();
        writeln!(
            formatter,
            "Layer {scoring_idx} (scoring): {}",
            self.scoring_layer
        )?;
        write!(formatter, "Total parameters: {}", self.parameter_count())
    }
}

impl<Encoder: NeighborhoodEncoder, const CHANNELS: usize, const LAYERS: usize> EvolvableGenes
    for NeuralNetwork<Encoder, CHANNELS, LAYERS>
{
    fn gene_groups(&self) -> impl Iterator<Item = &[f32]> {
        self.board_layer
            .gene_groups()
            .chain(self.middle_layers.iter().flat_map(Layer::gene_groups))
            .chain(self.scoring_layer.gene_groups())
    }

    fn gene_groups_mut(&mut self) -> impl Iterator<Item = &mut [f32]> {
        self.board_layer
            .gene_groups_mut()
            .chain(
                self.middle_layers
                    .iter_mut()
                    .flat_map(Layer::gene_groups_mut),
            )
            .chain(self.scoring_layer.gene_groups_mut())
    }
}

/// Replaces every negative value with zero, in place.
fn zero_negatives_inplace(data: &mut [f32]) {
    for val in data.iter_mut() {
        *val = val.max(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::square3x3::Square3x3;
    use crate::position_id::PositionId;
    use crate::test_utils::LimitedWriter;

    // 3×3 square, 8 channels, 2 layers.
    type TestNetwork = NeuralNetwork<Square3x3, 8, 2>;

    #[test]
    fn random_creates_correct_structure() {
        let mut rng = fastrand::Rng::with_seed(42);
        let network = TestNetwork::random(&mut rng);

        // Board: 2 stone channels × 9 square cells -> 8 channels
        assert_eq!(network.board_layer.weights().len(), 2 * 9 * 8);
        assert_eq!(network.board_layer.bias().len(), 8);

        // Middle: LAYERS - 1 = 1 layer, 8 channels × 9 cells -> 8 channels
        assert_eq!(network.middle_layers.len(), 1);
        assert_eq!(network.middle_layers[0].weights().len(), 8 * 9 * 8);
        assert_eq!(network.middle_layers[0].bias().len(), 8);

        // Scoring: 8 channels, no neighbors -> 1 score
        assert_eq!(network.scoring_layer.weights().len(), 8);
        assert_eq!(network.scoring_layer.bias().len(), 1);
    }

    #[test]
    fn random_single_layer_has_no_middle_layers() {
        let mut rng = fastrand::Rng::with_seed(42);
        let network = NeuralNetwork::<Square3x3, 8, 1>::random(&mut rng);

        assert!(network.middle_layers.is_empty());
    }

    #[test]
    fn biases_initialized_to_zero() {
        let mut rng = fastrand::Rng::with_seed(42);
        let network = TestNetwork::random(&mut rng);

        assert!(network.board_layer.bias().iter().all(|&b| b == 0.0));
        assert!(network.middle_layers[0].bias().iter().all(|&b| b == 0.0));
        assert!(network.scoring_layer.bias().iter().all(|&b| b == 0.0));
    }

    #[test]
    fn score_positions_gives_every_position_a_finite_score() {
        let mut rng = fastrand::Rng::with_seed(42);
        let network = TestNetwork::random(&mut rng);
        let mut stone_channels = PositionMap::new(0.0);
        *stone_channels.get_mut(PositionId::center()) = [1.0, 0.0];

        let scores = network.score_positions(stone_channels);

        for pos in PositionId::iter() {
            let [score] = *scores.get(pos);
            assert!(score.is_finite(), "score at {pos:?} is {score}");
        }
    }

    #[test]
    fn gene_groups_walk_every_layer_in_order() {
        let mut rng = fastrand::Rng::with_seed(42);
        let network = TestNetwork::random(&mut rng);

        let group_lengths: Vec<usize> = network.gene_groups().map(<[f32]>::len).collect();

        // Board weights and bias, middle weights and bias, scoring weights and bias.
        assert_eq!(group_lengths, [2 * 9 * 8, 8, 8 * 9 * 8, 8, 8, 1]);
    }

    #[test]
    fn gene_groups_mut_reaches_the_same_values() {
        let mut rng = fastrand::Rng::with_seed(42);
        let mut network = TestNetwork::random(&mut rng);

        for (gene_group, fill_value) in network
            .gene_groups_mut()
            .zip([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        {
            gene_group.fill(fill_value);
        }

        assert!(
            network
                .board_layer
                .weights()
                .iter()
                .all(|&weight| weight == 0.0)
        );
        assert!(network.board_layer.bias().iter().all(|&bias| bias == 1.0));
        assert!(
            network.middle_layers[0]
                .weights()
                .iter()
                .all(|&weight| weight == 2.0)
        );
        assert!(
            network.middle_layers[0]
                .bias()
                .iter()
                .all(|&bias| bias == 3.0)
        );
        assert!(
            network
                .scoring_layer
                .weights()
                .iter()
                .all(|&weight| weight == 4.0)
        );
        assert_eq!(network.scoring_layer.bias(), &[5.0]);
    }

    #[test]
    fn clone_produces_equal_network() {
        let mut rng = fastrand::Rng::with_seed(42);
        let network = TestNetwork::random(&mut rng);

        assert_eq!(network, network.clone());
    }

    #[test]
    fn display_lists_layers_and_total_parameters() {
        let mut rng = fastrand::Rng::with_seed(42);
        let network = TestNetwork::random(&mut rng);

        let rendered = network.to_string();

        assert!(
            rendered.starts_with("Neural network: 3x3 square, 8 channels, 2 layers\n"),
            "{rendered}"
        );
        assert!(
            rendered.contains("Layer 1 (board): 2 -> 8 channels, 3x3 square"),
            "{rendered}"
        );
        assert!(
            rendered.contains("Layer 2 (middle): 8 -> 8 channels, 3x3 square"),
            "{rendered}"
        );
        assert!(
            rendered.contains("Layer 3 (scoring): 8 -> 1 channels, no neighbors"),
            "{rendered}"
        );
        // (2*9*8 + 8) + (8*9*8 + 8) + (8 + 1)
        assert!(rendered.ends_with("Total parameters: 745"), "{rendered}");
    }

    #[test]
    fn display_propagates_write_errors_at_every_line() {
        use std::fmt::Write as _;
        let mut rng = fastrand::Rng::with_seed(42);
        let network = TestNetwork::random(&mut rng);
        let full = network.to_string();

        // Cut the budget before the first line and at the end of every line
        // but the last; each must fail.
        let line_ends = full.match_indices('\n').map(|(idx, _)| idx + 1);
        for budget in std::iter::once(0).chain(line_ends) {
            let mut sink = LimitedWriter::with_budget(budget);
            assert!(write!(sink, "{network}").is_err(), "budget {budget}");
        }
    }

    mod zero_negatives_tests {
        use super::*;

        #[test]
        fn positive_values_unchanged() {
            let mut data = vec![1.0, 2.5, 0.001];
            zero_negatives_inplace(&mut data);
            assert_eq!(data, vec![1.0, 2.5, 0.001]);
        }

        #[test]
        fn negative_values_become_zero() {
            let mut data = vec![-1.0, -0.001, -100.0];
            zero_negatives_inplace(&mut data);
            assert_eq!(data, vec![0.0, 0.0, 0.0]);
        }

        #[test]
        fn mixed_values() {
            let mut data = vec![-2.0, 0.0, 3.0, -0.5, 1.0];
            zero_negatives_inplace(&mut data);
            assert_eq!(data, vec![0.0, 0.0, 3.0, 0.0, 1.0]);
        }

        #[test]
        fn empty_slice() {
            let mut data: Vec<f32> = vec![];
            zero_negatives_inplace(&mut data);
            assert!(data.is_empty());
        }
    }
}
