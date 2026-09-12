use std::fmt;

use serde::{Deserialize, Serialize};

use crate::offset::Offset;
use crate::position_map::PositionMap;

use super::board_symmetry::BoardSymmetry;
use super::neighborhood_encoder::NeighborhoodEncoder;

/// Looks at a position and its eight neighbors, exactly as they are.
/// Neighbors off the edge of the board count as empty.
///
/// A square read this way looks different when the board is rotated or
/// reflected, so this encoder asks for a random board symmetry before every
/// move.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Square3x3;

impl fmt::Display for Square3x3 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "3x3 square")
    }
}

impl NeighborhoodEncoder for Square3x3 {
    type Neighborhood = [f32; Offset::CENTER_AND_NEIGHBORS.len()];

    fn encode<const IN_CHANNELS: usize>(
        input: PositionMap<f32, IN_CHANNELS>,
    ) -> PositionMap<Self::Neighborhood, IN_CHANNELS> {
        input.neighborhoods(&Offset::CENTER_AND_NEIGHBORS, 0.0)
    }

    fn board_symmetry(rng: &mut fastrand::Rng) -> BoardSymmetry {
        BoardSymmetry::random(rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::layer::Layer;
    use crate::position::Position;
    use crate::position_id::PositionId;

    // Slots in a channel's nine values: center, then clockwise from north.
    const CENTER: usize = 0;
    const NORTH: usize = 1;
    const EAST: usize = 3;
    const NORTH_WEST: usize = 8;

    fn pos(row: usize, col: usize) -> PositionId {
        PositionId::from_position(Position::new(row, col))
    }

    /// Input with a single non-zero value at the given position in channel 0.
    fn single_value_input<const CHANNELS: usize>(
        position: PositionId,
        value: f32,
    ) -> PositionMap<f32, CHANNELS> {
        let mut input = PositionMap::new(0.0);
        input.get_mut(position)[0] = value;
        input
    }

    /// Weights that copy one slot of one input channel into every output channel.
    fn weights_reading<const IN: usize, const OUT: usize>(channel: usize, slot: usize) -> Vec<f32> {
        let mut weights = vec![0.0; IN * 9 * OUT];
        for out in 0..OUT {
            weights[(channel * 9 + slot) * OUT + out] = 1.0;
        }
        weights
    }

    #[test]
    fn encode_reads_center_then_neighbors_clockwise() {
        let mut input = PositionMap::<f32, 1>::new(0.0);
        *input.get_mut(pos(7, 7)) = [1.0];
        *input.get_mut(pos(6, 8)) = [2.0]; // NE of (7,7)

        let encoded = Square3x3::encode(input);

        assert_eq!(
            encoded.get(pos(7, 7)),
            &[[1.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        );
        // From (6,8), the value at (7,7) is to the SW.
        assert_eq!(
            encoded.get(pos(6, 8)),
            &[[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
        );
    }

    #[test]
    fn encode_treats_off_board_neighbors_as_empty() {
        let input = PositionMap::<f32, 1>::new(1.0);

        let encoded = Square3x3::encode(input);

        // Only the center, E, SE, and S are on the board.
        assert_eq!(
            encoded.get(pos(0, 0)),
            &[[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]
        );
    }

    #[test]
    fn identity_weights_copy_input() {
        let center = PositionId::center();
        let input = single_value_input::<1>(center, 7.0);
        let layer = Layer::<Square3x3, 1, 1>::new(weights_reading::<1, 1>(0, CENTER), vec![0.0]);

        let output = layer.apply(input);

        assert_eq!(output.get(center), &[7.0]);
        assert_eq!(output.get(pos(0, 0)), &[0.0]);
    }

    #[test]
    fn north_weights_move_value_south() {
        let input = single_value_input::<1>(pos(5, 5), 3.0);
        let layer = Layer::<Square3x3, 1, 1>::new(weights_reading::<1, 1>(0, NORTH), vec![0.0]);

        let output = layer.apply(input);

        assert_eq!(output.get(pos(6, 5)), &[3.0]);
        assert_eq!(output.get(pos(5, 5)), &[0.0]);
    }

    #[test]
    fn summing_weights_sum_the_square() {
        let mut input = PositionMap::<f32, 1>::new(0.0);
        for &p in &[pos(7, 7), pos(7, 8), pos(8, 7), pos(8, 8)] {
            *input.get_mut(p) = [1.0];
        }
        let layer = Layer::<Square3x3, 1, 1>::new(vec![1.0; 9], vec![0.0]);

        let output = layer.apply(input);

        assert_eq!(output.get(pos(7, 7)), &[4.0]);
        assert_eq!(output.get(pos(6, 6)), &[1.0]);
        assert_eq!(output.get(pos(9, 9)), &[1.0]);
    }

    #[test]
    fn multiple_input_channels_are_summed() {
        let center = PositionId::center();
        let mut input = PositionMap::<f32, 2>::new(0.0);
        *input.get_mut(center) = [2.0, 3.0];
        let mut weights = weights_reading::<2, 1>(0, CENTER);
        weights[9 + CENTER] = 1.0;
        let layer = Layer::<Square3x3, 2, 1>::new(weights, vec![0.0]);

        let output = layer.apply(input);

        assert_eq!(output.get(center), &[5.0]);
    }

    #[test]
    fn off_board_neighbors_count_as_empty() {
        let mut input = PositionMap::<f32, 1>::new(0.0);
        for &corner in &[pos(0, 0), pos(0, 14), pos(14, 0), pos(14, 14)] {
            *input.get_mut(corner) = [1.0];
        }
        let layer = Layer::<Square3x3, 1, 1>::new(vec![1.0; 9], vec![0.0]);

        let output = layer.apply(input);

        for &corner in &[pos(0, 0), pos(0, 14), pos(14, 0), pos(14, 14)] {
            assert_eq!(output.get(corner), &[1.0]);
        }
        assert_eq!(output.get(pos(1, 1)), &[1.0]);
    }

    #[test]
    fn weights_and_bias_combine() {
        let mut input = PositionMap::<f32, 1>::new(0.0);
        *input.get_mut(pos(7, 7)) = [2.0];
        *input.get_mut(pos(7, 8)) = [3.0];
        let mut weights = vec![0.0f32; 9];
        weights[CENTER] = 2.0;
        weights[EAST] = 3.0;
        let layer = Layer::<Square3x3, 1, 1>::new(weights, vec![1.0]);

        let output = layer.apply(input);

        // (7,7): 2*2 + 3*3 + 1;  (7,8): 3*2 + 0 + 1
        assert_eq!(output.get(pos(7, 7)), &[14.0]);
        assert_eq!(output.get(pos(7, 8)), &[7.0]);
    }

    #[test]
    fn multiple_output_channels_read_the_same_square() {
        let center = PositionId::center();
        let mut input = PositionMap::<f32, 1>::new(0.0);
        *input.get_mut(center) = [5.0];
        *input.get_mut(pos(6, 6)) = [1.0]; // NW of center
        let mut weights = weights_reading::<1, 2>(0, CENTER);
        weights[NORTH_WEST * 2 + 1] = 2.0;
        let layer = Layer::<Square3x3, 1, 2>::new(weights, vec![0.0, 10.0]);

        let output = layer.apply(input);

        assert_eq!(output.get(center), &[5.0, 17.0]);
    }

    #[test]
    fn board_symmetry_is_random() {
        let mut rng = fastrand::Rng::with_seed(42);
        let mut seen = [false; 8];
        for _ in 0..1000 {
            let symmetry = Square3x3::board_symmetry(&mut rng);
            let idx = BoardSymmetry::ALL
                .iter()
                .position(|&s| s == symmetry)
                .unwrap();
            seen[idx] = true;
        }

        assert!(seen.iter().all(|&s| s), "should see every symmetry");
    }

    #[test]
    fn display_names_the_square() {
        assert_eq!(Square3x3.to_string(), "3x3 square");
    }
}
