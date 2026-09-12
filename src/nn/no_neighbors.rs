use std::fmt;

use serde::{Deserialize, Serialize};

use crate::position_map::PositionMap;

use super::neighborhood_encoder::NeighborhoodEncoder;

/// Looks at the position itself and none of its neighbors.
///
/// The final scoring layer uses this to turn a position's channels into one
/// score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct NoNeighbors;

impl fmt::Display for NoNeighbors {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "no neighbors")
    }
}

impl NeighborhoodEncoder for NoNeighbors {
    type Neighborhood = f32;

    fn encode<const IN_CHANNELS: usize>(
        input: PositionMap<f32, IN_CHANNELS>,
    ) -> PositionMap<f32, IN_CHANNELS> {
        input
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::board_symmetry::BoardSymmetry;
    use crate::position_id::PositionId;

    #[test]
    fn encodes_to_the_same_values() {
        let mut input = PositionMap::<f32, 2>::new(0.0);
        *input.get_mut(PositionId::center()) = [3.0, 4.0];

        let encoded = NoNeighbors::encode(input.clone());

        for pos in PositionId::iter() {
            assert_eq!(encoded.get(pos), input.get(pos));
        }
    }

    #[test]
    fn leaves_the_board_as_it_is() {
        let mut rng = fastrand::Rng::with_seed(42);

        assert_eq!(
            NoNeighbors::board_symmetry(&mut rng),
            BoardSymmetry::Identity
        );
    }

    #[test]
    fn display_names_the_encoder() {
        assert_eq!(NoNeighbors.to_string(), "no neighbors");
    }
}
