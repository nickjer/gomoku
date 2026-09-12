use std::fmt;

use crate::position_map::PositionMap;

use super::board_symmetry::BoardSymmetry;
use super::format_slice_stats;

/// Numbers that can be read as one long list: a single number or a row of them.
///
/// A layer reads its inputs this way to multiply them against its weights.
pub trait FlatValues: Copy {
    /// How many numbers one `Self` holds.
    const COUNT: usize;

    /// Reads a slice of `Self` as one list of numbers.
    fn as_flat(items: &[Self]) -> &[f32];
}

impl FlatValues for f32 {
    const COUNT: usize = 1;

    fn as_flat(items: &[f32]) -> &[f32] {
        items
    }
}

impl<const N: usize> FlatValues for [f32; N] {
    const COUNT: usize = N;

    fn as_flat(items: &[[f32; N]]) -> &[f32] {
        items.as_flattened()
    }
}

/// The way a layer looks at the board around each position.
///
/// Each encoder decides which nearby intersections a layer reads and how their
/// values are combined before weighting. The encoder is chosen as a type, so a
/// network's shape is fixed when the program is compiled.
pub trait NeighborhoodEncoder: Copy + Default + fmt::Display {
    /// What one encoder reads from one channel around one position.
    type Neighborhood: FlatValues;

    /// Reads the neighborhood of every position, for every input channel.
    fn encode<const IN_CHANNELS: usize>(
        input: PositionMap<f32, IN_CHANNELS>,
    ) -> PositionMap<Self::Neighborhood, IN_CHANNELS>;

    /// Which rotation or reflection to apply to the board before scoring it.
    /// The chosen move is mapped back afterwards.
    ///
    /// Encoders that give different answers for a rotated board return a
    /// random symmetry, so the network learns every orientation equally.
    /// Encoders that already treat all orientations alike keep the default:
    /// leave the board as it is.
    fn board_symmetry(_rng: &mut fastrand::Rng) -> BoardSymmetry {
        BoardSymmetry::Identity
    }

    /// Writes a summary of a layer's weights for the `inspect` command.
    ///
    /// The default is one line of statistics over all weights.
    fn fmt_weight_stats<const IN_CHANNELS: usize, const OUT_CHANNELS: usize>(
        formatter: &mut fmt::Formatter<'_>,
        weights: &[f32],
    ) -> fmt::Result {
        writeln!(formatter, "  Weights {}", format_slice_stats(weights))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_values_count_matches_the_row_length() {
        assert_eq!(<f32 as FlatValues>::COUNT, 1);
        assert_eq!(<[f32; 9] as FlatValues>::COUNT, 9);
    }

    #[test]
    fn flat_values_as_flat_joins_rows_in_order() {
        assert_eq!(f32::as_flat(&[1.0, 2.0]), &[1.0, 2.0]);
        assert_eq!(
            <[f32; 2]>::as_flat(&[[1.0, 2.0], [3.0, 4.0]]),
            &[1.0, 2.0, 3.0, 4.0]
        );
    }
}
