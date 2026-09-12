use std::fmt;

use serde::{Deserialize, Serialize};

use crate::offset::Offset;
use crate::position_map::PositionMap;

use super::format_slice_stats;
use super::neighborhood_encoder::NeighborhoodEncoder;

/// How many clusters the expansion has: the position itself, two kinds of
/// single neighbor, and six kinds of neighbor pair.
///
/// Clusters are ordered by size, so keeping the first N of them truncates the
/// expansion: 1 is the position only, 3 adds single neighbors, 9 adds every
/// neighbor pair.
pub const CLUSTER_COUNT: usize = 9;

/// A name for each cluster, in expansion order.
const CLUSTER_NAMES: [&str; CLUSTER_COUNT] = [
    "Center",
    "Ortho",
    "Diag",
    "Wedge-45",
    "Ortho-90",
    "Wedge-135",
    "Ortho-180",
    "Diag-90",
    "Diag-180",
];

/// Sums each cluster over a position and its eight neighbors (center first,
/// then clockwise from north).
///
/// Neighbors that look the same after rotating or reflecting the board belong
/// to the same cluster, so the sums do not change when the board is turned.
///
/// | # | Order | Name       | Terms |
/// |---|-------|------------|-------|
/// | 0 |   0   | Center     |   1   |
/// | 1 |   1   | Ortho      |   4   |
/// | 2 |   1   | Diag       |   4   |
/// | 3 |   2   | Wedge-45   |   8   |
/// | 4 |   2   | Ortho-90   |   4   |
/// | 5 |   2   | Wedge-135  |   8   |
/// | 6 |   2   | Ortho-180  |   2   |
/// | 7 |   2   | Diag-90    |   4   |
/// | 8 |   2   | Diag-180   |   2   |
fn cluster_sums(center_and_neighbors: &[f32; 9]) -> [f32; CLUSTER_COUNT] {
    let [center, n0, n1, n2, n3, n4, n5, n6, n7] = *center_and_neighbors;
    [
        // 0th order (1 cluster)
        center,
        // 1st order (2 clusters)
        n0 + n2 + n4 + n6,
        n1 + n3 + n5 + n7,
        // 2nd order (6 clusters)
        n0 * n1 + n1 * n2 + n2 * n3 + n3 * n4 + n4 * n5 + n5 * n6 + n6 * n7 + n7 * n0,
        n0 * n2 + n2 * n4 + n4 * n6 + n6 * n0,
        n0 * n3 + n1 * n4 + n2 * n5 + n3 * n6 + n4 * n7 + n5 * n0 + n6 * n1 + n7 * n2,
        n0 * n4 + n2 * n6,
        n1 * n3 + n3 * n5 + n5 * n7 + n7 * n1,
        n1 * n5 + n3 * n7,
    ]
}

/// Sums the values around each position by shape: how many neighbors, how
/// many pairs side by side, how many pairs across, and so on.
///
/// Each sum treats all rotations and reflections alike, so the board never
/// needs to be turned before scoring. `CLUSTERS` is how many of the nine
/// cluster sums to keep, smallest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ClusterExpansion<const CLUSTERS: usize>;

impl<const CLUSTERS: usize> fmt::Display for ClusterExpansion<CLUSTERS> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{CLUSTERS}-cluster expansion")
    }
}

impl<const CLUSTERS: usize> NeighborhoodEncoder for ClusterExpansion<CLUSTERS> {
    type Neighborhood = [f32; CLUSTERS];

    fn encode<const IN_CHANNELS: usize>(
        input: PositionMap<f32, IN_CHANNELS>,
    ) -> PositionMap<[f32; CLUSTERS], IN_CHANNELS> {
        const {
            assert!(
                CLUSTERS >= 1 && CLUSTERS <= CLUSTER_COUNT,
                "CLUSTERS must be between 1 and CLUSTER_COUNT"
            );
        }

        input.map_neighborhoods(&Offset::CENTER_AND_NEIGHBORS, 0.0, |square| {
            let mut kept = [0.0; CLUSTERS];
            kept.copy_from_slice(&cluster_sums(square)[..CLUSTERS]);
            kept
        })
    }

    /// One line of statistics per cluster, gathered across all input channels.
    fn fmt_weight_stats<const IN_CHANNELS: usize, const OUT_CHANNELS: usize>(
        formatter: &mut fmt::Formatter<'_>,
        weights: &[f32],
    ) -> fmt::Result {
        // One weight row of OUT_CHANNELS per input value; channel `ch`'s
        // cluster `idx` is row `ch * CLUSTERS + idx`.
        for (idx, name) in CLUSTER_NAMES.iter().take(CLUSTERS).enumerate() {
            let per_cluster: Vec<f32> = (0..IN_CHANNELS)
                .flat_map(|ch| {
                    let start = (ch * CLUSTERS + idx) * OUT_CHANNELS;
                    weights[start..start + OUT_CHANNELS].iter().copied()
                })
                .collect();
            writeln!(
                formatter,
                "  {name:<9} {}",
                format_slice_stats(&per_cluster)
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::layer::Layer;
    use crate::position::Position;
    use crate::position_id::PositionId;
    use crate::test_utils::LimitedWriter;

    fn pos(row: usize, col: usize) -> PositionId {
        PositionId::from_position(Position::new(row, col))
    }

    mod cluster_sums_tests {
        use super::*;

        #[test]
        fn all_zeros_produces_all_zeros() {
            assert_eq!(cluster_sums(&[0.0; 9]), [0.0; CLUSTER_COUNT]);
        }

        #[test]
        fn all_ones_counts_the_terms_of_each_cluster() {
            assert_eq!(
                cluster_sums(&[1.0; 9]),
                [1.0, 4.0, 4.0, 8.0, 4.0, 8.0, 2.0, 4.0, 2.0]
            );
        }

        #[test]
        fn center_only_affects_first_cluster() {
            let clusters = cluster_sums(&[5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

            assert_eq!(clusters[0], 5.0);
            assert!(clusters[1..].iter().all(|&cluster| cluster == 0.0));
        }

        #[test]
        fn rotation_by_90_degrees_preserves_clusters() {
            let original = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
            // A quarter turn moves each neighbor two slots along; the center stays.
            let rotated = [
                original[0],
                original[7],
                original[8],
                original[1],
                original[2],
                original[3],
                original[4],
                original[5],
                original[6],
            ];

            let clusters_original = cluster_sums(&original);
            let clusters_rotated = cluster_sums(&rotated);

            for (idx, (&orig, &rot)) in clusters_original.iter().zip(&clusters_rotated).enumerate()
            {
                assert!(
                    (orig - rot).abs() < f32::EPSILON,
                    "cluster {idx} differs: {orig} vs {rot}"
                );
            }
        }

        #[test]
        fn reflection_preserves_clusters() {
            let original = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
            // A left-right flip reverses the clockwise order around north.
            let reflected = [
                original[0],
                original[1],
                original[8],
                original[7],
                original[6],
                original[5],
                original[4],
                original[3],
                original[2],
            ];

            let clusters_original = cluster_sums(&original);
            let clusters_reflected = cluster_sums(&reflected);

            for (idx, (&orig, &refl)) in clusters_original
                .iter()
                .zip(&clusters_reflected)
                .enumerate()
            {
                assert!(
                    (orig - refl).abs() < f32::EPSILON,
                    "cluster {idx} differs: {orig} vs {refl}"
                );
            }
        }

        #[test]
        fn single_neighbor_contributes_to_correct_clusters() {
            // Only the north neighbor is set.
            let clusters = cluster_sums(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

            assert_eq!(clusters[0], 0.0); // center
            assert_eq!(clusters[1], 1.0); // ortho
            assert_eq!(clusters[2], 0.0); // diag
            // Every pair needs two neighbors.
            assert!(clusters[3..].iter().all(|&cluster| cluster == 0.0));
        }

        #[test]
        fn two_adjacent_neighbors_produce_wedge_cluster() {
            // North and north-east are both set.
            let clusters = cluster_sums(&[0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

            assert_eq!(clusters[1], 1.0); // ortho: N
            assert_eq!(clusters[2], 1.0); // diag: NE
            assert_eq!(clusters[3], 1.0); // wedge-45: N × NE
        }
    }

    #[test]
    fn bias_only_produces_constant_output() {
        let layer = Layer::<ClusterExpansion<9>, 1, 2>::new(vec![0.0; 9 * 2], vec![1.5, -0.5]);

        let output = layer.apply(PositionMap::new(0.0));

        for pos in PositionId::iter() {
            assert_eq!(output.get(pos), &[1.5, -0.5]);
        }
    }

    #[test]
    fn one_cluster_passes_the_center_through() {
        let center = PositionId::center();
        let mut input = PositionMap::<f32, 1>::new(0.0);
        *input.get_mut(center) = [7.0];
        let layer = Layer::<ClusterExpansion<1>, 1, 1>::new(vec![1.0], vec![0.0]);

        let output = layer.apply(input);

        assert_eq!(output.get(center), &[7.0]);
        assert_eq!(output.get(pos(0, 0)), &[0.0]);
    }

    #[test]
    fn off_board_neighbors_count_as_empty() {
        let corner = pos(0, 0);
        let mut input = PositionMap::<f32, 1>::new(0.0);
        *input.get_mut(corner) = [1.0];
        // Center, ortho, and diag clusters, each weighted 1.
        let layer = Layer::<ClusterExpansion<3>, 1, 1>::new(vec![1.0; 3], vec![0.0]);

        let output = layer.apply(input);

        // Only the center contributes: the corner has no set neighbors.
        assert_eq!(output.get(corner), &[1.0]);
    }

    #[test]
    fn multiple_input_channels_are_summed() {
        let center = PositionId::center();
        let mut input = PositionMap::<f32, 2>::new(0.0);
        *input.get_mut(center) = [2.0, 3.0];
        let layer = Layer::<ClusterExpansion<1>, 2, 1>::new(vec![1.0, 1.0], vec![0.0]);

        let output = layer.apply(input);

        assert_eq!(output.get(center), &[5.0]);
    }

    #[test]
    fn multiple_output_channels() {
        let center = PositionId::center();
        let mut input = PositionMap::<f32, 1>::new(0.0);
        *input.get_mut(center) = [5.0];
        let layer = Layer::<ClusterExpansion<1>, 1, 2>::new(vec![1.0, 2.0], vec![0.0, 10.0]);

        let output = layer.apply(input);

        assert_eq!(output.get(center), &[5.0, 20.0]);
    }

    #[test]
    fn neighbor_pairs_reach_the_second_order_clusters() {
        let center = PositionId::center();
        let mut input = PositionMap::<f32, 1>::new(0.0);
        *input.get_mut(pos(6, 7)) = [1.0]; // N of center
        *input.get_mut(pos(6, 8)) = [1.0]; // NE of center
        // Only the wedge-45 cluster is weighted.
        let mut weights = vec![0.0; 9];
        weights[3] = 1.0;
        let layer = Layer::<ClusterExpansion<9>, 1, 1>::new(weights, vec![0.0]);

        let output = layer.apply(input);

        assert_eq!(output.get(center), &[1.0]);
    }

    #[test]
    fn display_names_the_expansion() {
        assert_eq!(ClusterExpansion::<9>.to_string(), "9-cluster expansion");
        assert_eq!(ClusterExpansion::<3>.to_string(), "3-cluster expansion");
    }

    #[test]
    fn layer_display_shows_one_row_per_cluster() {
        let layer = Layer::<ClusterExpansion<9>, 2, 4>::new(vec![0.5; 2 * 9 * 4], vec![0.0; 4]);

        let rendered = layer.to_string();

        assert!(
            rendered.starts_with("2 -> 4 channels, 9-cluster expansion\n"),
            "{rendered}"
        );
        // Each row gathers IN_CHANNELS × OUT_CHANNELS = 8 weights.
        assert!(rendered.contains("  Center    [8]:"), "{rendered}");
        assert!(rendered.contains("  Diag-180  [8]:"), "{rendered}");
        assert!(rendered.contains("  Bias    [4]:"), "{rendered}");
    }

    #[test]
    fn layer_display_propagates_cluster_row_write_error() {
        use std::fmt::Write as _;
        let layer = Layer::<ClusterExpansion<9>, 2, 4>::new(vec![0.0; 2 * 9 * 4], vec![0.0; 4]);
        // Exactly enough budget for the header line, so the first cluster row fails.
        let mut sink = LimitedWriter::with_budget("2 -> 4 channels, 9-cluster expansion\n".len());

        assert!(write!(sink, "{layer}").is_err());
    }
}
