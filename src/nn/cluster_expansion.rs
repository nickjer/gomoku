use crate::offset::Offset;

/// Number of symmetry-distinct neighbor clusters: the center point, single
/// neighbors, and neighbor pairs (cluster expansion up to second order).
///
/// Clusters are ordered by size — this ordering is load-bearing for
/// truncating the expansion (see `ClusterParams`). Keeping the first N
/// clusters gives a natural subset: 1 is the center only (0th order), 3 adds
/// single neighbors (1st order), 9 includes all neighbor pairs (2nd order).
pub const CLUSTER_COUNT: usize = 9;

/// Human-readable names for each cluster index.
pub const CLUSTER_NAMES: [&str; CLUSTER_COUNT] = [
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

/// Offsets to the 8 neighbors, indexed clockwise from North.
///
/// Even indices (0, 2, 4, 6) are orthogonal (distance 1).
/// Odd indices (1, 3, 5, 7) are diagonal (distance √2).
pub const NEIGHBOR_OFFSETS: [Offset; 8] = [
    Offset::new(-1, 0),  // 0: N
    Offset::new(-1, 1),  // 1: NE
    Offset::new(0, 1),   // 2: E
    Offset::new(1, 1),   // 3: SE
    Offset::new(1, 0),   // 4: S
    Offset::new(1, -1),  // 5: SW
    Offset::new(0, -1),  // 6: W
    Offset::new(-1, -1), // 7: NW
];

/// Computes the sum over each of the 9 clusters from a center value and 8 neighbors.
///
/// `center_and_neighbors[0]` is the center value; `center_and_neighbors[1..9]` are the 8 neighbors in clockwise order
/// (N, NE, E, SE, S, SW, W, NW).
///
/// Each cluster sum adds one neighbor pattern over all its D8 rotations/reflections,
/// making the output invariant to D8 transforms of the input.
///
/// | # | Order | Name       | Orbit size |
/// |---|-------|------------|------------|
/// | 0 |   0   | Center     |     1      |
/// | 1 |   1   | Ortho      |     4      |
/// | 2 |   1   | Diag       |     4      |
/// | 3 |   2   | Wedge-45   |     8      |
/// | 4 |   2   | Ortho-90   |     4      |
/// | 5 |   2   | Wedge-135  |     8      |
/// | 6 |   2   | Ortho-180  |     2      |
/// | 7 |   2   | Diag-90    |     4      |
/// | 8 |   2   | Diag-180   |     2      |
#[must_use]
pub fn cluster_sums(center_and_neighbors: &[f32; 9]) -> [f32; CLUSTER_COUNT] {
    let center = center_and_neighbors[0];
    let [n0, n1, n2, n3, n4, n5, n6, n7] = [
        center_and_neighbors[1],
        center_and_neighbors[2],
        center_and_neighbors[3],
        center_and_neighbors[4],
        center_and_neighbors[5],
        center_and_neighbors[6],
        center_and_neighbors[7],
        center_and_neighbors[8],
    ];
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_zeros_produces_all_zeros() {
        let center_and_neighbors = [0.0; 9];
        let clusters = cluster_sums(&center_and_neighbors);
        assert_eq!(clusters, [0.0; CLUSTER_COUNT]);
    }

    #[test]
    fn all_ones_produces_orbit_counts() {
        let center_and_neighbors = [1.0; 9];
        let clusters = cluster_sums(&center_and_neighbors);
        // Each cluster value equals the number of terms in its orbit sum
        assert_eq!(clusters, [1.0, 4.0, 4.0, 8.0, 4.0, 8.0, 2.0, 4.0, 2.0]);
    }

    #[test]
    fn center_only_affects_first_cluster() {
        let center_and_neighbors = [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let clusters = cluster_sums(&center_and_neighbors);
        assert_eq!(clusters[0], 5.0);
        for &cluster in &clusters[1..] {
            assert_eq!(cluster, 0.0);
        }
    }

    #[test]
    fn rotation_by_90_degrees_preserves_clusters() {
        // Original: distinct values for each neighbor
        let original = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        // 90° clockwise rotation maps index i → (i+2) mod 8 for neighbors.
        // So neighbor at index 0 (N) goes to index 2 (E), etc.
        // Rotated neighbors: original[7,8,1,2,3,4,5,6] (center unchanged)
        let rotated = [
            original[0], // center
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

        for (idx, (&orig, &rot)) in clusters_original
            .iter()
            .zip(clusters_rotated.iter())
            .enumerate()
        {
            assert!(
                (orig - rot).abs() < f32::EPSILON,
                "cluster {idx} differs: {orig} vs {rot}"
            );
        }
    }

    #[test]
    fn reflection_preserves_clusters() {
        // Horizontal flip: index i → (8-i) mod 8 for neighbors
        let original = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let reflected = [
            original[0], // center
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
            .zip(clusters_reflected.iter())
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
        // Only neighbor 0 (N) is nonzero
        let center_and_neighbors = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let clusters = cluster_sums(&center_and_neighbors);

        assert_eq!(clusters[0], 0.0); // center
        assert_eq!(clusters[1], 1.0); // ortho sum: n0=1
        assert_eq!(clusters[2], 0.0); // diag sum
        // All 2nd-order clusters are 0 (products with zero neighbors)
        for &cluster in &clusters[3..] {
            assert_eq!(cluster, 0.0);
        }
    }

    #[test]
    fn two_adjacent_neighbors_produce_wedge_cluster() {
        // Neighbors 0 (N) and 1 (NE) are both 1.0
        let center_and_neighbors = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let clusters = cluster_sums(&center_and_neighbors);

        assert_eq!(clusters[1], 1.0); // ortho: n0
        assert_eq!(clusters[2], 1.0); // diag: n1
        assert_eq!(clusters[3], 1.0); // wedge-45: n0*n1 = 1
    }
}
