use crate::offset::Offset;

/// Number of D8-equivariant polynomial features (orders 0 through 2).
///
/// Features are ordered by polynomial order — this ordering is load-bearing
/// for F-truncation (see `ClusterParams`). Truncating to the first F features
/// gives a natural subset: F=1 is pointwise (0th order), F=3 adds 1st order,
/// F=9 includes all 2nd order features.
pub const FEATURE_COUNT: usize = 9;

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

/// Computes all 9 D8-equivariant polynomial features from a center value and 8 neighbors.
///
/// `raw[0]` is the center value; `raw[1..9]` are the 8 neighbors in clockwise order
/// (N, NE, E, SE, S, SW, W, NW).
///
/// Each feature is the sum of a pattern over all its D8 rotations/reflections,
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
pub fn compute_features(raw: &[f32; 9]) -> [f32; FEATURE_COUNT] {
    let center = raw[0];
    let [n0, n1, n2, n3, n4, n5, n6, n7] = [
        raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7], raw[8],
    ];
    [
        // 0th order (1 feature)
        center,
        // 1st order (2 features)
        n0 + n2 + n4 + n6,
        n1 + n3 + n5 + n7,
        // 2nd order (6 features)
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
        let raw = [0.0; 9];
        let features = compute_features(&raw);
        assert_eq!(features, [0.0; FEATURE_COUNT]);
    }

    #[test]
    fn all_ones_produces_orbit_counts() {
        let raw = [1.0; 9];
        let features = compute_features(&raw);
        // Each feature value equals the number of terms in its orbit sum
        assert_eq!(features, [1.0, 4.0, 4.0, 8.0, 4.0, 8.0, 2.0, 4.0, 2.0]);
    }

    #[test]
    fn center_only_affects_first_feature() {
        let raw = [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let features = compute_features(&raw);
        assert_eq!(features[0], 5.0);
        for &feature in &features[1..] {
            assert_eq!(feature, 0.0);
        }
    }

    #[test]
    fn rotation_by_90_degrees_preserves_features() {
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

        let features_original = compute_features(&original);
        let features_rotated = compute_features(&rotated);

        for (idx, (&orig, &rot)) in features_original
            .iter()
            .zip(features_rotated.iter())
            .enumerate()
        {
            assert!(
                (orig - rot).abs() < f32::EPSILON,
                "feature {idx} differs: {orig} vs {rot}"
            );
        }
    }

    #[test]
    fn reflection_preserves_features() {
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

        let features_original = compute_features(&original);
        let features_reflected = compute_features(&reflected);

        for (idx, (&orig, &refl)) in features_original
            .iter()
            .zip(features_reflected.iter())
            .enumerate()
        {
            assert!(
                (orig - refl).abs() < f32::EPSILON,
                "feature {idx} differs: {orig} vs {refl}"
            );
        }
    }

    #[test]
    fn single_neighbor_contributes_to_correct_features() {
        // Only neighbor 0 (N) is nonzero
        let raw = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let features = compute_features(&raw);

        assert_eq!(features[0], 0.0); // center
        assert_eq!(features[1], 1.0); // ortho sum: n0=1
        assert_eq!(features[2], 0.0); // diag sum
        // All 2nd-order features are 0 (products with zero neighbors)
        for &feature in &features[3..] {
            assert_eq!(feature, 0.0);
        }
    }

    #[test]
    fn two_adjacent_neighbors_produce_wedge_feature() {
        // Neighbors 0 (N) and 1 (NE) are both 1.0
        let raw = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let features = compute_features(&raw);

        assert_eq!(features[1], 1.0); // ortho: n0
        assert_eq!(features[2], 1.0); // diag: n1
        assert_eq!(features[3], 1.0); // wedge-45: n0*n1 = 1
    }
}
