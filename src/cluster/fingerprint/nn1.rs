use std::collections::BTreeSet;
use std::sync::LazyLock;

use rapidhash::RapidHashMap;
use serde::{Deserialize, Serialize};

use crate::cluster::NeighborCounts;
use crate::cluster::fingerprint::combinations_for_total;
use crate::cluster::offsets;
use crate::position_id::PositionId;

type Nn1Tuple = (NeighborCounts,);

/// Fingerprint based on NN1 (orthogonal) neighbors only.
///
/// Serializes as a 1-tuple `((player, opponent, empty),)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(from = "Nn1Tuple", into = "Nn1Tuple")]
pub struct FingerprintNN1 {
    counts: NeighborCounts,
}

impl From<Nn1Tuple> for FingerprintNN1 {
    fn from((counts,): Nn1Tuple) -> Self {
        Self::new(counts)
    }
}

impl From<FingerprintNN1> for Nn1Tuple {
    fn from(fp: FingerprintNN1) -> Self {
        (fp.counts,)
    }
}

impl FingerprintNN1 {
    #[must_use]
    pub fn new(counts: NeighborCounts) -> Self {
        Self { counts }
    }

    #[must_use]
    pub fn all() -> &'static [Self] {
        static ALL: LazyLock<Vec<FingerprintNN1>> = LazyLock::new(|| {
            let totals: BTreeSet<u8> = PositionId::iter()
                .map(|pos| u8::try_from(pos.neighbor_count(&offsets::NN1)).unwrap())
                .collect();

            totals
                .into_iter()
                .flat_map(|total| {
                    combinations_for_total(total)
                        .into_iter()
                        .map(FingerprintNN1::new)
                })
                .collect()
        });
        &ALL
    }

    #[must_use]
    pub fn counts(&self) -> NeighborCounts {
        self.counts
    }

    /// Returns the index of this fingerprint in the `all()` array.
    ///
    /// # Panics
    ///
    /// Panics if the fingerprint is not in the `all()` array (invalid fingerprint).
    #[must_use]
    pub fn index(self) -> usize {
        static INDEX_MAP: LazyLock<RapidHashMap<FingerprintNN1, usize>> = LazyLock::new(|| {
            FingerprintNN1::all()
                .iter()
                .enumerate()
                .map(|(i, &fp)| (fp, i))
                .collect()
        });
        *INDEX_MAP.get(&self).expect("Invalid fingerprint")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_returns_fingerprints() {
        let all = FingerprintNN1::all();

        assert!(!all.is_empty());
    }

    #[test]
    fn serializes_as_1_tuple() {
        let fingerprint = FingerprintNN1::new(NeighborCounts::new(1, 2, 1));

        let serialized = ron::to_string(&fingerprint).unwrap();

        assert_eq!(serialized, "((1,2,1))");
    }

    #[test]
    fn deserializes_from_1_tuple() {
        let serialized = "((1, 2, 1))";

        let fingerprint: FingerprintNN1 = ron::from_str(serialized).unwrap();

        assert_eq!(fingerprint.counts(), NeighborCounts::new(1, 2, 1));
    }

    #[test]
    fn serialization_round_trip() {
        let original = FingerprintNN1::new(NeighborCounts::new(0, 0, 4));

        let serialized = ron::to_string(&original).unwrap();
        let deserialized: FingerprintNN1 = ron::from_str(&serialized).unwrap();

        assert_eq!(deserialized, original);
    }
}
