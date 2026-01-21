use std::collections::BTreeSet;
use std::sync::LazyLock;

use rapidhash::RapidHashMap;
use serde::{Deserialize, Serialize};

use crate::cluster::NeighborCounts;
use crate::cluster::fingerprint::combinations_for_total;
use crate::cluster::offsets;
use crate::position_id::PositionId;

type Nn2Tuple = (NeighborCounts, NeighborCounts);

/// Fingerprint based on NN1 and NN2 neighbors.
///
/// Serializes as a 2-tuple of neighbor counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(from = "Nn2Tuple", into = "Nn2Tuple")]
pub struct FingerprintNN2 {
    nn1: NeighborCounts,
    nn2: NeighborCounts,
}

impl From<Nn2Tuple> for FingerprintNN2 {
    fn from((nn1, nn2): Nn2Tuple) -> Self {
        Self::new(nn1, nn2)
    }
}

impl From<FingerprintNN2> for Nn2Tuple {
    fn from(fp: FingerprintNN2) -> Self {
        (fp.nn1, fp.nn2)
    }
}

impl FingerprintNN2 {
    #[must_use]
    pub fn new(nn1: NeighborCounts, nn2: NeighborCounts) -> Self {
        Self { nn1, nn2 }
    }

    #[must_use]
    pub fn all() -> &'static [Self] {
        static ALL: LazyLock<Vec<FingerprintNN2>> = LazyLock::new(|| {
            let topology_pairs: BTreeSet<(u8, u8)> = PositionId::iter()
                .map(|pos| {
                    (
                        u8::try_from(pos.neighbor_count(&offsets::NN1)).unwrap(),
                        u8::try_from(pos.neighbor_count(&offsets::NN2)).unwrap(),
                    )
                })
                .collect();

            topology_pairs
                .into_iter()
                .flat_map(|(nn1_total, nn2_total)| {
                    let nn1_combos = combinations_for_total(nn1_total);
                    let nn2_combos = combinations_for_total(nn2_total);

                    nn1_combos.into_iter().flat_map(move |nn1| {
                        nn2_combos
                            .clone()
                            .into_iter()
                            .map(move |nn2| FingerprintNN2::new(nn1, nn2))
                    })
                })
                .collect()
        });
        &ALL
    }

    #[must_use]
    pub fn nn1(&self) -> NeighborCounts {
        self.nn1
    }

    #[must_use]
    pub fn nn2(&self) -> NeighborCounts {
        self.nn2
    }

    /// Returns the index of this fingerprint in the `all()` array.
    ///
    /// # Panics
    ///
    /// Panics if the fingerprint is not in the `all()` array (invalid fingerprint).
    #[must_use]
    pub fn index(self) -> usize {
        static INDEX_MAP: LazyLock<RapidHashMap<FingerprintNN2, usize>> = LazyLock::new(|| {
            FingerprintNN2::all()
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
        let all = FingerprintNN2::all();

        assert!(!all.is_empty());
    }

    #[test]
    fn serialization_round_trip() {
        let original =
            FingerprintNN2::new(NeighborCounts::new(2, 1, 1), NeighborCounts::new(3, 0, 1));

        let bytes = postcard::to_allocvec(&original).unwrap();
        let deserialized: FingerprintNN2 = postcard::from_bytes(&bytes).unwrap();

        assert_eq!(deserialized, original);
    }
}
