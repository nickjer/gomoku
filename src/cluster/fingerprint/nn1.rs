use std::collections::BTreeSet;
use std::sync::LazyLock;

use rapidhash::RapidHashMap;
use serde::{Deserialize, Serialize};

use crate::cache_repository::CacheRepository;
use crate::cluster::NeighborCounts;
use crate::cluster::fingerprint::combinations_for_total;
use crate::cluster::offsets;
use crate::position_id::PositionId;
use crate::stone::Stone;

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

    /// Calculates the fingerprint for a position given the current cache state.
    ///
    /// # Panics
    ///
    /// Panics if the NN1 cache has not been activated.
    #[must_use]
    pub fn calculate(
        position_id: PositionId,
        current_stone: Stone,
        cache_repo: &CacheRepository,
    ) -> Self {
        let nn1_cache = cache_repo
            .neighbor_nn1()
            .expect("FingerprintNN1 requires neighbor_nn1 cache");
        let counts = nn1_cache.counts_for(position_id, current_stone);
        Self::new(counts)
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
    pub fn index(self) -> u32 {
        static INDEX_MAP: LazyLock<RapidHashMap<FingerprintNN1, u32>> = LazyLock::new(|| {
            FingerprintNN1::all()
                .iter()
                .enumerate()
                .map(|(i, &fp)| (fp, u32::try_from(i).unwrap()))
                .collect()
        });
        *INDEX_MAP.get(&self).expect("Invalid fingerprint")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache_id::CacheId;
    use crate::offset::Offset;

    #[test]
    fn all_returns_fingerprints() {
        let all = FingerprintNN1::all();

        assert!(!all.is_empty());
    }

    #[test]
    fn calculate_returns_fingerprint_for_position() {
        let mut cache_repo = CacheRepository::new();
        cache_repo.activate(CacheId::NeighborNN1);
        let center = PositionId::center();

        let fingerprint = FingerprintNN1::calculate(center, Stone::Black, &cache_repo);

        assert_eq!(fingerprint.counts().player(), 0);
        assert_eq!(fingerprint.counts().opponent(), 0);
        assert_eq!(fingerprint.counts().empty(), 4);
    }

    #[test]
    fn calculate_reflects_placed_stones() {
        let mut cache_repo = CacheRepository::new();
        cache_repo.activate(CacheId::NeighborNN1);
        let center = PositionId::center();

        cache_repo.place(center, Stone::Black);

        let neighbor = center.from_offset(Offset::new(0, 1)).unwrap();
        let fingerprint = FingerprintNN1::calculate(neighbor, Stone::Black, &cache_repo);

        assert_eq!(fingerprint.counts().player(), 1);
        assert_eq!(fingerprint.counts().empty(), 3);
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
