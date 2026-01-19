use std::collections::BTreeSet;
use std::sync::LazyLock;

use serde::{Deserialize, Serialize};

use crate::cache_repository::CacheRepository;
use crate::cluster::NeighborCounts;
use crate::cluster::fingerprint::combinations_for_total;
use crate::cluster::offsets;
use crate::position_id::PositionId;
use crate::stone::Stone;

type Nn4Tuple = (NeighborCounts, NeighborCounts, NeighborCounts, NeighborCounts);

/// Fingerprint based on all four neighbor rings (NN1-NN4).
///
/// Serializes as a 4-tuple of neighbor counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(from = "Nn4Tuple", into = "Nn4Tuple")]
pub struct FingerprintNN4 {
    nn1: NeighborCounts,
    nn2: NeighborCounts,
    nn3: NeighborCounts,
    nn4: NeighborCounts,
}

impl From<Nn4Tuple> for FingerprintNN4 {
    fn from((nn1, nn2, nn3, nn4): Nn4Tuple) -> Self {
        Self::new(nn1, nn2, nn3, nn4)
    }
}

impl From<FingerprintNN4> for Nn4Tuple {
    fn from(fp: FingerprintNN4) -> Self {
        (fp.nn1, fp.nn2, fp.nn3, fp.nn4)
    }
}

impl FingerprintNN4 {
    #[must_use]
    pub fn new(
        nn1: NeighborCounts,
        nn2: NeighborCounts,
        nn3: NeighborCounts,
        nn4: NeighborCounts,
    ) -> Self {
        Self { nn1, nn2, nn3, nn4 }
    }

    #[must_use]
    pub fn all() -> &'static [Self] {
        static ALL: LazyLock<Vec<FingerprintNN4>> = LazyLock::new(|| {
            let topology_quads: BTreeSet<(u8, u8, u8, u8)> = PositionId::iter()
                .map(|pos| {
                    (
                        u8::try_from(pos.neighbor_count(&offsets::NN1)).unwrap(),
                        u8::try_from(pos.neighbor_count(&offsets::NN2)).unwrap(),
                        u8::try_from(pos.neighbor_count(&offsets::NN3)).unwrap(),
                        u8::try_from(pos.neighbor_count(&offsets::NN4)).unwrap(),
                    )
                })
                .collect();

            topology_quads
                .into_iter()
                .flat_map(|(nn1_total, nn2_total, nn3_total, nn4_total)| {
                    let nn1_combos = combinations_for_total(nn1_total);
                    let nn2_combos = combinations_for_total(nn2_total);
                    let nn3_combos = combinations_for_total(nn3_total);
                    let nn4_combos = combinations_for_total(nn4_total);

                    nn1_combos.into_iter().flat_map(move |nn1| {
                        let nn2_combos = nn2_combos.clone();
                        let nn3_combos = nn3_combos.clone();
                        let nn4_combos = nn4_combos.clone();
                        nn2_combos.into_iter().flat_map(move |nn2| {
                            let nn3_combos = nn3_combos.clone();
                            let nn4_combos = nn4_combos.clone();
                            nn3_combos.into_iter().flat_map(move |nn3| {
                                nn4_combos
                                    .clone()
                                    .into_iter()
                                    .map(move |nn4| FingerprintNN4::new(nn1, nn2, nn3, nn4))
                            })
                        })
                    })
                })
                .collect()
        });
        &ALL
    }

    /// Calculates the fingerprint for a position given the current cache state.
    ///
    /// # Panics
    ///
    /// Panics if any of the NN1-NN4 caches have not been activated.
    #[must_use]
    pub fn calculate(
        position_id: PositionId,
        current_stone: Stone,
        cache_repo: &CacheRepository,
    ) -> Self {
        let nn1_cache = cache_repo
            .neighbor_nn1()
            .expect("FingerprintNN4 requires neighbor_nn1 cache");
        let nn2_cache = cache_repo
            .neighbor_nn2()
            .expect("FingerprintNN4 requires neighbor_nn2 cache");
        let nn3_cache = cache_repo
            .neighbor_nn3()
            .expect("FingerprintNN4 requires neighbor_nn3 cache");
        let nn4_cache = cache_repo
            .neighbor_nn4()
            .expect("FingerprintNN4 requires neighbor_nn4 cache");

        Self::new(
            nn1_cache.counts_for(position_id, current_stone),
            nn2_cache.counts_for(position_id, current_stone),
            nn3_cache.counts_for(position_id, current_stone),
            nn4_cache.counts_for(position_id, current_stone),
        )
    }

    #[must_use]
    pub fn nn1(&self) -> NeighborCounts {
        self.nn1
    }

    #[must_use]
    pub fn nn2(&self) -> NeighborCounts {
        self.nn2
    }

    #[must_use]
    pub fn nn3(&self) -> NeighborCounts {
        self.nn3
    }

    #[must_use]
    pub fn nn4(&self) -> NeighborCounts {
        self.nn4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache_id::CacheId;

    #[test]
    fn all_returns_fingerprints() {
        let all = FingerprintNN4::all();

        assert!(!all.is_empty());
    }

    #[test]
    fn calculate_returns_fingerprint_for_position() {
        let mut cache_repo = CacheRepository::new();
        cache_repo.activate(CacheId::NeighborNN1);
        cache_repo.activate(CacheId::NeighborNN2);
        cache_repo.activate(CacheId::NeighborNN3);
        cache_repo.activate(CacheId::NeighborNN4);
        let center = PositionId::center();

        let fingerprint = FingerprintNN4::calculate(center, Stone::Black, &cache_repo);

        assert_eq!(fingerprint.nn1().empty(), 4);
        assert_eq!(fingerprint.nn2().empty(), 4);
        assert_eq!(fingerprint.nn3().empty(), 4);
        assert_eq!(fingerprint.nn4().empty(), 8);
    }

    #[test]
    fn serializes_as_4_tuple() {
        let fingerprint = FingerprintNN4::new(
            NeighborCounts::new(1, 2, 1),
            NeighborCounts::new(0, 0, 4),
            NeighborCounts::new(2, 1, 1),
            NeighborCounts::new(3, 2, 3),
        );

        let serialized = ron::to_string(&fingerprint).unwrap();

        assert_eq!(serialized, "((1,2,1),(0,0,4),(2,1,1),(3,2,3))");
    }

    #[test]
    fn deserializes_from_4_tuple() {
        let serialized = "((1, 2, 1), (0, 0, 4), (2, 1, 1), (3, 2, 3))";

        let fingerprint: FingerprintNN4 = ron::from_str(serialized).unwrap();

        assert_eq!(fingerprint.nn1(), NeighborCounts::new(1, 2, 1));
        assert_eq!(fingerprint.nn2(), NeighborCounts::new(0, 0, 4));
        assert_eq!(fingerprint.nn3(), NeighborCounts::new(2, 1, 1));
        assert_eq!(fingerprint.nn4(), NeighborCounts::new(3, 2, 3));
    }

    #[test]
    fn serialization_round_trip() {
        let original = FingerprintNN4::new(
            NeighborCounts::new(2, 1, 1),
            NeighborCounts::new(3, 0, 1),
            NeighborCounts::new(0, 2, 2),
            NeighborCounts::new(4, 1, 3),
        );

        let serialized = ron::to_string(&original).unwrap();
        let deserialized: FingerprintNN4 = ron::from_str(&serialized).unwrap();

        assert_eq!(deserialized, original);
    }
}
