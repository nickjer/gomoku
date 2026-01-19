use std::collections::BTreeSet;
use std::sync::LazyLock;

use serde::{Deserialize, Serialize};

use crate::cache_repository::CacheRepository;
use crate::cluster::NeighborCounts;
use crate::cluster::fingerprint::combinations_for_total;
use crate::cluster::offsets;
use crate::position_id::PositionId;
use crate::stone::Stone;

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

    /// Calculates the fingerprint for a position given the current cache state.
    ///
    /// # Panics
    ///
    /// Panics if the NN1 or NN2 caches have not been activated.
    #[must_use]
    pub fn calculate(
        position_id: PositionId,
        current_stone: Stone,
        cache_repo: &CacheRepository,
    ) -> Self {
        let nn1_cache = cache_repo
            .neighbor_nn1()
            .expect("FingerprintNN2 requires neighbor_nn1 cache");
        let nn2_cache = cache_repo
            .neighbor_nn2()
            .expect("FingerprintNN2 requires neighbor_nn2 cache");

        Self::new(
            nn1_cache.counts_for(position_id, current_stone),
            nn2_cache.counts_for(position_id, current_stone),
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache_id::CacheId;

    #[test]
    fn all_returns_fingerprints() {
        let all = FingerprintNN2::all();

        assert!(!all.is_empty());
    }

    #[test]
    fn calculate_returns_fingerprint_for_position() {
        let mut cache_repo = CacheRepository::new();
        cache_repo.activate(CacheId::NeighborNN1);
        cache_repo.activate(CacheId::NeighborNN2);
        let center = PositionId::center();

        let fingerprint = FingerprintNN2::calculate(center, Stone::Black, &cache_repo);

        assert_eq!(fingerprint.nn1().empty(), 4);
        assert_eq!(fingerprint.nn2().empty(), 4);
    }
}
