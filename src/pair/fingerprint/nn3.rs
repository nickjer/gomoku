use std::collections::BTreeSet;
use std::sync::LazyLock;

use crate::cache_repository::CacheRepository;
use crate::pair::NeighborCounts;
use crate::pair::fingerprint::combinations_for_total;
use crate::pair::offsets;
use crate::position_id::PositionId;
use crate::stone::Stone;

/// Fingerprint based on NN1, NN2, and NN3 neighbors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FingerprintNN3 {
    nn1: NeighborCounts,
    nn2: NeighborCounts,
    nn3: NeighborCounts,
}

impl FingerprintNN3 {
    #[must_use]
    pub fn new(nn1: NeighborCounts, nn2: NeighborCounts, nn3: NeighborCounts) -> Self {
        Self { nn1, nn2, nn3 }
    }

    #[must_use]
    pub fn all() -> &'static [Self] {
        static ALL: LazyLock<Vec<FingerprintNN3>> = LazyLock::new(|| {
            let topology_triples: BTreeSet<(u8, u8, u8)> = PositionId::iter()
                .map(|pos| {
                    (
                        u8::try_from(pos.neighbor_count(&offsets::NN1)).unwrap(),
                        u8::try_from(pos.neighbor_count(&offsets::NN2)).unwrap(),
                        u8::try_from(pos.neighbor_count(&offsets::NN3)).unwrap(),
                    )
                })
                .collect();

            topology_triples
                .into_iter()
                .flat_map(|(nn1_total, nn2_total, nn3_total)| {
                    let nn1_combos = combinations_for_total(nn1_total);
                    let nn2_combos = combinations_for_total(nn2_total);
                    let nn3_combos = combinations_for_total(nn3_total);

                    nn1_combos.into_iter().flat_map(move |nn1| {
                        let nn2_combos = nn2_combos.clone();
                        let nn3_combos = nn3_combos.clone();
                        nn2_combos.into_iter().flat_map(move |nn2| {
                            nn3_combos
                                .clone()
                                .into_iter()
                                .map(move |nn3| FingerprintNN3::new(nn1, nn2, nn3))
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
    /// Panics if the NN1, NN2, or NN3 caches have not been activated.
    #[must_use]
    pub fn calculate(
        position_id: PositionId,
        current_stone: Stone,
        cache_repo: &CacheRepository,
    ) -> Self {
        let nn1_cache = cache_repo
            .neighbor_nn1()
            .expect("FingerprintNN3 requires neighbor_nn1 cache");
        let nn2_cache = cache_repo
            .neighbor_nn2()
            .expect("FingerprintNN3 requires neighbor_nn2 cache");
        let nn3_cache = cache_repo
            .neighbor_nn3()
            .expect("FingerprintNN3 requires neighbor_nn3 cache");

        Self::new(
            nn1_cache.counts_for(position_id, current_stone),
            nn2_cache.counts_for(position_id, current_stone),
            nn3_cache.counts_for(position_id, current_stone),
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache_id::CacheId;

    #[test]
    fn all_returns_fingerprints() {
        let all = FingerprintNN3::all();

        assert!(!all.is_empty());
    }

    #[test]
    fn calculate_returns_fingerprint_for_position() {
        let mut cache_repo = CacheRepository::new();
        cache_repo.activate(CacheId::NeighborNN1);
        cache_repo.activate(CacheId::NeighborNN2);
        cache_repo.activate(CacheId::NeighborNN3);
        let center = PositionId::center();

        let fingerprint = FingerprintNN3::calculate(center, Stone::Black, &cache_repo);

        assert_eq!(fingerprint.nn1().empty(), 4);
        assert_eq!(fingerprint.nn2().empty(), 4);
        assert_eq!(fingerprint.nn3().empty(), 4);
    }
}
