use std::collections::BTreeSet;
use std::sync::LazyLock;

use crate::cache_repository::CacheRepository;
use crate::pair::fingerprint::combinations_for_total;
use crate::pair::offsets;
use crate::pair::NeighborCounts;
use crate::position_id::PositionId;
use crate::stone::Stone;

/// Fingerprint based on NN1 (orthogonal) neighbors only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FingerprintNN1 {
    counts: NeighborCounts,
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
}
