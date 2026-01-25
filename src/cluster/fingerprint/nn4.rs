use std::collections::BTreeSet;
use std::sync::LazyLock;

use itertools::iproduct;
use rapidhash::RapidHashMap;
use serde::{Deserialize, Serialize};

use crate::cache_id::CacheId;
use crate::cache_repository::CacheRepository;
use crate::cluster::fingerprint::combinations_for_total;
use crate::cluster::offsets;
use crate::cluster::{FingerprintNN4IndexCache, NeighborCounts};
use crate::position_id::PositionId;

type Nn4Tuple = (
    NeighborCounts,
    NeighborCounts,
    NeighborCounts,
    NeighborCounts,
);

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

impl super::Fingerprint for FingerprintNN4 {
    type IndexCache = FingerprintNN4IndexCache;
    const CACHE_ID: CacheId = CacheId::FingerprintNN4Index;
    const CACHE_DEPENDENCIES: &'static [CacheId] = &[CacheId::FingerprintNN4Index];

    fn all() -> &'static [Self] {
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
                    iproduct!(
                        combinations_for_total(nn1_total),
                        combinations_for_total(nn2_total),
                        combinations_for_total(nn3_total),
                        combinations_for_total(nn4_total)
                    )
                    .map(|(nn1, nn2, nn3, nn4)| FingerprintNN4::new(nn1, nn2, nn3, nn4))
                })
                .collect()
        });
        &ALL
    }

    /// Returns the index of this fingerprint in the `all()` array.
    ///
    /// This is used for O(1) priority lookups instead of `HashMap` lookups.
    ///
    /// # Panics
    ///
    /// Panics if the fingerprint is not found in `all()` (should never happen
    /// for valid fingerprints).
    fn index(self) -> usize {
        static INDEX_MAP: LazyLock<RapidHashMap<FingerprintNN4, usize>> = LazyLock::new(|| {
            FingerprintNN4::all()
                .iter()
                .enumerate()
                .map(|(i, &fp)| (fp, i))
                .collect()
        });
        *INDEX_MAP.get(&self).expect("Invalid fingerprint")
    }

    fn get_cache(repo: &CacheRepository) -> Option<&Self::IndexCache> {
        repo.fingerprint_nn4_index()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::fingerprint::Fingerprint;

    #[test]
    fn all_returns_fingerprints() {
        let all = FingerprintNN4::all();

        assert!(!all.is_empty());
    }

    #[test]
    fn serialization_round_trip() {
        let original = FingerprintNN4::new(
            NeighborCounts::new(2, 1, 1),
            NeighborCounts::new(3, 0, 1),
            NeighborCounts::new(0, 2, 2),
            NeighborCounts::new(4, 1, 3),
        );

        let bytes = postcard::to_allocvec(&original).unwrap();
        let deserialized: FingerprintNN4 = postcard::from_bytes(&bytes).unwrap();

        assert_eq!(deserialized, original);
    }
}
