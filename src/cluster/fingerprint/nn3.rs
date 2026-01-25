use std::collections::BTreeSet;
use std::sync::LazyLock;

use rapidhash::RapidHashMap;
use serde::{Deserialize, Serialize};

use crate::cache_id::CacheId;
use crate::cache_repository::CacheRepository;
use crate::cluster::fingerprint::combinations_for_total;
use crate::cluster::offsets;
use crate::cluster::{FingerprintNN3IndexCache, NeighborCounts};
use crate::position_id::PositionId;

type Nn3Tuple = (NeighborCounts, NeighborCounts, NeighborCounts);

/// Fingerprint based on NN1, NN2, and NN3 neighbors.
///
/// Serializes as a 3-tuple of neighbor counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(from = "Nn3Tuple", into = "Nn3Tuple")]
pub struct FingerprintNN3 {
    nn1: NeighborCounts,
    nn2: NeighborCounts,
    nn3: NeighborCounts,
}

impl From<Nn3Tuple> for FingerprintNN3 {
    fn from((nn1, nn2, nn3): Nn3Tuple) -> Self {
        Self::new(nn1, nn2, nn3)
    }
}

impl From<FingerprintNN3> for Nn3Tuple {
    fn from(fp: FingerprintNN3) -> Self {
        (fp.nn1, fp.nn2, fp.nn3)
    }
}

impl FingerprintNN3 {
    #[must_use]
    pub fn new(nn1: NeighborCounts, nn2: NeighborCounts, nn3: NeighborCounts) -> Self {
        Self { nn1, nn2, nn3 }
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

impl super::Fingerprint for FingerprintNN3 {
    type IndexCache = FingerprintNN3IndexCache;
    const CACHE_ID: CacheId = CacheId::FingerprintNN3Index;
    const CACHE_DEPENDENCIES: &'static [CacheId] = &[CacheId::FingerprintNN3Index];

    fn all() -> &'static [Self] {
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

    /// Returns the index of this fingerprint in the `all()` array.
    ///
    /// # Panics
    ///
    /// Panics if the fingerprint is not in the `all()` array (invalid fingerprint).
    fn index(self) -> usize {
        static INDEX_MAP: LazyLock<RapidHashMap<FingerprintNN3, usize>> = LazyLock::new(|| {
            FingerprintNN3::all()
                .iter()
                .enumerate()
                .map(|(i, &fp)| (fp, i))
                .collect()
        });
        *INDEX_MAP.get(&self).expect("Invalid fingerprint")
    }

    fn get_cache(repo: &CacheRepository) -> Option<&Self::IndexCache> {
        repo.fingerprint_nn3_index()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::fingerprint::Fingerprint;

    #[test]
    fn all_returns_fingerprints() {
        let all = FingerprintNN3::all();

        assert!(!all.is_empty());
    }

    #[test]
    fn serialization_round_trip() {
        let original = FingerprintNN3::new(
            NeighborCounts::new(2, 1, 1),
            NeighborCounts::new(3, 0, 1),
            NeighborCounts::new(0, 2, 2),
        );

        let bytes = postcard::to_allocvec(&original).unwrap();
        let deserialized: FingerprintNN3 = postcard::from_bytes(&bytes).unwrap();

        assert_eq!(deserialized, original);
    }
}
