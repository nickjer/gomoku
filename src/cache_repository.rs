use crate::cache_id::CacheId;
use crate::cluster::{
    FingerprintNN1IndexCache, FingerprintNN2IndexCache, FingerprintNN3IndexCache,
    FingerprintNN4IndexCache,
};
use crate::position_id::PositionId;
use crate::stone::Stone;

/// Repository for lazily-activated fingerprint index caches.
pub struct CacheRepository {
    nn1: Option<FingerprintNN1IndexCache>,
    nn2: Option<FingerprintNN2IndexCache>,
    nn3: Option<FingerprintNN3IndexCache>,
    nn4: Option<FingerprintNN4IndexCache>,
}

impl CacheRepository {
    #[must_use]
    pub fn new() -> Self {
        Self {
            nn1: None,
            nn2: None,
            nn3: None,
            nn4: None,
        }
    }

    pub fn activate(&mut self, cache_id: CacheId) {
        match cache_id {
            CacheId::FingerprintNN1Index => {
                self.nn1.get_or_insert_with(FingerprintNN1IndexCache::new);
            }
            CacheId::FingerprintNN2Index => {
                self.nn2.get_or_insert_with(FingerprintNN2IndexCache::new);
            }
            CacheId::FingerprintNN3Index => {
                self.nn3.get_or_insert_with(FingerprintNN3IndexCache::new);
            }
            CacheId::FingerprintNN4Index => {
                self.nn4.get_or_insert_with(FingerprintNN4IndexCache::new);
            }
        }
    }

    pub fn place(&mut self, position_id: PositionId, stone: Stone) {
        if let Some(cache) = &mut self.nn1 {
            cache.place(position_id, stone);
        }
        if let Some(cache) = &mut self.nn2 {
            cache.place(position_id, stone);
        }
        if let Some(cache) = &mut self.nn3 {
            cache.place(position_id, stone);
        }
        if let Some(cache) = &mut self.nn4 {
            cache.place(position_id, stone);
        }
    }

    #[must_use]
    pub fn fingerprint_nn1_index(&self) -> Option<&FingerprintNN1IndexCache> {
        self.nn1.as_ref()
    }

    #[must_use]
    pub fn fingerprint_nn2_index(&self) -> Option<&FingerprintNN2IndexCache> {
        self.nn2.as_ref()
    }

    #[must_use]
    pub fn fingerprint_nn3_index(&self) -> Option<&FingerprintNN3IndexCache> {
        self.nn3.as_ref()
    }

    #[must_use]
    pub fn fingerprint_nn4_index(&self) -> Option<&FingerprintNN4IndexCache> {
        self.nn4.as_ref()
    }
}

impl Default for CacheRepository {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::NeighborCounts;
    use crate::cluster::fingerprint::FingerprintNN1;
    use crate::offset::Offset;

    #[test]
    fn caches_are_none_by_default() {
        let repo = CacheRepository::new();

        assert!(repo.fingerprint_nn1_index().is_none());
        assert!(repo.fingerprint_nn2_index().is_none());
        assert!(repo.fingerprint_nn3_index().is_none());
        assert!(repo.fingerprint_nn4_index().is_none());
    }

    #[test]
    fn activate_creates_cache() {
        let mut repo = CacheRepository::new();

        repo.activate(CacheId::FingerprintNN1Index);

        assert!(repo.fingerprint_nn1_index().is_some());
        assert!(repo.fingerprint_nn2_index().is_none());
    }

    #[test]
    fn place_updates_activated_caches() {
        let mut repo = CacheRepository::new();
        repo.activate(CacheId::FingerprintNN1Index);
        let center = PositionId::center();

        repo.place(center, Stone::Black);

        let cache = repo.fingerprint_nn1_index().unwrap();
        let neighbor = center.from_offset(Offset::new(0, 1)).unwrap();
        // After placing black at center, neighbor sees 1 player, 0 opponent, 3 empty
        let expected = FingerprintNN1::new(NeighborCounts::new(1, 0, 3)).index();
        assert_eq!(cache.get(neighbor, Stone::Black), expected);
    }
}
