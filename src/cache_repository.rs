use crate::cache_id::CacheId;
use crate::neighbor_cache::NeighborCache;
use crate::neighbor_counts_cache::NeighborCountsCache;
use crate::position_id::PositionId;
use crate::stone::Stone;

/// Repository for lazily-activated neighbor counts caches.
#[allow(clippy::struct_field_names)]
pub struct CacheRepository {
    neighbor_nn1: Option<NeighborCountsCache>,
    neighbor_nn2: Option<NeighborCountsCache>,
    neighbor_nn3: Option<NeighborCountsCache>,
    neighbor_nn4: Option<NeighborCountsCache>,
}

impl CacheRepository {
    #[must_use]
    pub fn new() -> Self {
        Self {
            neighbor_nn1: None,
            neighbor_nn2: None,
            neighbor_nn3: None,
            neighbor_nn4: None,
        }
    }

    pub fn activate(&mut self, cache_id: CacheId) {
        match cache_id {
            CacheId::NeighborNN1 => {
                self.neighbor_nn1
                    .get_or_insert_with(|| NeighborCountsCache::new(NeighborCache::nn1()));
            }
            CacheId::NeighborNN2 => {
                self.neighbor_nn2
                    .get_or_insert_with(|| NeighborCountsCache::new(NeighborCache::nn2()));
            }
            CacheId::NeighborNN3 => {
                self.neighbor_nn3
                    .get_or_insert_with(|| NeighborCountsCache::new(NeighborCache::nn3()));
            }
            CacheId::NeighborNN4 => {
                self.neighbor_nn4
                    .get_or_insert_with(|| NeighborCountsCache::new(NeighborCache::nn4()));
            }
        }
    }

    pub fn place(&mut self, position_id: PositionId, stone: Stone) {
        if let Some(cache) = &mut self.neighbor_nn1 {
            cache.place(position_id, stone);
        }
        if let Some(cache) = &mut self.neighbor_nn2 {
            cache.place(position_id, stone);
        }
        if let Some(cache) = &mut self.neighbor_nn3 {
            cache.place(position_id, stone);
        }
        if let Some(cache) = &mut self.neighbor_nn4 {
            cache.place(position_id, stone);
        }
    }

    #[must_use]
    pub fn neighbor_nn1(&self) -> Option<&NeighborCountsCache> {
        self.neighbor_nn1.as_ref()
    }

    #[must_use]
    pub fn neighbor_nn2(&self) -> Option<&NeighborCountsCache> {
        self.neighbor_nn2.as_ref()
    }

    #[must_use]
    pub fn neighbor_nn3(&self) -> Option<&NeighborCountsCache> {
        self.neighbor_nn3.as_ref()
    }

    #[must_use]
    pub fn neighbor_nn4(&self) -> Option<&NeighborCountsCache> {
        self.neighbor_nn4.as_ref()
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
    use crate::offset::Offset;

    #[test]
    fn caches_are_none_by_default() {
        let repo = CacheRepository::new();

        assert!(repo.neighbor_nn1().is_none());
        assert!(repo.neighbor_nn2().is_none());
        assert!(repo.neighbor_nn3().is_none());
        assert!(repo.neighbor_nn4().is_none());
    }

    #[test]
    fn activate_creates_cache() {
        let mut repo = CacheRepository::new();

        repo.activate(CacheId::NeighborNN1);

        assert!(repo.neighbor_nn1().is_some());
        assert!(repo.neighbor_nn2().is_none());
    }

    #[test]
    fn place_updates_activated_caches() {
        let mut repo = CacheRepository::new();
        repo.activate(CacheId::NeighborNN1);
        let center = PositionId::center();

        repo.place(center, Stone::Black);

        let cache = repo.neighbor_nn1().unwrap();
        let neighbor = center.from_offset(Offset::new(0, 1)).unwrap();
        let counts = cache.counts_for(neighbor, Stone::Black);
        assert_eq!(counts.player(), 1);
    }
}
