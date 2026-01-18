use std::sync::LazyLock;

use crate::offset::Offset;
use crate::pair::offsets;
use crate::position_id::PositionId;
use crate::position_map::PositionMap;

/// Pre-computed cache of valid neighbor positions for each board position.
pub struct NeighborCache {
    data: PositionMap<Vec<PositionId>>,
}

impl NeighborCache {
    fn new(offsets: &[Offset]) -> Self {
        Self {
            data: PositionMap::from_fn(|position_id| {
                offsets
                    .iter()
                    .filter_map(|&offset| position_id.from_offset(offset))
                    .collect()
            }),
        }
    }

    #[must_use]
    pub fn neighbors_for(&self, position_id: PositionId) -> &[PositionId] {
        &self.data[position_id]
    }

    #[must_use]
    pub fn nn1() -> &'static Self {
        static CACHE: LazyLock<NeighborCache> = LazyLock::new(|| NeighborCache::new(&offsets::NN1));
        &CACHE
    }

    #[must_use]
    pub fn nn2() -> &'static Self {
        static CACHE: LazyLock<NeighborCache> = LazyLock::new(|| NeighborCache::new(&offsets::NN2));
        &CACHE
    }

    #[must_use]
    pub fn nn3() -> &'static Self {
        static CACHE: LazyLock<NeighborCache> = LazyLock::new(|| NeighborCache::new(&offsets::NN3));
        &CACHE
    }

    #[must_use]
    pub fn nn4() -> &'static Self {
        static CACHE: LazyLock<NeighborCache> = LazyLock::new(|| NeighborCache::new(&offsets::NN4));
        &CACHE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;

    fn pos(row: u8, col: u8) -> PositionId {
        PositionId::from_position(Position::new(row, col))
    }

    #[test]
    fn nn1_corner_has_two_neighbors() {
        let cache = NeighborCache::nn1();
        let corner = pos(0, 0);

        assert_eq!(cache.neighbors_for(corner).len(), 2);
    }

    #[test]
    fn nn1_center_has_four_neighbors() {
        let cache = NeighborCache::nn1();
        let center = PositionId::center();

        assert_eq!(cache.neighbors_for(center).len(), 4);
    }

    #[test]
    fn nn2_corner_has_one_neighbor() {
        let cache = NeighborCache::nn2();
        let corner = pos(0, 0);

        assert_eq!(cache.neighbors_for(corner).len(), 1);
    }

    #[test]
    fn nn2_center_has_four_neighbors() {
        let cache = NeighborCache::nn2();
        let center = PositionId::center();

        assert_eq!(cache.neighbors_for(center).len(), 4);
    }

    #[test]
    fn nn3_corner_has_two_neighbors() {
        let cache = NeighborCache::nn3();
        let corner = pos(0, 0);

        assert_eq!(cache.neighbors_for(corner).len(), 2);
    }

    #[test]
    fn nn4_corner_has_two_neighbors() {
        let cache = NeighborCache::nn4();
        let corner = pos(0, 0);

        assert_eq!(cache.neighbors_for(corner).len(), 2);
    }

    #[test]
    fn nn4_center_has_eight_neighbors() {
        let cache = NeighborCache::nn4();
        let center = PositionId::center();

        assert_eq!(cache.neighbors_for(center).len(), 8);
    }
}
