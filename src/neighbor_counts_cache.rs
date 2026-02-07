use crate::cluster::NeighborCounts;
use crate::neighbor_cache::NeighborCache;
use crate::position_id::PositionId;
use crate::position_map::PositionMap;
use crate::stone::Stone;

/// Tracks neighbor counts for each position, updated as stones are placed.
#[derive(Debug, Clone)]
pub struct NeighborCountsCache {
    neighbor_cache: &'static NeighborCache,
    black_counts: PositionMap<NeighborCounts>,
    white_counts: PositionMap<NeighborCounts>,
}

impl NeighborCountsCache {
    /// Creates a new cache initialized with empty neighbor counts.
    ///
    /// # Panics
    ///
    /// Panics if the neighbor count exceeds `u8::MAX`.
    #[must_use]
    pub fn new(neighbor_cache: &'static NeighborCache) -> Self {
        let black_counts = PositionMap::from_fn(|position_id| {
            let total = neighbor_cache.neighbors_for(position_id).len();
            NeighborCounts::new(0, 0, u8::try_from(total).unwrap())
        });
        let white_counts = PositionMap::from_fn(|position_id| {
            let total = neighbor_cache.neighbors_for(position_id).len();
            NeighborCounts::new(0, 0, u8::try_from(total).unwrap())
        });

        Self {
            neighbor_cache,
            black_counts,
            white_counts,
        }
    }

    /// Returns the neighbor counts for a position from the given stone's perspective.
    ///
    /// # Panics
    ///
    /// Panics if `current_stone` is `Stone::Empty`.
    #[must_use]
    pub fn counts_for(&self, position_id: PositionId, current_stone: Stone) -> NeighborCounts {
        match current_stone {
            Stone::Black => self.black_counts[position_id],
            Stone::White => self.white_counts[position_id],
            Stone::Empty => panic!("Invalid stone"),
        }
    }

    /// Returns the positions affected when a stone is placed at `position_id`.
    ///
    /// These are the positions that have `position_id` as a neighbor.
    #[must_use]
    pub fn affected_positions(&self, position_id: PositionId) -> &[PositionId] {
        self.neighbor_cache.neighbors_for(position_id)
    }

    pub fn place(&mut self, position_id: PositionId, stone: Stone) {
        for &affected_pos in self.neighbor_cache.neighbors_for(position_id) {
            let black = self.black_counts[affected_pos];
            self.black_counts[affected_pos] = match stone {
                Stone::Black => {
                    NeighborCounts::new(black.player() + 1, black.opponent(), black.empty() - 1)
                }
                Stone::White => {
                    NeighborCounts::new(black.player(), black.opponent() + 1, black.empty() - 1)
                }
                Stone::Empty => unreachable!(),
            };

            let white = self.white_counts[affected_pos];
            self.white_counts[affected_pos] = match stone {
                Stone::Black => {
                    NeighborCounts::new(white.player(), white.opponent() + 1, white.empty() - 1)
                }
                Stone::White => {
                    NeighborCounts::new(white.player() + 1, white.opponent(), white.empty() - 1)
                }
                Stone::Empty => unreachable!(),
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::offset::Offset;

    #[test]
    fn initially_all_counts_are_empty() {
        let cache = NeighborCountsCache::new(NeighborCache::nn1());
        let center = PositionId::center();

        let counts = cache.counts_for(center, Stone::Black);

        assert_eq!(counts.player(), 0);
        assert_eq!(counts.opponent(), 0);
        assert_eq!(counts.empty(), 4);
    }

    #[test]
    fn placing_black_updates_neighbor_counts() {
        let mut cache = NeighborCountsCache::new(NeighborCache::nn1());
        let center = PositionId::center();
        let neighbor = center.from_offset(Offset::new(0, 1)).unwrap();

        cache.place(center, Stone::Black);
        let counts = cache.counts_for(neighbor, Stone::Black);

        assert_eq!(counts.player(), 1);
        assert_eq!(counts.opponent(), 0);
        assert_eq!(counts.empty(), 3);
    }

    #[test]
    fn placing_white_updates_black_opponent_count() {
        let mut cache = NeighborCountsCache::new(NeighborCache::nn1());
        let center = PositionId::center();
        let neighbor = center.from_offset(Offset::new(0, 1)).unwrap();

        cache.place(center, Stone::White);
        let counts = cache.counts_for(neighbor, Stone::Black);

        assert_eq!(counts.player(), 0);
        assert_eq!(counts.opponent(), 1);
        assert_eq!(counts.empty(), 3);
    }
}
