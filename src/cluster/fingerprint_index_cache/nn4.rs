use crate::cluster::fingerprint::{Fingerprint, FingerprintNN4};
use crate::neighbor_cache::NeighborCache;
use crate::neighbor_counts_cache::NeighborCountsCache;
use crate::position_id::PositionId;
use crate::position_map::PositionMap;
use crate::stone::Stone;

/// Cache of precomputed fingerprint indices for NN4 fingerprints.
pub struct FingerprintNN4IndexCache {
    neighbor_nn1: NeighborCountsCache,
    neighbor_nn2: NeighborCountsCache,
    neighbor_nn3: NeighborCountsCache,
    neighbor_nn4: NeighborCountsCache,
    black_indices: PositionMap<usize>,
    white_indices: PositionMap<usize>,
}

impl FingerprintNN4IndexCache {
    #[must_use]
    pub fn new() -> Self {
        let neighbor_nn1 = NeighborCountsCache::new(NeighborCache::nn1());
        let neighbor_nn2 = NeighborCountsCache::new(NeighborCache::nn2());
        let neighbor_nn3 = NeighborCountsCache::new(NeighborCache::nn3());
        let neighbor_nn4 = NeighborCountsCache::new(NeighborCache::nn4());

        let black_indices = PositionMap::from_fn(|pos| {
            let nn1 = neighbor_nn1.counts_for(pos, Stone::Black);
            let nn2 = neighbor_nn2.counts_for(pos, Stone::Black);
            let nn3 = neighbor_nn3.counts_for(pos, Stone::Black);
            let nn4 = neighbor_nn4.counts_for(pos, Stone::Black);
            FingerprintNN4::new(nn1, nn2, nn3, nn4).index()
        });
        let white_indices = PositionMap::from_fn(|pos| {
            let nn1 = neighbor_nn1.counts_for(pos, Stone::White);
            let nn2 = neighbor_nn2.counts_for(pos, Stone::White);
            let nn3 = neighbor_nn3.counts_for(pos, Stone::White);
            let nn4 = neighbor_nn4.counts_for(pos, Stone::White);
            FingerprintNN4::new(nn1, nn2, nn3, nn4).index()
        });

        Self {
            neighbor_nn1,
            neighbor_nn2,
            neighbor_nn3,
            neighbor_nn4,
            black_indices,
            white_indices,
        }
    }

    pub fn place(&mut self, position_id: PositionId, stone: Stone) {
        self.neighbor_nn1.place(position_id, stone);
        self.neighbor_nn2.place(position_id, stone);
        self.neighbor_nn3.place(position_id, stone);
        self.neighbor_nn4.place(position_id, stone);

        for &affected_pos in self
            .neighbor_nn1
            .affected_positions(position_id)
            .iter()
            .chain(self.neighbor_nn2.affected_positions(position_id))
            .chain(self.neighbor_nn3.affected_positions(position_id))
            .chain(self.neighbor_nn4.affected_positions(position_id))
        {
            self.black_indices[affected_pos] = {
                let nn1 = self.neighbor_nn1.counts_for(affected_pos, Stone::Black);
                let nn2 = self.neighbor_nn2.counts_for(affected_pos, Stone::Black);
                let nn3 = self.neighbor_nn3.counts_for(affected_pos, Stone::Black);
                let nn4 = self.neighbor_nn4.counts_for(affected_pos, Stone::Black);
                FingerprintNN4::new(nn1, nn2, nn3, nn4).index()
            };
            self.white_indices[affected_pos] = {
                let nn1 = self.neighbor_nn1.counts_for(affected_pos, Stone::White);
                let nn2 = self.neighbor_nn2.counts_for(affected_pos, Stone::White);
                let nn3 = self.neighbor_nn3.counts_for(affected_pos, Stone::White);
                let nn4 = self.neighbor_nn4.counts_for(affected_pos, Stone::White);
                FingerprintNN4::new(nn1, nn2, nn3, nn4).index()
            };
        }
    }
}

impl Default for FingerprintNN4IndexCache {
    fn default() -> Self {
        Self::new()
    }
}

impl super::FingerprintIndexCache for FingerprintNN4IndexCache {
    /// Returns the fingerprint index for the given position and stone color.
    ///
    /// # Panics
    ///
    /// Panics if `current_stone` is `Stone::Empty`.
    fn get(&self, position_id: PositionId, current_stone: Stone) -> usize {
        match current_stone {
            Stone::Black => self.black_indices[position_id],
            Stone::White => self.white_indices[position_id],
            Stone::Empty => panic!("Invalid stone"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::FingerprintIndexCache;
    use crate::cluster::NeighborCounts;
    use crate::offset::Offset;

    #[test]
    fn initial_indices_match_empty_board() {
        let cache = FingerprintNN4IndexCache::new();
        let center = PositionId::center();

        // Center has 4 NN1, 4 NN2, 4 NN3, 8 NN4 neighbors, all empty
        let expected = FingerprintNN4::new(
            NeighborCounts::new(0, 0, 4),
            NeighborCounts::new(0, 0, 4),
            NeighborCounts::new(0, 0, 4),
            NeighborCounts::new(0, 0, 8),
        )
        .index();

        assert_eq!(cache.get(center, Stone::Black), expected);
        assert_eq!(cache.get(center, Stone::White), expected);
    }

    #[test]
    fn placing_stone_updates_neighbor_indices() {
        let mut cache = FingerprintNN4IndexCache::new();
        let center = PositionId::center();
        let neighbor = center.from_offset(Offset::new(0, 1)).unwrap();

        cache.place(center, Stone::Black);

        // Neighbor's NN1 now has 1 black stone
        let expected_black = FingerprintNN4::new(
            NeighborCounts::new(1, 0, 3),
            NeighborCounts::new(0, 0, 4),
            NeighborCounts::new(0, 0, 4),
            NeighborCounts::new(0, 0, 8),
        )
        .index();

        assert_eq!(cache.get(neighbor, Stone::Black), expected_black);
    }
}
