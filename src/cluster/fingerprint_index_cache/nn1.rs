use crate::cluster::fingerprint::{Fingerprint, FingerprintNN1};
use crate::neighbor_cache::NeighborCache;
use crate::neighbor_counts_cache::NeighborCountsCache;
use crate::position_id::PositionId;
use crate::position_map::PositionMap;
use crate::stone::Stone;

/// Cache of precomputed fingerprint indices for NN1 fingerprints.
pub struct FingerprintNN1IndexCache {
    neighbor_nn1: NeighborCountsCache,
    black_indices: PositionMap<usize>,
    white_indices: PositionMap<usize>,
}

impl FingerprintNN1IndexCache {
    #[must_use]
    pub fn new() -> Self {
        let neighbor_nn1 = NeighborCountsCache::new(NeighborCache::nn1());

        let black_indices = PositionMap::from_fn(|pos| {
            let nn1 = neighbor_nn1.counts_for(pos, Stone::Black);
            FingerprintNN1::new(nn1).index()
        });
        let white_indices = PositionMap::from_fn(|pos| {
            let nn1 = neighbor_nn1.counts_for(pos, Stone::White);
            FingerprintNN1::new(nn1).index()
        });

        Self {
            neighbor_nn1,
            black_indices,
            white_indices,
        }
    }

    pub fn place(&mut self, position_id: PositionId, stone: Stone) {
        self.neighbor_nn1.place(position_id, stone);

        for &affected_pos in self.neighbor_nn1.affected_positions(position_id) {
            self.black_indices[affected_pos] = {
                let nn1 = self.neighbor_nn1.counts_for(affected_pos, Stone::Black);
                FingerprintNN1::new(nn1).index()
            };
            self.white_indices[affected_pos] = {
                let nn1 = self.neighbor_nn1.counts_for(affected_pos, Stone::White);
                FingerprintNN1::new(nn1).index()
            };
        }
    }
}

impl Default for FingerprintNN1IndexCache {
    fn default() -> Self {
        Self::new()
    }
}

impl super::FingerprintIndexCache for FingerprintNN1IndexCache {
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
        let cache = FingerprintNN1IndexCache::new();
        let center = PositionId::center();

        let expected = FingerprintNN1::new(NeighborCounts::new(0, 0, 4)).index();

        assert_eq!(cache.get(center, Stone::Black), expected);
        assert_eq!(cache.get(center, Stone::White), expected);
    }

    #[test]
    fn placing_stone_updates_neighbor_indices() {
        let mut cache = FingerprintNN1IndexCache::new();
        let center = PositionId::center();
        let neighbor = center.from_offset(Offset::new(0, 1)).unwrap();

        cache.place(center, Stone::Black);

        let expected_black = FingerprintNN1::new(NeighborCounts::new(1, 0, 3)).index();
        let expected_white = FingerprintNN1::new(NeighborCounts::new(0, 1, 3)).index();

        assert_eq!(cache.get(neighbor, Stone::Black), expected_black);
        assert_eq!(cache.get(neighbor, Stone::White), expected_white);
    }
}
