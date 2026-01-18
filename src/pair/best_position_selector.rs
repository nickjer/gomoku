use std::collections::HashMap;
use std::hash::Hash;

use crate::position_id::PositionId;

/// Selects the best position based on fingerprint priority.
pub struct BestPositionSelector<F> {
    priority_map: HashMap<F, usize>,
}

impl<F: Eq + Hash + Clone> BestPositionSelector<F> {
    #[must_use]
    pub fn new(fingerprint_priority: &[F]) -> Self {
        let priority_map = fingerprint_priority
            .iter()
            .cloned()
            .enumerate()
            .map(|(idx, f)| (f, idx))
            .collect();
        Self { priority_map }
    }

    /// Selects the best position from position-fingerprint pairs.
    ///
    /// # Panics
    ///
    /// Panics if empty or contains unknown fingerprint.
    #[must_use]
    pub fn select(
        &self,
        position_fingerprints: &[(PositionId, F)],
        rng: &mut fastrand::Rng,
    ) -> PositionId {
        let best_priority = position_fingerprints
            .iter()
            .map(|(_, f)| {
                *self
                    .priority_map
                    .get(f)
                    .expect("Unknown fingerprint in priority map")
            })
            .min()
            .expect("No positions provided");

        let best: Vec<_> = position_fingerprints
            .iter()
            .filter(|(_, f)| self.priority_map[f] == best_priority)
            .map(|(pos, _)| *pos)
            .collect();

        best[rng.usize(..best.len())]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pair::NeighborCounts;
    use crate::position::Position;

    fn pos(row: u8, col: u8) -> PositionId {
        PositionId::from_position(Position::new(row, col))
    }

    #[test]
    fn selects_position_with_highest_priority() {
        let fingerprints = vec![
            NeighborCounts::new(0, 0, 4),
            NeighborCounts::new(1, 0, 3),
            NeighborCounts::new(0, 1, 3),
        ];
        let selector = BestPositionSelector::new(&fingerprints);
        let mut rng = fastrand::Rng::new();

        let position_fingerprints = vec![
            (pos(0, 0), NeighborCounts::new(0, 1, 3)), // priority 2
            (pos(0, 1), NeighborCounts::new(0, 0, 4)), // priority 0 (best)
            (pos(0, 2), NeighborCounts::new(1, 0, 3)), // priority 1
        ];

        let chosen = selector.select(&position_fingerprints, &mut rng);

        assert_eq!(chosen, pos(0, 1));
    }

    #[test]
    fn selects_from_tied_positions() {
        let fingerprints = vec![NeighborCounts::new(0, 0, 4)];
        let selector = BestPositionSelector::new(&fingerprints);
        let mut rng = fastrand::Rng::new();

        let position_fingerprints = vec![
            (pos(0, 0), NeighborCounts::new(0, 0, 4)),
            (pos(0, 1), NeighborCounts::new(0, 0, 4)),
        ];

        let chosen = selector.select(&position_fingerprints, &mut rng);

        assert!(chosen == pos(0, 0) || chosen == pos(0, 1));
    }
}
