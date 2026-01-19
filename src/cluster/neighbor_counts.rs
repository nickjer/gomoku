use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};

/// Counts of stones in a neighbor ring from the current player's perspective.
///
/// Serializes as a tuple `(player, opponent, empty)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "(u8, u8, u8)", into = "(u8, u8, u8)")]
pub struct NeighborCounts {
    player: u8,
    opponent: u8,
    empty: u8,
}

impl From<(u8, u8, u8)> for NeighborCounts {
    fn from((player, opponent, empty): (u8, u8, u8)) -> Self {
        Self::new(player, opponent, empty)
    }
}

impl From<NeighborCounts> for (u8, u8, u8) {
    fn from(counts: NeighborCounts) -> Self {
        (counts.player, counts.opponent, counts.empty)
    }
}

impl NeighborCounts {
    #[must_use]
    pub const fn new(player: u8, opponent: u8, empty: u8) -> Self {
        Self {
            player,
            opponent,
            empty,
        }
    }

    #[must_use]
    pub const fn player(&self) -> u8 {
        self.player
    }

    #[must_use]
    pub const fn opponent(&self) -> u8 {
        self.opponent
    }

    #[must_use]
    pub const fn empty(&self) -> u8 {
        self.empty
    }

    /// Packs the three u8 values into a single u64 for efficient hashing.
    #[allow(clippy::as_conversions)] // Widening u8 to u64 is always safe
    #[must_use]
    const fn packed(self) -> u64 {
        (self.player as u64) | ((self.opponent as u64) << 8) | ((self.empty as u64) << 16)
    }
}

impl Hash for NeighborCounts {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.packed().hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_as_tuple() {
        let counts = NeighborCounts::new(1, 2, 3);

        let serialized = ron::to_string(&counts).unwrap();

        assert_eq!(serialized, "(1,2,3)");
    }

    #[test]
    fn deserializes_from_tuple() {
        let serialized = "(1, 2, 3)";

        let counts: NeighborCounts = ron::from_str(serialized).unwrap();

        assert_eq!(counts.player(), 1);
        assert_eq!(counts.opponent(), 2);
        assert_eq!(counts.empty(), 3);
    }

    #[test]
    fn serialization_round_trip() {
        let original = NeighborCounts::new(4, 0, 0);

        let serialized = ron::to_string(&original).unwrap();
        let deserialized: NeighborCounts = ron::from_str(&serialized).unwrap();

        assert_eq!(deserialized, original);
    }
}
