use serde::{Deserialize, Serialize};

/// Counts of stones in a neighbor ring from the current player's perspective.
///
/// Serializes as a tuple `(player, opponent, empty)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
