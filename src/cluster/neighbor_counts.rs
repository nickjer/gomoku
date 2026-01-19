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
