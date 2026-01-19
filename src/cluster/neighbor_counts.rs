/// Counts of stones in a neighbor ring from the current player's perspective.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NeighborCounts {
    player: u8,
    opponent: u8,
    empty: u8,
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
