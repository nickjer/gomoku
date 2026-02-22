use std::ops;

use derive_more::{Add, AddAssign, From, Neg, Sub, SubAssign};

/// Evaluation score for minimax search.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Add, AddAssign, Sub, SubAssign, Neg, From,
)]
pub struct Score(i32);

impl Score {
    pub const WIN: Self = Self(100_000_000);
    pub const DRAW: Self = Self(0);
    pub const MIN: Self = Self(-i32::MAX);

    /// Open four is an unstoppable win — score just below WIN.
    pub const OPEN_FOUR: Self = Self(10_000_000);
    /// Half-open four forces exactly one blocking reply.
    pub const HALF_OPEN_FOUR: Self = Self(1_000_000);
    /// Open three becomes open four if not blocked — near-decisive.
    pub const OPEN_THREE: Self = Self(100_000);
    pub const HALF_OPEN_THREE: Self = Self(10_000);
    pub const OPEN_TWO: Self = Self(1_000);
    pub const HALF_OPEN_TWO: Self = Self(100);

    /// Win score adjusted for depth — prefers faster wins (fewer moves).
    #[must_use]
    pub fn win_at_depth(move_count: usize) -> Self {
        Self(
            Self::WIN
                .0
                .checked_sub(i32::try_from(move_count).expect("move count overflow"))
                .expect("win score underflow"),
        )
    }
}

impl ops::Mul<usize> for Score {
    type Output = Self;

    fn mul(self, rhs: usize) -> Self {
        Self(self.0 * i32::try_from(rhs).expect("count overflow"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn win_at_depth_prefers_faster_wins() {
        let early = Score::win_at_depth(3);
        let late = Score::win_at_depth(10);

        assert!(early > late, "fewer moves to win should score higher");
        assert!(
            early < Score::WIN,
            "win_at_depth should be below the base WIN score"
        );
    }
}
