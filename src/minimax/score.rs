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
    pub const MAX: Self = Self(i32::MAX);

    #[must_use]
    pub const fn new(points: i32) -> Self {
        Self(points)
    }

    /// Returns the score normalized to `[-1.0, 1.0]` relative to `Score::WIN`.
    #[must_use]
    #[allow(clippy::cast_precision_loss, clippy::as_conversions)]
    pub fn normalized(self) -> f32 {
        self.0 as f32 / Self::WIN.0 as f32
    }

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

    /// Whether this is a win or a loss found by the search rather than an
    /// estimate: every such score lies within a board's worth of `WIN`.
    #[cfg(test)]
    #[must_use]
    pub fn is_decided(self) -> bool {
        let margin = i32::try_from(crate::position_id::PositionId::COUNT).expect("small");
        self.0.abs() >= Self::WIN.0 - margin
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

    #[test]
    fn only_wins_and_losses_are_decided() {
        assert!(Score::win_at_depth(225).is_decided());
        assert!((-Score::win_at_depth(1)).is_decided());
        assert!(!Score::new(50_000).is_decided());
        assert!(!Score::DRAW.is_decided());
    }
}
