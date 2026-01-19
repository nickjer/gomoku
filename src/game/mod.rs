mod freestyle;
// mod standard;  // TODO: Overlines (6+) don't count as a win
// mod renju;     // TODO: Forbidden moves for Black (3-3, 4-4, overlines)
// mod caro;      // TODO: Row must not be blocked at both ends to win

use enum_dispatch::enum_dispatch;

pub use freestyle::Freestyle;

use crate::match_result::MatchResult;
use crate::strategy::Strategy;

/// Trait for playing games between strategies.
#[enum_dispatch]
pub trait Play {
    fn play(
        &self,
        black_strategy: &dyn Strategy,
        white_strategy: &dyn Strategy,
        rng: &mut fastrand::Rng,
    ) -> MatchResult;
}

/// Enum for polymorphic game variant dispatch.
#[enum_dispatch(Play)]
#[derive(Debug, Clone, Copy)]
pub enum Game {
    Freestyle,
    // Standard,
    // Renju,
    // Caro,
}

impl Default for Game {
    fn default() -> Self {
        Freestyle.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::outcome::Outcome;
    use crate::test_utils::ScriptedStrategy;

    #[test]
    fn freestyle_variant_plays_game() {
        let (black, white) = ScriptedStrategy::black_wins();
        let game: Game = Freestyle.into();
        let mut rng = fastrand::Rng::new();

        let result = game.play(&black, &white, &mut rng);

        assert_eq!(result.outcome(), Outcome::BlackWins);
    }
}
