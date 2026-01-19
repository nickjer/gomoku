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

/// Test game: panics if play() is called.
#[cfg(test)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Stub;

#[cfg(test)]
impl Play for Stub {
    fn play(
        &self,
        _black_strategy: &dyn Strategy,
        _white_strategy: &dyn Strategy,
        _rng: &mut fastrand::Rng,
    ) -> MatchResult {
        panic!("Stub game should not be called")
    }
}

/// Test game: returns predetermined outcomes.
///
/// Outcomes are stored as `Option<&'static str>` where:
/// - `Some(label)` means the strategy with that label wins
/// - `None` means a draw
#[cfg(test)]
#[derive(Debug, Clone, Default)]
pub struct Scripted {
    outcomes: std::collections::HashMap<(String, String), Option<&'static str>>,
}

#[cfg(test)]
impl Scripted {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a game outcome. The labels are sorted internally for lookup.
    pub fn add(
        mut self,
        label1: &'static str,
        label2: &'static str,
        winner: Option<&'static str>,
    ) -> Self {
        let key = Self::make_key(label1, label2);
        self.outcomes.insert(key, winner);
        self
    }

    fn make_key(label1: &str, label2: &str) -> (String, String) {
        let mut labels = [label1.to_string(), label2.to_string()];
        labels.sort();
        (labels[0].clone(), labels[1].clone())
    }
}

#[cfg(test)]
impl Play for Scripted {
    fn play(
        &self,
        black_strategy: &dyn Strategy,
        white_strategy: &dyn Strategy,
        _rng: &mut fastrand::Rng,
    ) -> MatchResult {
        let black_label = black_strategy.label();
        let white_label = white_strategy.label();
        let key = Self::make_key(black_label, white_label);

        let winner = self
            .outcomes
            .get(&key)
            .unwrap_or_else(|| panic!("No outcome defined for {key:?}"));

        let outcome = match winner {
            None => crate::outcome::Outcome::Draw,
            Some(label) if *label == black_label => crate::outcome::Outcome::BlackWins,
            Some(label) if *label == white_label => crate::outcome::Outcome::WhiteWins,
            Some(label) => panic!("Invalid winner label: {label}"),
        };

        MatchResult::new(
            outcome,
            black_label.to_string(),
            white_label.to_string(),
            0,
            String::new(),
        )
    }
}

/// Enum for polymorphic game variant dispatch.
#[enum_dispatch(Play)]
#[derive(Debug, Clone)]
pub enum Game {
    Freestyle,
    #[cfg(test)]
    Stub,
    #[cfg(test)]
    Scripted,
    // Standard,
    // Renju,
    // Caro,
}

impl Default for Game {
    fn default() -> Self {
        #[cfg(not(test))]
        {
            Freestyle.into()
        }
        #[cfg(test)]
        {
            Stub.into()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::outcome::Outcome;
    use crate::test_utils::{ScriptedStrategy, StubStrategy};

    #[test]
    fn freestyle_variant_plays_game() {
        let (black, white) = ScriptedStrategy::black_wins();
        let game: Game = Freestyle.into();
        let mut rng = fastrand::Rng::new();

        let result = game.play(&black, &white, &mut rng);

        assert_eq!(result.outcome(), Outcome::BlackWins);
    }

    #[test]
    fn scripted_variant_returns_predetermined_winner() {
        let game: Game = Scripted::new().add("a", "b", Some("a")).into();
        let a = StubStrategy::new("a");
        let b = StubStrategy::new("b");
        let mut rng = fastrand::Rng::new();

        let result = game.play(&a, &b, &mut rng);

        assert_eq!(result.outcome(), Outcome::BlackWins);
    }

    #[test]
    fn scripted_variant_returns_draw() {
        let game: Game = Scripted::new().add("a", "b", None).into();
        let a = StubStrategy::new("a");
        let b = StubStrategy::new("b");
        let mut rng = fastrand::Rng::new();

        let result = game.play(&a, &b, &mut rng);

        assert_eq!(result.outcome(), Outcome::Draw);
    }

    #[test]
    #[should_panic(expected = "Stub game should not be called")]
    fn stub_variant_panics_when_called() {
        let game: Game = Stub.into();
        let a = StubStrategy::new("a");
        let b = StubStrategy::new("b");
        let mut rng = fastrand::Rng::new();

        game.play(&a, &b, &mut rng);
    }
}
