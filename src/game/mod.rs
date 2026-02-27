mod freestyle;
mod observer;
mod random_opening;
// mod standard;  // TODO: Overlines (6+) don't count as a win
// mod renju;     // TODO: Forbidden moves for Black (3-3, 4-4, overlines)
// mod caro;      // TODO: Row must not be blocked at both ends to win

use enum_dispatch::enum_dispatch;

pub use freestyle::Freestyle;
pub use observer::{GameObserver, NoOpObserver};
pub use random_opening::RandomOpening;

use crate::board::Board;
use crate::match_result::MatchResult;
use crate::stone::Stone;
use crate::strategy::Strategy;

/// Trait for playing games between strategies.
///
/// The required method is [`Play::play_from`], which accepts a mutable board so callers can
/// pre-populate it with an opening position and inspect its state after an early exit.
/// [`Play::play`] is a provided method that creates an empty board and plays to completion.
#[enum_dispatch]
pub trait Play {
    /// Plays a game starting from `board`, calling `observer` after each move is chosen but
    /// before it is placed.
    ///
    /// Returns `None` if the observer breaks early (e.g. move limit reached). On return,
    /// `board` reflects the final state whether the game finished naturally or was aborted.
    fn play_from(
        &self,
        board: &mut Board,
        black_strategy: &dyn Strategy,
        white_strategy: &dyn Strategy,
        observer: &mut dyn GameObserver,
        rng: &mut fastrand::Rng,
    ) -> Option<MatchResult>;

    /// Plays a complete game from an empty board with no observer hooks.
    fn play(
        &self,
        black_strategy: &dyn Strategy,
        white_strategy: &dyn Strategy,
        rng: &mut fastrand::Rng,
    ) -> MatchResult {
        let mut board = Board::new();
        self.play_from(
            &mut board,
            black_strategy,
            white_strategy,
            &mut NoOpObserver,
            rng,
        )
        .expect("NoOpObserver never breaks early")
    }
}

/// Shared game loop used by all game variants. Drives alternating play until the board is
/// finished or the observer breaks early. Takes `&mut Board` so callers retain the board
/// state after an early exit.
pub fn run_from(
    board: &mut Board,
    black_strategy: &dyn Strategy,
    white_strategy: &dyn Strategy,
    observer: &mut dyn GameObserver,
    rng: &mut fastrand::Rng,
) -> Option<MatchResult> {
    while !board.is_finished() {
        let (stone, strategy): (Stone, &dyn Strategy) = if board.move_count().is_multiple_of(2) {
            (Stone::Black, black_strategy)
        } else {
            (Stone::White, white_strategy)
        };

        let position = strategy.choose_move(stone, board, rng);

        if observer.on_move(stone, position, board).is_break() {
            return None;
        }

        board
            .place(position, stone)
            .expect("strategy returned invalid move");
    }

    let outcome = board.outcome().expect("game finished without outcome");

    Some(MatchResult::new(
        outcome,
        black_strategy.label().to_string(),
        white_strategy.label().to_string(),
        board.move_count(),
        board.to_string(),
    ))
}

/// Test game: panics if `play_from` is called.
#[cfg(test)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Stub;

#[cfg(test)]
impl Play for Stub {
    fn play_from(
        &self,
        _board: &mut Board,
        _black_strategy: &dyn Strategy,
        _white_strategy: &dyn Strategy,
        _observer: &mut dyn GameObserver,
        _rng: &mut fastrand::Rng,
    ) -> Option<MatchResult> {
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
    fn play_from(
        &self,
        _board: &mut Board,
        black_strategy: &dyn Strategy,
        white_strategy: &dyn Strategy,
        _observer: &mut dyn GameObserver,
        _rng: &mut fastrand::Rng,
    ) -> Option<MatchResult> {
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

        Some(MatchResult::new(
            outcome,
            black_label.to_string(),
            white_label.to_string(),
            0,
            String::new(),
        ))
    }
}

/// Enum for polymorphic game variant dispatch.
#[enum_dispatch(Play)]
#[derive(Debug, Clone)]
pub enum Game {
    Freestyle,
    RandomOpening,
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
    use std::ops::ControlFlow;

    use super::*;
    use crate::outcome::Outcome;
    use crate::position_id::PositionId;
    use crate::test_utils::{ScriptedStrategy, StubStrategy};

    /// Test observer that records the board's move count on the first call.
    struct FirstMoveCapture {
        captured_move_count: Option<usize>,
    }

    impl GameObserver for FirstMoveCapture {
        fn on_move(&mut self, _: Stone, _: PositionId, board: &Board) -> ControlFlow<()> {
            if self.captured_move_count.is_none() {
                self.captured_move_count = Some(board.move_count());
            }
            ControlFlow::Continue(())
        }
    }

    /// Test observer that breaks after allowing a fixed number of moves through.
    struct BreakAfter {
        remaining: usize,
    }

    impl GameObserver for BreakAfter {
        fn on_move(&mut self, _: Stone, _: PositionId, _: &Board) -> ControlFlow<()> {
            if self.remaining == 0 {
                ControlFlow::Break(())
            } else {
                self.remaining -= 1;
                ControlFlow::Continue(())
            }
        }
    }

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

    #[test]
    fn play_default_starts_from_empty_board() {
        let (black, white) = ScriptedStrategy::black_wins();
        let mut capture = FirstMoveCapture {
            captured_move_count: None,
        };
        let mut board = Board::new();
        let mut rng = fastrand::Rng::new();

        Freestyle.play_from(&mut board, &black, &white, &mut capture, &mut rng);

        assert_eq!(capture.captured_move_count, Some(0));
    }

    #[test]
    fn play_from_board_reflects_final_state_on_natural_finish() {
        let (black, white) = ScriptedStrategy::black_wins();
        let mut board = Board::new();
        let mut rng = fastrand::Rng::new();

        Freestyle.play_from(&mut board, &black, &white, &mut NoOpObserver, &mut rng);

        assert!(board.is_finished());
    }

    #[test]
    fn play_from_board_reflects_partial_state_on_early_exit() {
        let (black, white) = ScriptedStrategy::black_wins();
        let mut board = Board::new();
        let mut rng = fastrand::Rng::new();
        let mut observer = BreakAfter { remaining: 2 };

        let result = Freestyle.play_from(&mut board, &black, &white, &mut observer, &mut rng);

        assert!(result.is_none());
        assert_eq!(board.move_count(), 2);
    }
}
