use tracing::instrument;

use crate::board::Board;
use crate::match_result::MatchResult;
use crate::strategy::Strategy;

use super::{GameObserver, Play, run_from};

/// Freestyle Gomoku: 5+ in a row wins, no restrictions.
#[derive(Debug, Clone, Copy, Default)]
pub struct Freestyle;

impl Play for Freestyle {
    #[instrument(level = "debug", skip_all, fields(black = black_strategy.label(), white = white_strategy.label()))]
    fn play_from(
        &self,
        board: &mut Board,
        black_strategy: &dyn Strategy,
        white_strategy: &dyn Strategy,
        observer: &mut dyn GameObserver,
        rng: &mut fastrand::Rng,
    ) -> Option<MatchResult> {
        run_from(board, black_strategy, white_strategy, observer, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::outcome::Outcome;
    use crate::test_utils::ScriptedStrategy;

    #[test]
    fn returns_match_result_with_black_wins() {
        let (black, white) = ScriptedStrategy::black_wins();
        let game = Freestyle;
        let mut rng = fastrand::Rng::new();

        let result = game.play(&black, &white, &mut rng);

        assert_eq!(result.outcome(), Outcome::BlackWins);
    }

    #[test]
    fn returns_match_result_with_white_wins() {
        let (black, white) = ScriptedStrategy::white_wins();
        let game = Freestyle;
        let mut rng = fastrand::Rng::new();

        let result = game.play(&black, &white, &mut rng);

        assert_eq!(result.outcome(), Outcome::WhiteWins);
    }

    #[test]
    fn returns_match_result_with_draw() {
        let (black, white) = ScriptedStrategy::draw();
        let game = Freestyle;
        let mut rng = fastrand::Rng::new();

        let result = game.play(&black, &white, &mut rng);

        assert_eq!(result.outcome(), Outcome::Draw);
    }

    #[test]
    fn turn_count_is_correct() {
        let (black, white) = ScriptedStrategy::black_wins();
        let game = Freestyle;
        let mut rng = fastrand::Rng::new();

        let result = game.play(&black, &white, &mut rng);

        // Black plays 5 moves, white plays 4 moves = 9 total
        assert_eq!(result.turn_count(), 9);
    }

    #[test]
    fn captures_strategy_labels() {
        let black = ScriptedStrategy::with_positions(
            "black_label",
            &[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
        );
        let white =
            ScriptedStrategy::with_positions("white_label", &[(1, 0), (1, 1), (1, 2), (1, 3)]);
        let game = Freestyle;
        let mut rng = fastrand::Rng::new();

        let result = game.play(&black, &white, &mut rng);

        assert_eq!(result.black_label(), "black_label");
        assert_eq!(result.white_label(), "white_label");
    }

    #[test]
    fn captures_board_state() {
        let (black, white) = ScriptedStrategy::black_wins();
        let game = Freestyle;
        let mut rng = fastrand::Rng::new();

        let result = game.play(&black, &white, &mut rng);

        assert!(!result.board_state().is_empty());
    }
}
