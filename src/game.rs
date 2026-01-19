use crate::board::Board;
use crate::cache_repository::CacheRepository;
use crate::match_result::MatchResult;
use crate::stone::Stone;
use crate::strategy::Strategy;

/// Trait for playing games between strategies.
pub trait Play {
    fn play(
        &self,
        black_strategy: &dyn Strategy,
        white_strategy: &dyn Strategy,
        rng: &mut fastrand::Rng,
    ) -> MatchResult;
}

/// Plays a game between two strategies.
#[derive(Debug, Default)]
pub struct Game;

impl Game {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Runs a match between the black and white strategies.
    ///
    /// Returns a `MatchResult` containing the outcome and metadata.
    ///
    /// # Panics
    ///
    /// Panics if a strategy returns an invalid move (non-empty position).
    pub fn run(
        &self,
        black_strategy: &dyn Strategy,
        white_strategy: &dyn Strategy,
        rng: &mut fastrand::Rng,
    ) -> MatchResult {
        let mut cache_repo = CacheRepository::new();

        // Activate cache dependencies for both strategies
        for &dep in black_strategy.cache_dependencies() {
            cache_repo.activate(dep);
        }
        for &dep in white_strategy.cache_dependencies() {
            cache_repo.activate(dep);
        }

        let mut board = Board::new();
        let mut turn_count: u32 = 0;

        while !board.is_finished() {
            let (strategy, stone): (&dyn Strategy, Stone) = if turn_count.is_multiple_of(2) {
                (black_strategy, Stone::Black)
            } else {
                (white_strategy, Stone::White)
            };

            let position_id = strategy.choose_move(stone, &board, &cache_repo, rng);
            board
                .place(position_id, stone)
                .expect("strategy returned invalid move");
            cache_repo.place(position_id, stone);
            turn_count += 1;
        }

        let outcome = board.outcome().expect("game finished without outcome");

        MatchResult::new(
            outcome,
            black_strategy.label().to_string(),
            white_strategy.label().to_string(),
            turn_count,
            board.to_string(),
        )
    }
}

impl Play for Game {
    fn play(
        &self,
        black_strategy: &dyn Strategy,
        white_strategy: &dyn Strategy,
        rng: &mut fastrand::Rng,
    ) -> MatchResult {
        self.run(black_strategy, white_strategy, rng)
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
        let game = Game::new();
        let mut rng = fastrand::Rng::new();

        let result = game.run(&black, &white, &mut rng);

        assert_eq!(result.outcome(), Outcome::BlackWins);
    }

    #[test]
    fn returns_match_result_with_white_wins() {
        let (black, white) = ScriptedStrategy::white_wins();
        let game = Game::new();
        let mut rng = fastrand::Rng::new();

        let result = game.run(&black, &white, &mut rng);

        assert_eq!(result.outcome(), Outcome::WhiteWins);
    }

    #[test]
    fn returns_match_result_with_draw() {
        let (black, white) = ScriptedStrategy::draw();
        let game = Game::new();
        let mut rng = fastrand::Rng::new();

        let result = game.run(&black, &white, &mut rng);

        assert_eq!(result.outcome(), Outcome::Draw);
    }

    #[test]
    fn turn_count_is_correct() {
        let (black, white) = ScriptedStrategy::black_wins();
        let game = Game::new();
        let mut rng = fastrand::Rng::new();

        let result = game.run(&black, &white, &mut rng);

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
        let game = Game::new();
        let mut rng = fastrand::Rng::new();

        let result = game.run(&black, &white, &mut rng);

        assert_eq!(result.black_label(), "black_label");
        assert_eq!(result.white_label(), "white_label");
    }

    #[test]
    fn captures_board_state() {
        let (black, white) = ScriptedStrategy::black_wins();
        let game = Game::new();
        let mut rng = fastrand::Rng::new();

        let result = game.run(&black, &white, &mut rng);

        assert!(!result.board_state().is_empty());
    }
}
