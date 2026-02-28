use std::ops::ControlFlow;

use tracing::{debug, info, instrument, trace};

use crate::board::Board;
use crate::evolution::fitness_score::FitnessScore;
use crate::game::{Game, GameObserver, Play};
use crate::minimax::{MinimaxStrategy, score_move};
use crate::outcome::Outcome;
use crate::position_id::PositionId;
use crate::stone::Stone;
use crate::strategy::Strategy;

use super::EvaluateFitness;

/// Evaluates fitness by playing each strategy against minimax at multiple depths.
/// Minimax plays as Black; the evaluated strategy plays as White.
///
/// Across depths, keeps the minimax win with the fewest turns (or the strategy win
/// with the most turns if minimax never wins). Deeper searches that cannot improve
/// on the current best are skipped.
///
/// Without `scoring_depth`: fitness is game length (longer survival or faster win).
/// With `scoring_depth`: fitness combines average per-move quality with game length;
/// quality dominates and length acts as a tiebreaker.
pub struct MinimaxFitness {
    game: Game,
    depths: Vec<u32>,
    scoring_depth: Option<u32>,
}

impl MinimaxFitness {
    #[must_use]
    pub fn new(game: Game, mut depths: Vec<u32>, scoring_depth: Option<u32>) -> Self {
        depths.sort_unstable();
        depths.dedup();
        Self {
            game,
            depths,
            scoring_depth,
        }
    }

    /// Minimum turns for Black to win: 5 stones on alternating turns starting from turn 1.
    const FASTEST_WIN_TURNS: usize = 9;

    fn evaluate_single(&self, strategy: &dyn Strategy, rng: &mut fastrand::Rng) -> f32 {
        let mut minimax_win: Option<EvalResult> = None;
        let mut strategy_win: Option<EvalResult> = None;

        for &depth in &self.depths {
            if minimax_win
                .as_ref()
                .is_some_and(|r| r.turn_count == Self::FASTEST_WIN_TURNS)
            {
                debug!(
                    label = strategy.label(),
                    depth,
                    turn_count = minimax_win.as_ref().unwrap().turn_count,
                    "Minimax perfect win cutoff"
                );
                break;
            }

            let minimax = MinimaxStrategy::new(depth);
            let move_limit = minimax_win.as_ref().map(|r| r.turn_count);
            let mut board = Board::new();
            let mut observer = MinimaxObserver::new(move_limit, self.scoring_depth);

            let Some(_) = self
                .game
                .play_from(&mut board, &minimax, strategy, &mut observer, rng)
            else {
                debug!(
                    label = strategy.label(),
                    depth,
                    turn_count = move_limit.expect("cutoff requires a prior result"),
                    "Minimax depth cutoff"
                );
                continue;
            };

            let result = EvalResult {
                turn_count: board.move_count(),
                outcome: board.outcome().expect("finished game has outcome"),
                score_sum: observer.score_sum,
            };

            debug!(
                label = strategy.label(),
                depth,
                turn_count = result.turn_count,
                outcome = ?result.outcome,
                "Minimax evaluation"
            );

            match result.outcome {
                Outcome::BlackWins | Outcome::Draw => {
                    if minimax_win
                        .as_ref()
                        .is_none_or(|r| result.turn_count < r.turn_count)
                    {
                        minimax_win = Some(result);
                    }
                }
                Outcome::WhiteWins => {
                    if strategy_win
                        .as_ref()
                        .is_none_or(|r| result.turn_count > r.turn_count)
                    {
                        strategy_win = Some(result);
                    }
                }
            }
        }

        match self.scoring_depth {
            None => match (&minimax_win, &strategy_win) {
                (Some(r), _) => turns_as_f32(r.turn_count),
                (None, Some(r)) => 1000.0 - turns_as_f32(r.turn_count),
                (None, None) => 0.0,
            },
            Some(_) => minimax_win
                .as_ref()
                .or(strategy_win.as_ref())
                .map_or(0.0, |r| {
                    scoring_depth_fitness(r.score_sum, r.turn_count, r.outcome)
                }),
        }
    }
}

impl EvaluateFitness for MinimaxFitness {
    #[instrument(name = "MinimaxFitness", skip_all)]
    fn evaluate<S: Strategy>(
        &self,
        strategies: &[S],
        rng: &mut fastrand::Rng,
    ) -> Vec<FitnessScore> {
        let scores: Vec<FitnessScore> = strategies
            .iter()
            .map(|strategy| FitnessScore::new(self.evaluate_single(strategy, rng)))
            .collect();
        log_minimax_scores(strategies, &scores);
        scores
    }
}

fn log_minimax_scores<S: Strategy>(strategies: &[S], scores: &[FitnessScore]) {
    let mut indices: Vec<usize> = (0..strategies.len()).collect();
    indices.sort_by(|&a, &b| scores[b].value().total_cmp(&scores[a].value()));

    let top_count = indices.len().min(5);
    let bottom_start = if indices.len() > 10 {
        indices.len() - 5
    } else {
        top_count
    };

    for (rank, &idx) in indices.iter().enumerate() {
        let label = strategies[idx].label();
        let score = scores[idx].value();
        if rank < top_count || rank >= bottom_start {
            info!(rank = rank + 1, label, score, "Minimax");
        } else {
            debug!(rank = rank + 1, label, score, "Minimax");
        }
    }
}

/// Fitness formula used when `scoring_depth` is set.
///
/// Sums average per-move quality (dominant signal) and game length (tiebreaker).
/// Longer losses and faster wins both score higher; wins land well above losses.
fn scoring_depth_fitness(score_sum: f32, turn_count: usize, outcome: Outcome) -> f32 {
    let white_moves = f32::from(u16::try_from(turn_count / 2).expect("white move count fits u16"));
    let quality = (score_sum / white_moves + 1.0) * 500.0;
    let length = match outcome {
        Outcome::WhiteWins => 1000.0 - 2.0 * turns_as_f32(turn_count),
        Outcome::BlackWins | Outcome::Draw => 2.0 * turns_as_f32(turn_count),
    };
    quality + length
}

fn turns_as_f32(turns: usize) -> f32 {
    f32::from(u16::try_from(turns).expect("turn count fits in u16"))
}

/// Tracks the best game result across depth iterations in [`MinimaxFitness::evaluate_single`].
struct EvalResult {
    turn_count: usize,
    outcome: Outcome,
    /// Sum of per-move contributions (delta + proximity bonus) for White's moves;
    /// `0.0` when `scoring_depth` is `None`.
    score_sum: f32,
}

/// Bonus added to each White move within [`PROXIMITY_RADIUS`](crate::bitboard::PROXIMITY_RADIUS)
/// of any existing stone.
///
/// With very few stones the depth delta is flat across all positions — Black can pivot among
/// four independent build directions so no single White placement changes the evaluation.
/// This nudges nearby moves above far corners as a tiebreaker without overriding real
/// tactical signals at the recommended depth of 4+.
const PROXIMITY_BONUS: f32 = 0.01;

/// Per-game observer for [`MinimaxFitness`]. Enforces an optional move limit and accumulates
/// per-move score contributions for White when `scoring_depth` is set. Each contribution
/// combines the symmetric depth delta with a [`PROXIMITY_BONUS`] for moves near existing stones.
struct MinimaxObserver {
    move_limit: Option<usize>,
    scoring_depth: Option<u32>,
    score_sum: f32,
    prev_black_move: Option<(Board, PositionId)>,
}

impl MinimaxObserver {
    fn new(move_limit: Option<usize>, scoring_depth: Option<u32>) -> Self {
        Self {
            move_limit,
            scoring_depth,
            score_sum: 0.0,
            prev_black_move: None,
        }
    }
}

impl GameObserver for MinimaxObserver {
    fn on_move(&mut self, stone: Stone, position: PositionId, board: &Board) -> ControlFlow<()> {
        if self
            .move_limit
            .is_some_and(|limit| board.move_count() >= limit)
        {
            return ControlFlow::Break(());
        }
        if let Some(depth) = self.scoring_depth {
            match stone {
                Stone::Black => {
                    self.prev_black_move = Some((*board, position));
                }
                Stone::White => {
                    if let Some((ref prev_board, black_pos)) = self.prev_black_move {
                        let before = -score_move(prev_board, Stone::Black, black_pos, depth);
                        let after = score_move(board, Stone::White, position, depth);
                        let delta = (after - before).normalized();
                        let occupied =
                            *board.bitboard(Stone::Black) | *board.bitboard(Stone::White);
                        let proximity = if occupied.expand_nearby().is_set(position) {
                            PROXIMITY_BONUS
                        } else {
                            0.0
                        };
                        trace!(
                            row = position.row(),
                            col = position.col(),
                            move_count = board.move_count(),
                            before = before.normalized(),
                            after = after.normalized(),
                            delta,
                            proximity,
                            "White move scored"
                        );
                        self.score_sum += delta + proximity;
                    }
                }
            }
        }
        ControlFlow::Continue(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{Freestyle, Play};
    use crate::outcome::Outcome;
    use crate::test_utils::ScriptedStrategy;

    #[test]
    fn minimax_depths_sorted_and_deduped() {
        let evaluator = MinimaxFitness::new(Freestyle.into(), vec![6, 2, 4, 2], None);

        assert_eq!(evaluator.depths, vec![2, 4, 6]);
    }

    #[test]
    fn move_limit_returns_none_when_game_exceeds_limit() {
        let (black, white) = ScriptedStrategy::black_wins(); // Black wins in 9 turns; limit 5 cuts it off
        let mut board = Board::new();
        let mut observer = MinimaxObserver::new(Some(5), None);
        let mut rng = fastrand::Rng::with_seed(42);

        let result = Freestyle.play_from(&mut board, &black, &white, &mut observer, &mut rng);

        assert!(result.is_none());
    }

    #[test]
    fn move_limit_returns_result_when_game_finishes_before_limit() {
        let (black, white) = ScriptedStrategy::black_wins(); // Black wins in 9 turns; limit 20 allows completion
        let mut board = Board::new();
        let mut observer = MinimaxObserver::new(Some(20), None);
        let mut rng = fastrand::Rng::with_seed(42);

        let result = Freestyle.play_from(&mut board, &black, &white, &mut observer, &mut rng);

        assert!(result.is_some());
        assert_eq!(board.move_count(), 9);
        assert_eq!(board.outcome(), Some(Outcome::BlackWins));
    }

    #[test]
    fn move_limit_none_plays_to_completion() {
        let (black, white) = ScriptedStrategy::black_wins();
        let mut board = Board::new();
        let mut observer = MinimaxObserver::new(None, None);
        let mut rng = fastrand::Rng::with_seed(42);

        let result = Freestyle.play_from(&mut board, &black, &white, &mut observer, &mut rng);

        assert!(result.is_some());
        assert_eq!(board.move_count(), 9);
        assert_eq!(board.outcome(), Some(Outcome::BlackWins));
    }

    #[test]
    fn observer_breaks_when_move_count_reaches_limit() {
        let mut observer = MinimaxObserver::new(Some(4), None);
        let mut board = Board::new();
        let positions = board.empty_position_ids();
        board.place(positions[0], Stone::Black).unwrap();
        board.place(positions[1], Stone::White).unwrap();
        board.place(positions[2], Stone::Black).unwrap();
        board.place(positions[3], Stone::White).unwrap();

        let result = observer.on_move(Stone::Black, positions[4], &board);

        assert!(result.is_break());
    }

    #[test]
    fn observer_continues_below_move_limit() {
        let mut observer = MinimaxObserver::new(Some(10), None);
        let board = Board::new();
        let position = board.empty_position_ids()[0];

        let result = observer.on_move(Stone::Black, position, &board);

        assert!(result.is_continue());
    }

    #[test]
    fn observer_accumulates_score_only_for_white() {
        use crate::position::Position;

        // Four Black stones in a row — blocking at (7,9) yields a non-zero score; Black's move doesn't.
        let mut board = Board::new();
        let pos = |row, col| PositionId::from_position(Position::new(row, col));
        board.place(pos(7, 5), Stone::Black).unwrap();
        board.place(pos(7, 6), Stone::Black).unwrap();
        board.place(pos(7, 7), Stone::Black).unwrap();
        board.place(pos(7, 8), Stone::Black).unwrap();
        let blocking_position = pos(7, 9);

        let mut observer = MinimaxObserver::new(None, Some(1));

        let _ = observer.on_move(Stone::Black, blocking_position, &board);
        let after_black = observer.score_sum;

        let _ = observer.on_move(Stone::White, blocking_position, &board);
        let after_white = observer.score_sum;

        assert_eq!(after_black, 0.0);
        assert_ne!(after_white, 0.0);
    }

    #[test]
    fn observer_no_score_when_scoring_depth_is_none() {
        let mut observer = MinimaxObserver::new(None, None);
        let board = Board::new();
        let position = board.empty_position_ids()[0];

        let _ = observer.on_move(Stone::White, position, &board);

        assert_eq!(observer.score_sum, 0.0);
    }

    #[test]
    fn white_move_without_prior_black_move_skips_scoring() {
        // No stored Black move means there is no baseline to compute a delta from.
        let mut observer = MinimaxObserver::new(None, Some(1));
        let board = Board::new();
        let position = board.empty_position_ids()[0];

        let _ = observer.on_move(Stone::White, position, &board);

        assert_eq!(observer.score_sum, 0.0);
    }

    #[test]
    fn blocking_imminent_win_scores_higher_than_ignoring() {
        use crate::position::Position;

        // Black extends three-in-a-row to an open four; one observer blocks, the other plays corner.
        let mut board_before_black = Board::new();
        let pos = |row, col| PositionId::from_position(Position::new(row, col));
        board_before_black.place(pos(7, 5), Stone::Black).unwrap();
        board_before_black.place(pos(7, 6), Stone::Black).unwrap();
        board_before_black.place(pos(7, 7), Stone::Black).unwrap();
        let black_pos = pos(7, 8);

        let mut board_after_black = board_before_black;
        board_after_black.place(black_pos, Stone::Black).unwrap();

        let scoring_depth = 1;

        let mut blocking_observer = MinimaxObserver::new(None, Some(scoring_depth));
        let _ = blocking_observer.on_move(Stone::Black, black_pos, &board_before_black);
        let _ = blocking_observer.on_move(Stone::White, pos(7, 9), &board_after_black);

        let mut ignoring_observer = MinimaxObserver::new(None, Some(scoring_depth));
        let _ = ignoring_observer.on_move(Stone::Black, black_pos, &board_before_black);
        let _ = ignoring_observer.on_move(Stone::White, pos(0, 0), &board_after_black);

        assert!(
            blocking_observer.score_sum > 0.0,
            "blocking an open-four threat should yield a positive delta (got {})",
            blocking_observer.score_sum,
        );
        assert!(
            blocking_observer.score_sum > ignoring_observer.score_sum,
            "blocking ({}) should outscore ignoring ({}) Black's open-four threat",
            blocking_observer.score_sum,
            ignoring_observer.score_sum,
        );
    }

    #[test]
    fn proximity_bonus_applied_near_stones_not_at_corner() {
        use crate::position::Position;

        // With one Black stone the depth signal is flat, so any score difference
        // comes from the proximity bonus alone.
        let board_before_black = Board::new();
        let black_pos = PositionId::center();
        let mut board_after_black = board_before_black;
        board_after_black.place(black_pos, Stone::Black).unwrap();
        let pos = |row, col| PositionId::from_position(Position::new(row, col));

        let mut nearby_observer = MinimaxObserver::new(None, Some(1));
        let _ = nearby_observer.on_move(Stone::Black, black_pos, &board_before_black);
        let _ = nearby_observer.on_move(Stone::White, pos(7, 6), &board_after_black);

        let mut corner_observer = MinimaxObserver::new(None, Some(1));
        let _ = corner_observer.on_move(Stone::Black, black_pos, &board_before_black);
        let _ = corner_observer.on_move(Stone::White, pos(0, 0), &board_after_black);

        assert!(
            nearby_observer.score_sum > corner_observer.score_sum,
            "nearby ({}) should beat corner ({}) when depth signal is flat",
            nearby_observer.score_sum,
            corner_observer.score_sum,
        );
        // The gap equals exactly PROXIMITY_BONUS because the deltas are identical.
        assert_eq!(
            nearby_observer.score_sum - corner_observer.score_sum,
            PROXIMITY_BONUS,
        );
    }

    #[test]
    fn longer_loss_scores_higher_than_shorter_loss() {
        // Same quality, different lengths — the length component must differentiate them.
        let short = scoring_depth_fitness(0.0, 9, Outcome::BlackWins);
        let long = scoring_depth_fitness(0.0, 21, Outcome::BlackWins);
        assert!(
            long > short,
            "longer loss ({long}) should beat shorter loss ({short})"
        );
    }

    #[test]
    fn faster_win_scores_higher_than_slower_win() {
        let fast = scoring_depth_fitness(0.0, 9, Outcome::WhiteWins);
        let slow = scoring_depth_fitness(0.0, 25, Outcome::WhiteWins);
        assert!(
            fast > slow,
            "faster win ({fast}) should beat slower win ({slow})"
        );
    }

    #[test]
    fn win_scores_higher_than_loss_with_equal_quality() {
        // At equal quality a win beats a loss at any length.  The invariant does not hold
        // across extreme quality differences, but equal quality is the realistic baseline.
        let win = scoring_depth_fitness(0.0, 225, Outcome::WhiteWins);
        let loss = scoring_depth_fitness(0.0, 225, Outcome::BlackWins);
        assert!(
            win > loss,
            "win ({win}) should beat loss ({loss}) at equal quality"
        );
    }

    #[test]
    fn score_accumulates_across_multiple_white_moves() {
        // All four White moves in a 9-turn game are preceded by Black moves, so each contributes.
        let (black, white) = ScriptedStrategy::black_wins();
        let mut board = Board::new();
        let mut observer = MinimaxObserver::new(None, Some(1));
        let mut rng = fastrand::Rng::with_seed(42);

        let _ = Freestyle.play_from(&mut board, &black, &white, &mut observer, &mut rng);

        assert_ne!(observer.score_sum, 0.0);
    }
}
