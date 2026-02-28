use std::ops::ControlFlow;

use enum_dispatch::enum_dispatch;
use tracing::{debug, info, instrument, trace};

use crate::board::Board;
use crate::game::{Game, GameObserver, Play};
use crate::minimax::{MinimaxStrategy, score_move};
use crate::outcome::Outcome;
use crate::position_id::PositionId;
use crate::stone::Stone;
use crate::strategy::Strategy;
use crate::threat::{generate_threat_scenarios, test_defense};
use crate::tournament::{RunTournament, Standing, Tournament};

use super::fitness_score::FitnessScore;

/// Trait for computing fitness scores for a population of strategies.
#[enum_dispatch]
pub trait EvaluateFitness {
    fn evaluate<S: Strategy>(&self, strategies: &[S], rng: &mut fastrand::Rng)
    -> Vec<FitnessScore>;
}

/// Evaluates fitness by running a tournament.
///
/// Scores are `population_size - rank` (1-indexed, winner gets highest).
#[derive(Default)]
pub struct TournamentFitness {
    tournament: Tournament,
    game: Game,
}

impl TournamentFitness {
    #[must_use]
    pub fn new(tournament: Tournament, game: Game) -> Self {
        Self { tournament, game }
    }
}

impl EvaluateFitness for TournamentFitness {
    #[instrument(name = "TournamentFitness", skip_all)]
    fn evaluate<S: Strategy>(
        &self,
        strategies: &[S],
        rng: &mut fastrand::Rng,
    ) -> Vec<FitnessScore> {
        let standings = self.tournament.run(strategies, &self.game, rng);
        log_standings(&standings, strategies);
        let population_size = strategies.len();
        let mut scores = vec![FitnessScore::new(0.0); population_size];
        for (rank, standing) in standings.into_iter().enumerate() {
            let fitness =
                u16::try_from(population_size - rank).expect("population size fits in u16");
            scores[standing.strategy_index()] = FitnessScore::new(f32::from(fitness));
        }
        scores
    }
}

fn log_standings<S: Strategy>(standings: &[Standing], strategies: &[S]) {
    let top_count = standings.len().min(5);
    let bottom_start = if standings.len() > 10 {
        standings.len() - 5
    } else {
        top_count
    };

    for (rank, standing) in standings.iter().enumerate() {
        let label = strategies[standing.strategy_index()].label();
        let wins = standing.wins();
        let losses = standing.losses();
        let draws = standing.draws();
        let byes = standing.byes();

        if rank < top_count || rank >= bottom_start {
            info!(
                rank = rank + 1,
                label, wins, losses, draws, byes, "Standing"
            );
        } else {
            debug!(
                rank = rank + 1,
                label, wins, losses, draws, byes, "Standing"
            );
        }
    }
}

/// Evaluates fitness using only defensive capability testing (threat blocking).
pub struct ThreatDefenseFitness;

impl EvaluateFitness for ThreatDefenseFitness {
    #[instrument(name = "ThreatDefenseFitness", skip_all)]
    fn evaluate<S: Strategy>(
        &self,
        strategies: &[S],
        rng: &mut fastrand::Rng,
    ) -> Vec<FitnessScore> {
        let scenarios = generate_threat_scenarios(rng);
        strategies
            .iter()
            .map(|strategy| FitnessScore::new(test_defense(strategy, &scenarios, rng)))
            .collect()
    }
}

/// Evaluates fitness by playing each strategy against minimax opponents at
/// multiple depths. The score reflects the strongest minimax performance:
///
/// - If any minimax wins: use the quickest win (fewest moves).
/// - If all minimax lose: use the one that lasted longest (most moves).
///
/// Depths are iterated smallest to largest. Each subsequent depth is cut off
/// at the best turn count so far (deeper search that can't improve is skipped).
///
/// Minimax plays as black; the evaluated strategy plays as white.
/// Score: if the strategy wins, `1000 - turn_count`; otherwise `turn_count`.
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

    /// The earliest the first player (minimax/black) can win: 5 stones placed
    /// on turns 1, 3, 5, 7, 9.
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
                    let white_moves = f32::from(
                        u16::try_from(r.turn_count / 2).expect("white move count fits u16"),
                    );
                    (r.score_sum / white_moves + 1.0) * 500.0
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

fn turns_as_f32(turns: usize) -> f32 {
    f32::from(u16::try_from(turns).expect("turn count fits in u16"))
}

/// Compact result used within [`MinimaxFitness::evaluate_single`] to track the best
/// minimax win and best strategy win across depth iterations.
struct EvalResult {
    turn_count: usize,
    outcome: Outcome,
    /// Sum of per-move contributions for White's moves; `0.0` when `scoring_depth` is `None`.
    /// Each contribution is the score delta plus [`PROXIMITY_BONUS`] when the move is within
    /// proximity of existing stones.
    score_sum: f32,
}

/// Tiebreaker added to each White move's contribution when the move is within
/// [`PROXIMITY_RADIUS`](crate::bitboard::PROXIMITY_RADIUS) of any existing stone.
///
/// With very few stones on the board the `before`/`after` delta is completely flat: Black has
/// four independent build directions and any single White placement blocks at most one, so every
/// position yields the same evaluation at any search depth.  This constant injects a gradient so
/// "near the action" beats "far corner" even when the depth signal is uninformative.
///
/// The value sits an order of magnitude above the flat early-game noise floor (~0.0001 at
/// depth 4) and well below the smallest meaningful blocking signal at the recommended depth of
/// 4+ (~0.009 for an open-two).  Numerically equal to `Score::OPEN_THREE / Score::WIN`.
const PROXIMITY_BONUS: f32 = 0.001;

/// Per-game observer for [`MinimaxFitness`]. Enforces an optional move limit and
/// accumulates per-move score contributions for White's moves when `scoring_depth` is set.
/// Each contribution is `(after − before).normalized() + proximity`, where `before` is
/// `-score_move(prev_board, Black, black_pos, depth)`, `after` is
/// `score_move(board, White, position, depth)`, and `proximity` is [`PROXIMITY_BONUS`] when
/// `position` is within proximity of existing stones and `0` otherwise.
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

/// Enum for polymorphic fitness evaluator dispatch.
#[enum_dispatch(EvaluateFitness)]
#[derive(strum::Display)]
#[allow(clippy::enum_variant_names)]
pub enum FitnessEvaluator {
    TournamentFitness,
    ThreatDefenseFitness,
    MinimaxFitness,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{Freestyle, Play, Stub};
    use crate::outcome::Outcome;
    use crate::test_utils::{ScriptedStrategy, StubStrategy};
    use crate::tournament::Scripted;

    #[test]
    fn tournament_fitness_scores_by_rank() {
        let strategies = vec![
            StubStrategy::new("a"),
            StubStrategy::new("b"),
            StubStrategy::new("c"),
        ];
        let tournament = Scripted::new(vec!["b", "a", "c"]).into();
        let evaluator = TournamentFitness::new(tournament, Stub.into());
        let mut rng = fastrand::Rng::with_seed(42);

        let scores = evaluator.evaluate(&strategies, &mut rng);

        // Indexed by original position: a=2nd (score 2), b=1st (score 3), c=3rd (score 1)
        assert_eq!(scores[0], FitnessScore::new(2.0)); // a
        assert_eq!(scores[1], FitnessScore::new(3.0)); // b
        assert_eq!(scores[2], FitnessScore::new(1.0)); // c
    }

    #[test]
    fn tournament_fitness_highest_score_for_first_place() {
        let strategies = vec![
            StubStrategy::new("a"),
            StubStrategy::new("b"),
            StubStrategy::new("c"),
        ];
        let tournament = Scripted::new(vec!["a", "b", "c"]).into();
        let evaluator = TournamentFitness::new(tournament, Stub.into());
        let mut rng = fastrand::Rng::with_seed(42);

        let scores = evaluator.evaluate(&strategies, &mut rng);

        assert_eq!(scores[0], FitnessScore::new(3.0)); // population_size - 0
        assert_eq!(scores[1], FitnessScore::new(2.0));
        assert_eq!(scores[2], FitnessScore::new(1.0));
    }

    #[test]
    fn minimax_depths_sorted_and_deduped() {
        let evaluator = MinimaxFitness::new(Freestyle.into(), vec![6, 2, 4, 2], None);

        assert_eq!(evaluator.depths, vec![2, 4, 6]);
    }

    #[test]
    fn move_limit_returns_none_when_game_exceeds_limit() {
        // Black wins in 9 turns; limit of 5 should cut it off
        let (black, white) = ScriptedStrategy::black_wins();
        let mut board = Board::new();
        let mut observer = MinimaxObserver::new(Some(5), None);
        let mut rng = fastrand::Rng::with_seed(42);

        let result = Freestyle.play_from(&mut board, &black, &white, &mut observer, &mut rng);

        assert!(result.is_none());
    }

    #[test]
    fn move_limit_returns_result_when_game_finishes_before_limit() {
        // Black wins in 9 turns; limit of 20 allows completion
        let (black, white) = ScriptedStrategy::black_wins();
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

        // Four Black stones in a row — White must block at (7, 9) to prevent a win.
        // That threat gives the blocking move a non-zero minimax score at depth 1.
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
        // White moves first — no Black move has been stored yet, so no delta
        // can be computed and score_sum must stay zero.
        let mut observer = MinimaxObserver::new(None, Some(1));
        let board = Board::new();
        let position = board.empty_position_ids()[0];

        let _ = observer.on_move(Stone::White, position, &board);

        assert_eq!(observer.score_sum, 0.0);
    }

    #[test]
    fn blocking_imminent_win_scores_higher_than_ignoring() {
        use crate::position::Position;

        // Black has three stones in a row; Black's next move (7,8) creates an
        // open four — an immediate winning threat.  Both observers see the same
        // `before` (derived from the board before Black's move).  The blocking
        // observer responds at (7,9); the ignoring observer plays the corner.
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

        // Board with only 1 Black stone at center.  The depth signal is
        // completely flat at any depth (all positions score identically), so
        // the entire score difference between the two observers must come from
        // the proximity bonus alone.
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
        // Deltas are identical (flat signal), so the gap is exactly PROXIMITY_BONUS.
        assert_eq!(
            nearby_observer.score_sum - corner_observer.score_sum,
            PROXIMITY_BONUS,
        );
    }

    #[test]
    fn score_accumulates_across_multiple_white_moves() {
        // Play a full scripted game (Black wins in 9 turns → 4 White moves).
        // Each White move is preceded by a Black move, so all four contribute
        // to score_sum via the stored prev_black_move.
        let (black, white) = ScriptedStrategy::black_wins();
        let mut board = Board::new();
        let mut observer = MinimaxObserver::new(None, Some(1));
        let mut rng = fastrand::Rng::with_seed(42);

        let _ = Freestyle.play_from(&mut board, &black, &white, &mut observer, &mut rng);

        assert_ne!(observer.score_sum, 0.0);
    }
}
