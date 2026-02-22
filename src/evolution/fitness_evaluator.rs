use enum_dispatch::enum_dispatch;
use tracing::{debug, info};

use crate::board::Board;
use crate::game::Game;
use crate::match_result::MatchResult;
use crate::minimax::MinimaxStrategy;
use crate::outcome::Outcome;
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
    depths: Vec<u32>,
}

impl MinimaxFitness {
    #[must_use]
    pub fn new(mut depths: Vec<u32>) -> Self {
        depths.sort_unstable();
        depths.dedup();
        Self { depths }
    }

    fn evaluate_single(&self, strategy: &dyn Strategy, rng: &mut fastrand::Rng) -> f32 {
        let mut best_win: Option<u32> = None;
        let mut best_loss: Option<u32> = None;

        for &depth in &self.depths {
            let minimax = MinimaxStrategy::new(depth);
            let Some(result) = play_with_move_limit(&minimax, strategy, best_win, rng) else {
                debug!(label = strategy.label(), depth, "Minimax depth cutoff");
                continue;
            };

            let turn_count = result.turn_count();
            let minimax_won = result.outcome() != Outcome::WhiteWins;

            if minimax_won {
                best_win = Some(best_win.map_or(turn_count, |prev| prev.min(turn_count)));
            } else {
                best_loss = Some(best_loss.map_or(turn_count, |prev| prev.max(turn_count)));
            }

            debug!(
                label = strategy.label(),
                depth,
                turn_count,
                outcome = ?result.outcome(),
                "Minimax evaluation"
            );
        }

        let turns_to_f32 =
            |turns: u32| f32::from(u16::try_from(turns).expect("turn count fits in u16"));
        match (best_win, best_loss) {
            (Some(turns), _) => turns_to_f32(turns),
            (None, Some(turns)) => 1000.0 - turns_to_f32(turns),
            (None, None) => 0.0,
        }
    }
}

impl EvaluateFitness for MinimaxFitness {
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

/// Plays a Freestyle game between `black` and `white`, returning `None` if
/// `move_limit` is reached before the game finishes naturally.
fn play_with_move_limit(
    black: &dyn Strategy,
    white: &dyn Strategy,
    move_limit: Option<u32>,
    rng: &mut fastrand::Rng,
) -> Option<MatchResult> {
    let mut board = Board::new();
    let mut turn_count: u32 = 0;

    while !board.is_finished() {
        if move_limit.is_some_and(|limit| turn_count >= limit) {
            return None;
        }

        let (strategy, stone): (&dyn Strategy, Stone) = if turn_count.is_multiple_of(2) {
            (black, Stone::Black)
        } else {
            (white, Stone::White)
        };

        let position_id = strategy.choose_move(stone, &board, rng);
        board
            .place(position_id, stone)
            .expect("strategy returned invalid move");
        turn_count += 1;
    }

    let outcome = board.outcome().expect("game finished without outcome");
    Some(MatchResult::new(
        outcome,
        black.label().to_string(),
        white.label().to_string(),
        turn_count,
        board.to_string(),
    ))
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
    use crate::game::Stub;
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
    fn move_limit_returns_none_when_game_exceeds_limit() {
        // Black wins in 9 turns; limit of 5 should cut it off
        let (black, white) = ScriptedStrategy::black_wins();
        let mut rng = fastrand::Rng::with_seed(42);

        let result = play_with_move_limit(&black, &white, Some(5), &mut rng);

        assert!(result.is_none());
    }

    #[test]
    fn move_limit_returns_result_when_game_finishes_before_limit() {
        // Black wins in 9 turns; limit of 20 allows completion
        let (black, white) = ScriptedStrategy::black_wins();
        let mut rng = fastrand::Rng::with_seed(42);

        let result = play_with_move_limit(&black, &white, Some(20), &mut rng);

        let result = result.expect("game should finish before limit");
        assert_eq!(result.outcome(), Outcome::BlackWins);
        assert_eq!(result.turn_count(), 9);
    }

    #[test]
    fn move_limit_none_plays_to_completion() {
        let (black, white) = ScriptedStrategy::black_wins();
        let mut rng = fastrand::Rng::with_seed(42);

        let result = play_with_move_limit(&black, &white, None, &mut rng);

        let result = result.expect("game should complete without limit");
        assert_eq!(result.outcome(), Outcome::BlackWins);
        assert_eq!(result.turn_count(), 9);
    }

    #[test]
    fn minimax_depths_sorted_and_deduped() {
        let evaluator = MinimaxFitness::new(vec![6, 2, 4, 2]);

        assert_eq!(evaluator.depths, vec![2, 4, 6]);
    }
}
