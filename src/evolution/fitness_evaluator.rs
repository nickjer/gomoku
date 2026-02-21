use enum_dispatch::enum_dispatch;
use tracing::{debug, info};

use crate::game::{Game, Play};
use crate::minimax::MinimaxStrategy;
use crate::outcome::Outcome;
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

/// Evaluates fitness by playing each strategy against a minimax opponent.
///
/// Minimax plays as black; the evaluated strategy plays as white.
/// The score is the number of moves played, plus a 1000-point bonus if the
/// strategy wins.
pub struct MinimaxFitness {
    game: Game,
    depth: u32,
}

impl MinimaxFitness {
    #[must_use]
    pub fn new(game: Game, depth: u32) -> Self {
        Self { game, depth }
    }
}

impl EvaluateFitness for MinimaxFitness {
    fn evaluate<S: Strategy>(
        &self,
        strategies: &[S],
        rng: &mut fastrand::Rng,
    ) -> Vec<FitnessScore> {
        let minimax = MinimaxStrategy::new(self.depth);
        strategies
            .iter()
            .map(|strategy| {
                let result = self.game.play(&minimax, strategy, rng);
                let move_count =
                    f32::from(u16::try_from(result.turn_count()).expect("turn count fits in u16"));
                let win_bonus = if result.outcome() == Outcome::WhiteWins {
                    1000.0
                } else {
                    0.0
                };
                info!(
                    label = strategy.label(),
                    move_count,
                    outcome = ?result.outcome(),
                    "Minimax evaluation"
                );
                FitnessScore::new(move_count + win_bonus)
            })
            .collect()
    }
}

/// Enum for polymorphic fitness evaluator dispatch.
#[enum_dispatch(EvaluateFitness)]
#[derive(strum::Display)]
pub enum FitnessEvaluator {
    TournamentFitness,
    ThreatDefenseFitness,
    MinimaxFitness,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Stub;
    use crate::test_utils::StubStrategy;
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
}
