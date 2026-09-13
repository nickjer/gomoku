use tracing::{debug, info, instrument};

use crate::evolution::fitness_score::FitnessScore;
use crate::game::Game;
use crate::strategy::{EvolvableStrategy, Strategy};
use crate::tournament::{RunTournament, Standing, Tournament};

use super::EvaluateFitness;

/// Evaluates fitness by running a tournament; score equals `population_size − rank`.
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
    fn evaluate<S: EvolvableStrategy>(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Stub;
    use crate::test_utils::FakeEvolvableStrategy;
    use crate::tournament::Scripted;

    fn strategies(rng: &mut fastrand::Rng) -> Vec<FakeEvolvableStrategy> {
        ["a", "b", "c"]
            .into_iter()
            .map(|label| FakeEvolvableStrategy::random(label, rng))
            .collect()
    }

    #[test]
    fn tournament_fitness_scores_by_rank() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategies = strategies(&mut rng);
        let tournament = Scripted::new(vec!["b", "a", "c"]).into();
        let evaluator = TournamentFitness::new(tournament, Stub.into());

        let scores = evaluator.evaluate(&strategies, &mut rng);

        assert_eq!(scores[0], FitnessScore::new(2.0)); // a finished 2nd
        assert_eq!(scores[1], FitnessScore::new(3.0)); // b finished 1st
        assert_eq!(scores[2], FitnessScore::new(1.0)); // c finished 3rd
    }

    #[test]
    fn tournament_fitness_highest_score_for_first_place() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategies = strategies(&mut rng);
        let tournament = Scripted::new(vec!["a", "b", "c"]).into();
        let evaluator = TournamentFitness::new(tournament, Stub.into());

        let scores = evaluator.evaluate(&strategies, &mut rng);

        assert_eq!(scores[0], FitnessScore::new(3.0));
        assert_eq!(scores[1], FitnessScore::new(2.0));
        assert_eq!(scores[2], FitnessScore::new(1.0));
    }
}
