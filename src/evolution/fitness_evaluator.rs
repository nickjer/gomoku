use tracing::instrument;

use crate::game::Game;
use crate::strategy::Strategy;
use crate::tournament::{RunTournament, Tournament};

use super::Individual;

/// Evaluates fitness by running a tournament.
///
/// Returns individuals sorted by rank (best first), with fitness = `population_size - rank`.
///
/// # Panics
///
/// Panics if the tournament returns duplicate or invalid strategy indices.
#[instrument(skip_all, fields(population_size = strategies.len(), %tournament))]
pub fn evaluate<S: Strategy>(
    strategies: Vec<S>,
    tournament: &Tournament,
    game: &Game,
    rng: &mut fastrand::Rng,
) -> Vec<Individual<S>> {
    let standings = tournament.run(&strategies, game, rng);
    let population_size = strategies.len();

    let mut strategies: Vec<Option<S>> = strategies.into_iter().map(Some).collect();

    standings
        .into_iter()
        .enumerate()
        .map(|(rank, standing)| {
            let strategy = strategies[standing.strategy_index()]
                .take()
                .expect("each strategy used exactly once");
            let fitness =
                u32::try_from(population_size - rank).expect("population size fits in u32");
            Individual::new(strategy, fitness)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolution::selection::HasFitness;
    use crate::game::Stub;
    use crate::test_utils::StubStrategy;
    use crate::tournament::Scripted;

    fn stub_game() -> Game {
        Stub.into()
    }

    #[test]
    fn returns_individuals_sorted_by_rank() {
        let strategies = vec![
            StubStrategy::new("a"),
            StubStrategy::new("b"),
            StubStrategy::new("c"),
        ];
        let tournament = Scripted::new(vec!["b", "a", "c"]).into();
        let mut rng = fastrand::Rng::with_seed(42);

        let individuals = evaluate(strategies, &tournament, &stub_game(), &mut rng);

        assert_eq!(individuals[0].strategy().label(), "b");
        assert_eq!(individuals[1].strategy().label(), "a");
        assert_eq!(individuals[2].strategy().label(), "c");
    }

    #[test]
    fn fitness_is_population_size_minus_rank() {
        let strategies = vec![
            StubStrategy::new("a"),
            StubStrategy::new("b"),
            StubStrategy::new("c"),
        ];
        let tournament = Scripted::new(vec!["a", "b", "c"]).into();
        let mut rng = fastrand::Rng::with_seed(42);

        let individuals = evaluate(strategies, &tournament, &stub_game(), &mut rng);

        assert_eq!(individuals[0].fitness(), 3); // population_size - 0
        assert_eq!(individuals[1].fitness(), 2); // population_size - 1
        assert_eq!(individuals[2].fitness(), 1); // population_size - 2
    }

    #[test]
    fn preserves_strategies() {
        let strategies = vec![
            StubStrategy::new("a"),
            StubStrategy::new("b"),
            StubStrategy::new("c"),
        ];
        let tournament = Scripted::new(vec!["c", "a", "b"]).into();
        let mut rng = fastrand::Rng::with_seed(42);

        let individuals = evaluate(strategies, &tournament, &stub_game(), &mut rng);
        let mut labels: Vec<_> = individuals.iter().map(|i| i.strategy().label()).collect();
        labels.sort();

        assert_eq!(labels, vec!["a", "b", "c"]);
    }
}
