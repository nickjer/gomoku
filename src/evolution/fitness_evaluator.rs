use crate::strategy::Strategy;
use crate::tournament::RunTournament;

use super::Individual;

/// Evaluates fitness by running a tournament.
///
/// Returns individuals sorted by rank (best first), with fitness = `population_size - rank`.
///
/// # Panics
///
/// Panics if the tournament returns duplicate or invalid strategy indices.
pub fn evaluate<S: Strategy, T: RunTournament>(
    strategies: Vec<S>,
    tournament: &T,
    rng: &mut fastrand::Rng,
) -> Vec<Individual<S>> {
    let standings = tournament.run(&strategies, rng);
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
    use crate::test_utils::{ScriptedTournament, StubStrategy};

    #[test]
    fn returns_individuals_sorted_by_rank() {
        let strategies = vec![
            StubStrategy::new("a"),
            StubStrategy::new("b"),
            StubStrategy::new("c"),
        ];
        let tournament = ScriptedTournament::new(vec!["b", "a", "c"]);
        let mut rng = fastrand::Rng::with_seed(42);

        let individuals = evaluate(strategies, &tournament, &mut rng);

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
        let tournament = ScriptedTournament::new(vec!["a", "b", "c"]);
        let mut rng = fastrand::Rng::with_seed(42);

        let individuals = evaluate(strategies, &tournament, &mut rng);

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
        let tournament = ScriptedTournament::new(vec!["c", "a", "b"]);
        let mut rng = fastrand::Rng::with_seed(42);

        let individuals = evaluate(strategies, &tournament, &mut rng);
        let mut labels: Vec<_> = individuals.iter().map(|i| i.strategy().label()).collect();
        labels.sort();

        assert_eq!(labels, vec!["a", "b", "c"]);
    }
}
