use super::{HasFitness, RunSelection};

/// How tournament selection draws the individuals it compares.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum TournamentMode {
    /// The same individual may be drawn more than once.
    WithReplacement,
    /// Every drawn individual is distinct.
    WithoutReplacement,
}

/// Tournament selection: selects the best individual from a random sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tournament {
    size: usize,
    mode: TournamentMode,
}

impl Tournament {
    #[must_use]
    pub fn new(size: usize, mode: TournamentMode) -> Self {
        Self { size, mode }
    }

    fn select_internal<'a, T: HasFitness>(
        &self,
        population: &'a [T],
        rng: &mut fastrand::Rng,
    ) -> &'a T {
        assert!(
            !population.is_empty(),
            "cannot select from empty population"
        );
        assert!(
            population.len() >= self.size,
            "population size {} is smaller than tournament size {}",
            population.len(),
            self.size
        );

        match self.mode {
            TournamentMode::WithReplacement => self.select_with_replacement(population, rng),
            TournamentMode::WithoutReplacement => self.select_without_replacement(population, rng),
        }
    }

    fn select_with_replacement<'a, T: HasFitness>(
        &self,
        population: &'a [T],
        rng: &mut fastrand::Rng,
    ) -> &'a T {
        (0..self.size)
            .map(|_| &population[rng.usize(..population.len())])
            .max_by_key(|ind| ind.fitness())
            .expect("tournament size is non-zero")
    }

    fn select_without_replacement<'a, T: HasFitness>(
        &self,
        population: &'a [T],
        rng: &mut fastrand::Rng,
    ) -> &'a T {
        let mut indices: Vec<usize> = (0..population.len()).collect();
        rng.shuffle(&mut indices);

        indices[..self.size]
            .iter()
            .map(|&i| &population[i])
            .max_by_key(|ind| ind.fitness())
            .expect("tournament size is non-zero")
    }
}

impl Default for Tournament {
    fn default() -> Self {
        Self::new(3, TournamentMode::WithReplacement)
    }
}

impl RunSelection for Tournament {
    fn select<'a, T: HasFitness>(&self, population: &'a [T], rng: &mut fastrand::Rng) -> &'a T {
        self.select_internal(population, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolution::fitness_score::FitnessScore;
    use crate::test_utils::TestIndividual;

    fn population(fitnesses: &[f32]) -> Vec<TestIndividual> {
        fitnesses
            .iter()
            .map(|&fit| TestIndividual {
                fitness: FitnessScore::new(fit),
            })
            .collect()
    }

    mod with_replacement {
        use super::*;

        fn selector(size: usize) -> Tournament {
            Tournament::new(size, TournamentMode::WithReplacement)
        }

        #[test]
        fn returns_individual_from_population() {
            let pop = population(&[10.0, 20.0, 30.0, 40.0, 50.0]);
            let mut rng = fastrand::Rng::with_seed(42);

            let result = selector(3).select(&pop, &mut rng);

            assert!(pop.iter().any(|ind| ind.fitness() == result.fitness()));
        }

        #[test]
        fn full_tournament_returns_best() {
            let pop = population(&[100.0, 1.0, 2.0, 3.0, 4.0]);
            let mut rng = fastrand::Rng::with_seed(42);

            let result = selector(5).select(&pop, &mut rng);

            assert_eq!(result.fitness(), FitnessScore::new(100.0));
        }

        #[test]
        fn deterministic_with_same_seed() {
            let pop = population(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
            let sel = selector(3);

            let results1: Vec<_> = {
                let mut rng = fastrand::Rng::with_seed(42);
                (0..5)
                    .map(|_| sel.select(&pop, &mut rng).fitness())
                    .collect()
            };
            let results2: Vec<_> = {
                let mut rng = fastrand::Rng::with_seed(42);
                (0..5)
                    .map(|_| sel.select(&pop, &mut rng).fitness())
                    .collect()
            };

            assert_eq!(results1, results2);
        }

        #[test]
        fn different_seeds_produce_different_results() {
            let pop = population(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
            let sel = selector(2);

            let results1: Vec<_> = {
                let mut rng = fastrand::Rng::with_seed(42);
                (0..10)
                    .map(|_| sel.select(&pop, &mut rng).fitness())
                    .collect()
            };
            let results2: Vec<_> = {
                let mut rng = fastrand::Rng::with_seed(99);
                (0..10)
                    .map(|_| sel.select(&pop, &mut rng).fitness())
                    .collect()
            };

            assert_ne!(results1, results2);
        }

        #[test]
        fn larger_tournament_increases_selection_pressure() {
            let pop = population(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]);

            let small_avg: f32 = {
                let mut rng = fastrand::Rng::with_seed(42);
                let sum: f32 = (0..50)
                    .map(|_| selector(2).select(&pop, &mut rng).fitness().value())
                    .sum();
                sum / 50.0
            };
            let large_avg: f32 = {
                let mut rng = fastrand::Rng::with_seed(42);
                let sum: f32 = (0..50)
                    .map(|_| selector(8).select(&pop, &mut rng).fitness().value())
                    .sum();
                sum / 50.0
            };

            assert!(large_avg > small_avg);
        }

        #[test]
        fn may_miss_best_when_tournament_equals_population() {
            let pop = population(&[100.0, 1.0, 1.0, 1.0, 1.0]);
            let sel = selector(5);

            let mut found_non_best = false;
            for seed in 0..1000 {
                let mut rng = fastrand::Rng::with_seed(seed);
                if sel.select(&pop, &mut rng).fitness() != FitnessScore::new(100.0) {
                    found_non_best = true;
                    break;
                }
            }
            assert!(
                found_non_best,
                "with replacement should sometimes miss the best"
            );
        }
    }

    mod without_replacement {
        use super::*;

        fn selector(size: usize) -> Tournament {
            Tournament::new(size, TournamentMode::WithoutReplacement)
        }

        #[test]
        fn returns_individual_from_population() {
            let pop = population(&[10.0, 20.0, 30.0, 40.0, 50.0]);
            let mut rng = fastrand::Rng::with_seed(42);

            let result = selector(3).select(&pop, &mut rng);

            assert!(pop.iter().any(|ind| ind.fitness() == result.fitness()));
        }

        #[test]
        fn full_tournament_returns_best() {
            let pop = population(&[100.0, 1.0, 2.0, 3.0, 4.0]);
            let mut rng = fastrand::Rng::with_seed(42);

            let result = selector(5).select(&pop, &mut rng);

            assert_eq!(result.fitness(), FitnessScore::new(100.0));
        }

        #[test]
        fn deterministic_with_same_seed() {
            let pop = population(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
            let sel = selector(3);

            let results1: Vec<_> = {
                let mut rng = fastrand::Rng::with_seed(42);
                (0..5)
                    .map(|_| sel.select(&pop, &mut rng).fitness())
                    .collect()
            };
            let results2: Vec<_> = {
                let mut rng = fastrand::Rng::with_seed(42);
                (0..5)
                    .map(|_| sel.select(&pop, &mut rng).fitness())
                    .collect()
            };

            assert_eq!(results1, results2);
        }

        #[test]
        fn different_seeds_produce_different_results() {
            let pop = population(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
            let sel = selector(2);

            let results1: Vec<_> = {
                let mut rng = fastrand::Rng::with_seed(42);
                (0..10)
                    .map(|_| sel.select(&pop, &mut rng).fitness())
                    .collect()
            };
            let results2: Vec<_> = {
                let mut rng = fastrand::Rng::with_seed(99);
                (0..10)
                    .map(|_| sel.select(&pop, &mut rng).fitness())
                    .collect()
            };

            assert_ne!(results1, results2);
        }

        #[test]
        fn larger_tournament_increases_selection_pressure() {
            let pop = population(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]);

            let small_avg: f32 = {
                let mut rng = fastrand::Rng::with_seed(42);
                let sum: f32 = (0..50)
                    .map(|_| selector(2).select(&pop, &mut rng).fitness().value())
                    .sum();
                sum / 50.0
            };
            let large_avg: f32 = {
                let mut rng = fastrand::Rng::with_seed(42);
                let sum: f32 = (0..50)
                    .map(|_| selector(8).select(&pop, &mut rng).fitness().value())
                    .sum();
                sum / 50.0
            };

            assert!(large_avg > small_avg);
        }

        #[test]
        fn always_finds_best_when_tournament_equals_population() {
            let pop = population(&[100.0, 1.0, 1.0, 1.0, 1.0]);
            let sel = selector(5);

            for seed in 0..100 {
                let mut rng = fastrand::Rng::with_seed(seed);
                assert_eq!(
                    sel.select(&pop, &mut rng).fitness(),
                    FitnessScore::new(100.0)
                );
            }
        }
    }

    #[test]
    fn tournament_size_one_returns_random() {
        let pop = population(&[10.0, 20.0, 30.0, 40.0, 50.0]);
        let sel = Tournament::new(1, TournamentMode::WithReplacement);

        let results: Vec<_> = (0..100)
            .map(|seed| {
                let mut rng = fastrand::Rng::with_seed(seed);
                sel.select(&pop, &mut rng).fitness().value().to_bits()
            })
            .collect();

        let unique: std::collections::HashSet<_> = results.iter().collect();
        assert!(unique.len() > 1, "should select different individuals");
    }

    #[test]
    #[should_panic(expected = "cannot select from empty population")]
    fn panics_on_empty_population() {
        let pop: Vec<TestIndividual> = vec![];
        let sel = Tournament::new(3, TournamentMode::WithReplacement);
        let mut rng = fastrand::Rng::with_seed(42);

        sel.select(&pop, &mut rng);
    }

    #[test]
    #[should_panic(expected = "population size 2 is smaller than tournament size 5")]
    fn panics_when_population_smaller_than_tournament() {
        let pop = population(&[10.0, 20.0]);
        let sel = Tournament::new(5, TournamentMode::WithReplacement);
        let mut rng = fastrand::Rng::with_seed(42);

        sel.select(&pop, &mut rng);
    }
}
