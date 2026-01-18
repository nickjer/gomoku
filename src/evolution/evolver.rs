use crate::strategy::EvolvableStrategy;
use crate::tournament::RunTournament;

use super::crossover::Crossover;
use super::mutation::Mutation;
use super::selection::Selection;
use super::{Individual, Population, evaluate};

/// Configuration and execution of the evolutionary algorithm.
pub struct Evolver<T> {
    population_size: usize,
    generations: u32,
    elitism: usize,
    crossover: Crossover,
    crossover_rate: f64,
    mutation: Mutation,
    mutation_rate: f64,
    selection: Selection,
    tournament: T,
}

impl<T: RunTournament> Evolver<T> {
    pub fn new(tournament: T) -> Self {
        Self {
            population_size: 32,
            generations: 10,
            elitism: 2,
            crossover: Crossover::default(),
            crossover_rate: 0.8,
            mutation: Mutation::default(),
            mutation_rate: 0.1,
            selection: Selection::default(),
            tournament,
        }
    }

    #[must_use]
    pub fn population_size(mut self, size: usize) -> Self {
        self.population_size = size;
        self
    }

    #[must_use]
    pub fn generations(mut self, generations: u32) -> Self {
        self.generations = generations;
        self
    }

    #[must_use]
    pub fn elitism(mut self, elitism: usize) -> Self {
        self.elitism = elitism;
        self
    }

    #[must_use]
    pub fn crossover(mut self, crossover: Crossover) -> Self {
        self.crossover = crossover;
        self
    }

    #[must_use]
    pub fn crossover_rate(mut self, rate: f64) -> Self {
        self.crossover_rate = rate;
        self
    }

    #[must_use]
    pub fn mutation(mut self, mutation: Mutation) -> Self {
        self.mutation = mutation;
        self
    }

    #[must_use]
    pub fn mutation_rate(mut self, rate: f64) -> Self {
        self.mutation_rate = rate;
        self
    }

    #[must_use]
    pub fn selection(mut self, selection: Selection) -> Self {
        self.selection = selection;
        self
    }

    /// Runs the evolutionary algorithm and returns the final population.
    pub fn evolve<S: EvolvableStrategy>(&self, rng: &mut fastrand::Rng) -> Population<S> {
        let strategies = self.create_initial_population::<S>(rng);
        let individuals = evaluate(strategies, &self.tournament, rng);
        let mut population = Population::new(individuals);

        for _ in 0..self.generations {
            let new_strategies = self.create_next_generation(&population, rng);
            let new_individuals = evaluate(new_strategies, &self.tournament, rng);
            population = population.next_generation(new_individuals);
        }

        population
    }

    fn create_initial_population<S: EvolvableStrategy>(&self, rng: &mut fastrand::Rng) -> Vec<S> {
        (0..self.population_size)
            .map(|i| S::random(format!("gen0_{i}"), rng))
            .collect()
    }

    fn create_next_generation<S: EvolvableStrategy>(
        &self,
        population: &Population<S>,
        rng: &mut fastrand::Rng,
    ) -> Vec<S> {
        let generation = population.generation() + 1;
        let individuals = population.individuals();
        let mut new_strategies = Vec::with_capacity(self.population_size);

        // Elitism: keep top performers
        for (i, individual) in individuals.iter().take(self.elitism).enumerate() {
            let genes = individual.strategy().genes().to_vec();
            new_strategies.push(S::from_genes(format!("gen{generation}_elite{i}"), genes));
        }

        // Fill rest with offspring
        new_strategies.extend(
            (self.elitism..self.population_size)
                .map(|i| self.create_offspring(individuals, generation, i, rng)),
        );

        new_strategies
    }

    fn create_offspring<S: EvolvableStrategy>(
        &self,
        individuals: &[Individual<S>],
        generation: u32,
        index: usize,
        rng: &mut fastrand::Rng,
    ) -> S {
        let parent1 = self.selection.select(individuals, rng);
        let parent2 = self.selection.select(individuals, rng);

        let mut child_genes = if rng.f64() < self.crossover_rate {
            self.crossover
                .crossover(parent1.strategy().genes(), parent2.strategy().genes(), rng)
        } else {
            parent1.strategy().genes().to_vec()
        };

        if rng.f64() < self.mutation_rate {
            child_genes = self.mutation.mutate(&child_genes, rng);
        }

        S::from_genes(format!("gen{generation}_{index}"), child_genes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::Strategy;
    use crate::test_utils::{FakeEvolvableStrategy, InputOrderTournament};

    #[test]
    fn evolve_returns_population_with_correct_size() {
        let evolver = Evolver::new(InputOrderTournament)
            .population_size(4)
            .generations(0);
        let mut rng = fastrand::Rng::with_seed(42);

        let population = evolver.evolve::<FakeEvolvableStrategy>(&mut rng);

        assert_eq!(population.individuals().len(), 4);
    }

    #[test]
    fn evolve_with_zero_generations_returns_initial_population() {
        let evolver = Evolver::new(InputOrderTournament)
            .population_size(4)
            .generations(0);
        let mut rng = fastrand::Rng::with_seed(42);

        let population = evolver.evolve::<FakeEvolvableStrategy>(&mut rng);

        assert_eq!(population.generation(), 0);
    }

    #[test]
    fn evolve_increments_generation() {
        let evolver = Evolver::new(InputOrderTournament)
            .population_size(4)
            .generations(3);
        let mut rng = fastrand::Rng::with_seed(42);

        let population = evolver.evolve::<FakeEvolvableStrategy>(&mut rng);

        assert_eq!(population.generation(), 3);
    }

    #[test]
    fn evolve_is_deterministic_with_same_seed() {
        let evolver1 = Evolver::new(InputOrderTournament)
            .population_size(4)
            .generations(2);
        let evolver2 = Evolver::new(InputOrderTournament)
            .population_size(4)
            .generations(2);

        let pop1 = evolver1.evolve::<FakeEvolvableStrategy>(&mut fastrand::Rng::with_seed(42));
        let pop2 = evolver2.evolve::<FakeEvolvableStrategy>(&mut fastrand::Rng::with_seed(42));

        let labels1: Vec<_> = pop1
            .individuals()
            .iter()
            .map(|i| i.strategy().label())
            .collect();
        let labels2: Vec<_> = pop2
            .individuals()
            .iter()
            .map(|i| i.strategy().label())
            .collect();
        assert_eq!(labels1, labels2);
    }

    #[test]
    fn evolve_preserves_elites() {
        let evolver = Evolver::new(InputOrderTournament)
            .population_size(4)
            .elitism(2)
            .generations(1)
            .crossover_rate(1.0)
            .mutation_rate(0.0);
        let mut rng = fastrand::Rng::with_seed(42);

        let population = evolver.evolve::<FakeEvolvableStrategy>(&mut rng);

        // With InputOrderTournament, gen0_0 and gen0_1 have highest fitness
        // They should be preserved as elites in gen1
        let labels: Vec<_> = population
            .individuals()
            .iter()
            .map(|i| i.strategy().label())
            .collect();
        assert!(labels.iter().any(|l| l.contains("elite")));
    }

    #[test]
    fn builder_methods_set_values() {
        let evolver = Evolver::new(InputOrderTournament)
            .population_size(64)
            .generations(20)
            .elitism(4)
            .crossover_rate(0.9)
            .mutation_rate(0.2);

        assert_eq!(evolver.population_size, 64);
        assert_eq!(evolver.generations, 20);
        assert_eq!(evolver.elitism, 4);
        assert!((evolver.crossover_rate - 0.9).abs() < f64::EPSILON);
        assert!((evolver.mutation_rate - 0.2).abs() < f64::EPSILON);
    }
}
