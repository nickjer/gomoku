use tracing::{debug_span, info_span, instrument};

use crate::game::Game;
use crate::strategy::EvolvableStrategy;
use crate::tournament::Tournament;

use super::crossover::Crossover;
use super::genes::EvolvableGenes;
use super::mutation::Mutation;
use super::selection::{RunSelection, Selection};
use super::{Individual, Population, evaluate};

/// Configuration and execution of the evolutionary algorithm.
pub struct Evolver {
    generations: u32,
    elitism: usize,
    crossover: Crossover,
    crossover_rate: f64,
    mutation: Mutation,
    mutation_rate: f64,
    selection: Selection,
    tournament: Tournament,
    game: Game,
}

impl Default for Evolver {
    fn default() -> Self {
        Self::new()
    }
}

impl Evolver {
    #[must_use]
    pub fn new() -> Self {
        Self {
            generations: 10,
            elitism: 2,
            crossover: Crossover::default(),
            crossover_rate: 0.8,
            mutation: Mutation::default(),
            mutation_rate: 0.1,
            selection: Selection::default(),
            tournament: Tournament::default(),
            game: Game::default(),
        }
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

    #[must_use]
    pub fn tournament(mut self, tournament: Tournament) -> Self {
        self.tournament = tournament;
        self
    }

    #[must_use]
    pub fn game(mut self, game: Game) -> Self {
        self.game = game;
        self
    }

    /// Runs the evolutionary algorithm and returns the final population.
    pub fn evolve<S: EvolvableStrategy>(
        &self,
        strategies: Vec<S>,
        rng: &mut fastrand::Rng,
    ) -> Population<S> {
        let individuals = evaluate(strategies, &self.tournament, &self.game, rng);
        let mut population = Population::new(individuals);

        for generation in 1..=self.generations {
            let span = info_span!("generation", number = generation);
            let _guard = span.enter();
            let new_strategies = self.create_next_generation(&population, rng);
            let new_individuals = evaluate(new_strategies, &self.tournament, &self.game, rng);
            population = population.next_generation(new_individuals);
        }

        population
    }

    #[instrument(skip_all, fields(population_size = population.individuals().len()))]
    fn create_next_generation<S: EvolvableStrategy>(
        &self,
        population: &Population<S>,
        rng: &mut fastrand::Rng,
    ) -> Vec<S> {
        let generation = population.generation() + 1;
        let individuals = population.individuals();
        let population_size = individuals.len();
        let mut new_strategies = Vec::with_capacity(population_size);

        // Elitism: keep top performers
        for (i, individual) in individuals.iter().take(self.elitism).enumerate() {
            let genes = individual.strategy().genes().clone();
            new_strategies.push(S::from_genes(format!("gen{generation}_elite{i}"), genes));
        }

        // Fill rest with offspring
        new_strategies.extend(
            (self.elitism..population_size)
                .map(|i| self.create_offspring(individuals, generation, i, rng)),
        );

        new_strategies
    }

    #[instrument(level = "debug", skip_all, fields(index = index))]
    fn create_offspring<S: EvolvableStrategy>(
        &self,
        individuals: &[Individual<S>],
        generation: u32,
        index: usize,
        rng: &mut fastrand::Rng,
    ) -> S {
        let (parent1, parent2) = debug_span!("selection").in_scope(|| {
            let p1 = self.selection.select(individuals, rng);
            let p2 = self.selection.select(individuals, rng);
            (p1, p2)
        });

        let child_genes = debug_span!("crossover").in_scope(|| {
            if rng.f64() < self.crossover_rate {
                parent1.strategy().genes().crossover(
                    parent2.strategy().genes(),
                    self.crossover,
                    rng,
                )
            } else {
                parent1.strategy().genes().clone()
            }
        });

        let child_genes = debug_span!("mutation").in_scope(|| {
            if rng.f64() < self.mutation_rate {
                child_genes.mutate(self.mutation, rng)
            } else {
                child_genes
            }
        });

        debug_span!("from_genes")
            .in_scope(|| S::from_genes(format!("gen{generation}_{index}"), child_genes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gene::Gene;
    use crate::strategy::{EvolvableStrategy, Strategy};
    use crate::test_utils::FakeEvolvableStrategy;

    fn make_strategies(rng: &mut fastrand::Rng, count: usize) -> Vec<FakeEvolvableStrategy> {
        (0..count)
            .map(|i| FakeEvolvableStrategy::random(format!("gen0_{i}"), rng))
            .collect()
    }

    #[test]
    fn evolve_returns_population_with_correct_size() {
        let evolver = Evolver::new().generations(0);
        let mut rng = fastrand::Rng::with_seed(42);
        let strategies = make_strategies(&mut rng, 4);

        let population = evolver.evolve(strategies, &mut rng);

        assert_eq!(population.individuals().len(), 4);
    }

    #[test]
    fn evolve_with_zero_generations_returns_initial_population() {
        let evolver = Evolver::new().generations(0);
        let mut rng = fastrand::Rng::with_seed(42);
        let strategies = make_strategies(&mut rng, 4);

        let population = evolver.evolve(strategies, &mut rng);

        assert_eq!(population.generation(), 0);
    }

    #[test]
    fn evolve_increments_generation() {
        let evolver = Evolver::new().generations(3);
        let mut rng = fastrand::Rng::with_seed(42);
        let strategies = make_strategies(&mut rng, 4);

        let population = evolver.evolve(strategies, &mut rng);

        assert_eq!(population.generation(), 3);
    }

    #[test]
    fn evolve_is_deterministic_with_same_seed() {
        let evolver1 = Evolver::new().generations(2);
        let evolver2 = Evolver::new().generations(2);

        let mut rng1 = fastrand::Rng::with_seed(42);
        let strategies1 = make_strategies(&mut rng1, 4);
        let pop1 = evolver1.evolve(strategies1, &mut rng1);

        let mut rng2 = fastrand::Rng::with_seed(42);
        let strategies2 = make_strategies(&mut rng2, 4);
        let pop2 = evolver2.evolve(strategies2, &mut rng2);

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
        let evolver = Evolver::new()
            .elitism(2)
            .generations(1)
            .crossover_rate(1.0)
            .mutation_rate(0.0);
        let mut rng = fastrand::Rng::with_seed(42);
        let strategies = make_strategies(&mut rng, 4);

        let population = evolver.evolve(strategies, &mut rng);

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
        let evolver = Evolver::new()
            .generations(20)
            .elitism(4)
            .crossover_rate(0.9)
            .mutation_rate(0.2);

        assert_eq!(evolver.generations, 20);
        assert_eq!(evolver.elitism, 4);
        assert!((evolver.crossover_rate - 0.9).abs() < f64::EPSILON);
        assert!((evolver.mutation_rate - 0.2).abs() < f64::EPSILON);
    }

    #[test]
    fn default_creates_evolver() {
        let evolver = Evolver::default();

        assert_eq!(evolver.generations, 10);
        assert_eq!(evolver.elitism, 2);
    }

    #[test]
    fn crossover_builder_sets_crossover() {
        let _evolver = Evolver::new().crossover(Crossover::default());
    }

    #[test]
    fn mutation_builder_sets_mutation() {
        let _evolver = Evolver::new().mutation(Mutation::default());
    }

    #[test]
    fn selection_builder_sets_selection() {
        let _evolver = Evolver::new().selection(Selection::default());
    }

    #[test]
    fn tournament_builder_sets_tournament() {
        let _evolver = Evolver::new().tournament(Tournament::default());
    }

    #[test]
    fn game_builder_sets_game() {
        let _evolver = Evolver::new().game(Game::default());
    }

    #[test]
    fn no_crossover_no_mutation_preserves_parent_genes() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategies = make_strategies(&mut rng, 4);

        let parent_genes: Vec<Vec<Gene>> = strategies.iter().map(|s| s.genes().clone()).collect();

        // Run one generation with no crossover, no mutation
        let gen1 = Evolver::new()
            .elitism(0)
            .generations(1)
            .crossover_rate(0.0)
            .mutation_rate(0.0)
            .evolve(strategies, &mut rng);

        // Every offspring should have genes identical to some parent
        for individual in gen1.individuals() {
            let genes = individual.strategy().genes();
            assert!(
                parent_genes.iter().any(|p| p == genes),
                "offspring genes {:?} should match a parent",
                genes
            );
        }
    }

    #[test]
    fn mutation_rate_one_changes_genes() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategies = make_strategies(&mut rng, 4);

        let parent_genes: Vec<Vec<Gene>> = strategies.iter().map(|s| s.genes().clone()).collect();

        // Run one generation with no crossover but 100% mutation
        let gen1 = Evolver::new()
            .elitism(0)
            .generations(1)
            .crossover_rate(0.0)
            .mutation_rate(1.0)
            .evolve(strategies, &mut rng);

        // At least one offspring should have different genes than all parents
        let any_mutated = gen1.individuals().iter().any(|individual| {
            let genes = individual.strategy().genes();
            !parent_genes.iter().any(|p| p == genes)
        });

        assert!(any_mutated, "some offspring should have mutated genes");
    }

    #[test]
    fn crossover_rate_one_produces_valid_permutations() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategies = make_strategies(&mut rng, 4);

        // Run one generation with 100% crossover but no mutation
        let gen1 = Evolver::new()
            .elitism(0)
            .generations(1)
            .crossover_rate(1.0)
            .mutation_rate(0.0)
            .evolve(strategies, &mut rng);

        // With crossover, offspring genes should still be valid permutations
        for individual in gen1.individuals() {
            let genes = individual.strategy().genes();
            let mut sorted: Vec<Gene> = genes.clone();
            sorted.sort();
            let expected: Vec<Gene> = (0..5).map(Gene::new).collect();
            assert_eq!(sorted, expected, "genes should be valid permutation");
        }
    }
}
