use tracing::{debug_span, info, info_span, instrument};

use crate::strategy::{EvolvableStrategy, Strategy};

use super::crossover::Crossover;
use super::fitness_evaluator::{EvaluateFitness, FitnessEvaluator, TournamentFitness};
use super::fitness_score::FitnessScore;
use super::fitness_weight::FitnessWeight;
use super::genes::EvolvableGenes;
use super::mutation::Mutation;
use super::selection::{HasFitness, RunSelection, Selection};
use super::{Individual, Population};

/// Configuration and execution of the evolutionary algorithm.
pub struct Evolver {
    generations: u32,
    elitism: usize,
    crossover: Crossover,
    crossover_rate: f64,
    mutation: Mutation,
    mutation_rate: f64,
    selection: Selection,
    evaluators: Vec<(FitnessEvaluator, FitnessWeight)>,
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
            evaluators: vec![(TournamentFitness::default().into(), FitnessWeight::new(1.0))],
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
    pub fn evaluators(mut self, evaluators: Vec<(FitnessEvaluator, FitnessWeight)>) -> Self {
        self.evaluators = evaluators;
        self
    }

    /// Runs the evolutionary algorithm and returns the final population.
    pub fn evolve<S: EvolvableStrategy>(
        &self,
        strategies: Vec<S>,
        rng: &mut fastrand::Rng,
    ) -> Population<S> {
        let individuals = self.evaluate_and_sort(strategies, rng);
        let mut population = Population::new(individuals);

        for generation in 1..=self.generations {
            let span = info_span!("generation", number = generation);
            let _guard = span.enter();
            let new_strategies = self.create_next_generation(&population, rng);
            let new_individuals = self.evaluate_and_sort(new_strategies, rng);
            let champion = new_individuals[0].strategy().label();
            let champion_is_elite = champion.contains("elite");
            info!(champion, champion_is_elite, "Generation complete");
            population = population.next_generation(new_individuals);
        }

        population
    }

    fn compute_fitness<S: Strategy>(
        &self,
        strategies: &[S],
        rng: &mut fastrand::Rng,
    ) -> Vec<FitnessScore> {
        let mut totals = vec![0.0_f32; strategies.len()];
        let mut per_evaluator: Vec<(&FitnessEvaluator, Vec<FitnessScore>)> = Vec::new();
        for (evaluator, weight) in &self.evaluators {
            let scores = evaluator.evaluate(strategies, rng);
            for (total, score) in totals.iter_mut().zip(&scores) {
                *total += score.value() * weight.value();
            }
            per_evaluator.push((evaluator, scores));
        }

        let mut indices: Vec<usize> = (0..strategies.len()).collect();
        indices.sort_by(|&a, &b| totals[b].total_cmp(&totals[a]));

        for (rank, &idx) in indices.iter().take(5).enumerate() {
            log_fitness_breakdown(rank + 1, idx, &totals, strategies, &per_evaluator);
        }
        if indices.len() > 10 {
            for (rank, &idx) in indices.iter().enumerate().skip(indices.len() - 5) {
                log_fitness_breakdown(rank + 1, idx, &totals, strategies, &per_evaluator);
            }
        }

        totals.into_iter().map(FitnessScore::new).collect()
    }

    fn evaluate_and_sort<S: Strategy>(
        &self,
        strategies: Vec<S>,
        rng: &mut fastrand::Rng,
    ) -> Vec<Individual<S>> {
        let scores = self.compute_fitness(&strategies, rng);
        let mut individuals: Vec<_> = strategies
            .into_iter()
            .zip(scores)
            .map(|(strategy, fitness)| Individual::new(strategy, fitness))
            .collect();
        individuals.sort_by_key(|ind| std::cmp::Reverse(ind.fitness()));
        individuals
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
            let first = self.selection.select(individuals, rng);
            let second = self.selection.select(individuals, rng);
            (first, second)
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

fn log_fitness_breakdown<S: Strategy>(
    rank: usize,
    idx: usize,
    totals: &[f32],
    strategies: &[S],
    per_evaluator: &[(&FitnessEvaluator, Vec<FitnessScore>)],
) {
    let breakdown: String = per_evaluator
        .iter()
        .map(|(evaluator, scores)| format!("{}={:.1}", evaluator, scores[idx].value()))
        .collect::<Vec<_>>()
        .join(", ");
    info!(
        rank,
        label = strategies[idx].label(),
        total = totals[idx],
        breakdown,
        "Fitness breakdown"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
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
            .map(|ind| ind.strategy().label())
            .collect();
        let labels2: Vec<_> = pop2
            .individuals()
            .iter()
            .map(|ind| ind.strategy().label())
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
            .map(|ind| ind.strategy().label())
            .collect();
        assert!(labels.iter().any(|label| label.contains("elite")));
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
    fn no_crossover_no_mutation_preserves_parent_genes() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategies = make_strategies(&mut rng, 4);

        let parent_genes: Vec<Vec<f32>> = strategies
            .iter()
            .map(|strat| strat.genes().as_ref().to_vec())
            .collect();

        // Run one generation with no crossover, no mutation
        let gen1 = Evolver::new()
            .elitism(0)
            .generations(1)
            .crossover_rate(0.0)
            .mutation_rate(0.0)
            .evolve(strategies, &mut rng);

        // Every offspring should have genes identical to some parent
        for individual in gen1.individuals() {
            let genes: Vec<f32> = individual.strategy().genes().as_ref().to_vec();
            assert!(
                parent_genes.iter().any(|parent| *parent == genes),
                "offspring genes should match a parent",
            );
        }
    }

    #[test]
    fn mutation_rate_one_changes_genes() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategies = make_strategies(&mut rng, 4);

        let parent_genes: Vec<Vec<f32>> = strategies
            .iter()
            .map(|strat| strat.genes().as_ref().to_vec())
            .collect();

        // Run one generation with no crossover but 100% mutation
        let gen1 = Evolver::new()
            .elitism(0)
            .generations(1)
            .crossover_rate(0.0)
            .mutation_rate(1.0)
            .evolve(strategies, &mut rng);

        // At least one offspring should have different genes than all parents
        let any_mutated = gen1.individuals().iter().any(|individual| {
            let genes: Vec<f32> = individual.strategy().genes().as_ref().to_vec();
            !parent_genes.iter().any(|parent| *parent == genes)
        });

        assert!(any_mutated, "some offspring should have mutated genes");
    }
}
