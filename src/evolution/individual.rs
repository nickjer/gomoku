use std::cmp::Ordering;

use super::fitness_score::FitnessScore;
use super::selection::HasFitness;

/// An individual in the evolutionary population, wrapping a strategy with its fitness.
#[derive(Debug, Clone)]
pub struct Individual<S> {
    strategy: S,
    fitness: FitnessScore,
}

impl<S> Individual<S> {
    #[must_use]
    pub fn new(strategy: S, fitness: FitnessScore) -> Self {
        Self { strategy, fitness }
    }

    #[must_use]
    pub fn strategy(&self) -> &S {
        &self.strategy
    }

    #[must_use]
    pub fn into_strategy(self) -> S {
        self.strategy
    }
}

impl<S> PartialEq for Individual<S> {
    fn eq(&self, other: &Self) -> bool {
        self.fitness == other.fitness
    }
}

impl<S> PartialOrd for Individual<S> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.fitness.cmp(&other.fitness))
    }
}

impl<S> HasFitness for Individual<S> {
    fn fitness(&self) -> FitnessScore {
        self.fitness
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq)]
    struct MockStrategy {
        genes: Vec<u32>,
    }

    #[test]
    fn new_creates_individual_with_strategy_and_fitness() {
        let strategy = MockStrategy {
            genes: vec![1, 2, 3],
        };
        let individual = Individual::new(strategy.clone(), FitnessScore::new(42.0));

        assert_eq!(individual.strategy(), &strategy);
        assert_eq!(individual.fitness(), FitnessScore::new(42.0));
    }

    #[test]
    fn into_strategy_consumes_and_returns_strategy() {
        let strategy = MockStrategy {
            genes: vec![1, 2, 3],
        };
        let individual = Individual::new(strategy.clone(), FitnessScore::new(42.0));

        assert_eq!(individual.into_strategy(), strategy);
    }

    #[test]
    fn implements_has_fitness() {
        let individual = Individual::new(MockStrategy { genes: vec![] }, FitnessScore::new(100.0));

        let fitness: FitnessScore = HasFitness::fitness(&individual);

        assert_eq!(fitness, FitnessScore::new(100.0));
    }
}
