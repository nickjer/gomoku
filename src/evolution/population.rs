use super::Individual;

/// A population of individuals at a specific generation.
#[derive(Debug, Clone)]
pub struct Population<S> {
    individuals: Vec<Individual<S>>,
    generation: u32,
}

impl<S> Population<S> {
    #[must_use]
    pub fn new(individuals: Vec<Individual<S>>) -> Self {
        Self {
            individuals,
            generation: 0,
        }
    }

    #[must_use]
    pub fn generation(&self) -> u32 {
        self.generation
    }

    #[must_use]
    pub fn individuals(&self) -> &[Individual<S>] {
        &self.individuals
    }

    #[must_use]
    pub fn next_generation(self, new_individuals: Vec<Individual<S>>) -> Self {
        Self {
            individuals: new_individuals,
            generation: self.generation + 1,
        }
    }

    #[must_use]
    pub fn into_strategies(self) -> Vec<S> {
        self.individuals
            .into_iter()
            .map(Individual::into_strategy)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolution::fitness_score::FitnessScore;
    use crate::evolution::selection::HasFitness;

    #[derive(Debug, Clone)]
    struct MockStrategy;

    fn population(fitnesses: &[f32]) -> Population<MockStrategy> {
        let individuals = fitnesses
            .iter()
            .map(|&fit| Individual::new(MockStrategy, FitnessScore::new(fit)))
            .collect();
        Population::new(individuals)
    }

    #[test]
    fn new_creates_population_at_generation_zero() {
        let pop = population(&[10.0, 20.0, 30.0]);

        assert_eq!(pop.generation(), 0);
    }

    #[test]
    fn individuals_returns_slice() {
        let pop = population(&[10.0, 20.0, 30.0]);

        assert_eq!(pop.individuals().len(), 3);
        assert_eq!(pop.individuals()[0].fitness(), FitnessScore::new(10.0));
    }

    #[test]
    fn next_generation_increments_generation() {
        let pop = population(&[10.0, 20.0]);
        let new_individuals = vec![Individual::new(MockStrategy, FitnessScore::new(30.0))];

        let next = pop.next_generation(new_individuals);

        assert_eq!(next.generation(), 1);
        assert_eq!(next.individuals().len(), 1);
    }

    #[test]
    fn into_strategies_extracts_strategies() {
        let pop = population(&[10.0, 20.0, 30.0]);

        let strategies = pop.into_strategies();

        assert_eq!(strategies.len(), 3);
    }
}
