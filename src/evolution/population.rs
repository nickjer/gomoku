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
    use crate::evolution::selection::HasFitness;

    #[derive(Debug, Clone)]
    struct MockStrategy;

    fn population(fitnesses: &[u32]) -> Population<MockStrategy> {
        let individuals = fitnesses
            .iter()
            .map(|&f| Individual::new(MockStrategy, f))
            .collect();
        Population::new(individuals)
    }

    #[test]
    fn new_creates_population_at_generation_zero() {
        let pop = population(&[10, 20, 30]);

        assert_eq!(pop.generation(), 0);
    }

    #[test]
    fn individuals_returns_slice() {
        let pop = population(&[10, 20, 30]);

        assert_eq!(pop.individuals().len(), 3);
        assert_eq!(pop.individuals()[0].fitness(), 10);
    }

    #[test]
    fn next_generation_increments_generation() {
        let pop = population(&[10, 20]);
        let new_individuals = vec![Individual::new(MockStrategy, 30)];

        let next = pop.next_generation(new_individuals);

        assert_eq!(next.generation(), 1);
        assert_eq!(next.individuals().len(), 1);
    }
}
