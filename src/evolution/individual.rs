use super::selection::HasFitness;

/// An individual in the evolutionary population, wrapping a strategy with its fitness.
#[derive(Debug, Clone)]
pub struct Individual<S> {
    strategy: S,
    fitness: u32,
}

impl<S> Individual<S> {
    #[must_use]
    pub fn new(strategy: S, fitness: u32) -> Self {
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

impl<S> HasFitness for Individual<S> {
    fn fitness(&self) -> u32 {
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
        let individual = Individual::new(strategy.clone(), 42);

        assert_eq!(individual.strategy(), &strategy);
        assert_eq!(individual.fitness(), 42);
    }

    #[test]
    fn into_strategy_consumes_and_returns_strategy() {
        let strategy = MockStrategy {
            genes: vec![1, 2, 3],
        };
        let individual = Individual::new(strategy.clone(), 42);

        assert_eq!(individual.into_strategy(), strategy);
    }

    #[test]
    fn implements_has_fitness() {
        let individual = Individual::new(MockStrategy { genes: vec![] }, 100);

        let fitness: u32 = HasFitness::fitness(&individual);

        assert_eq!(fitness, 100);
    }
}
