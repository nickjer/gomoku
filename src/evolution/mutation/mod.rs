mod gaussian;

pub use gaussian::gaussian_mutate;

/// Mutation method for genetic algorithms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mutation {
    /// Gaussian mutation: adds N(0, sigma) noise to all weights (continuous genes).
    Gaussian { sigma: f32 },
}

impl Default for Mutation {
    fn default() -> Self {
        Self::Gaussian { sigma: 0.01 }
    }
}
