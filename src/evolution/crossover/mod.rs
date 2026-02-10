mod uniform;

pub use uniform::uniform_crossover;

/// Crossover method for genetic algorithms.
#[derive(Debug, Default, Clone, Copy)]
pub enum Crossover {
    /// Uniform crossover for continuous genes.
    #[default]
    Uniform,
}
