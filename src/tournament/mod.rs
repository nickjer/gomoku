mod standing;
mod swiss;

pub use standing::Standing;
pub use swiss::{Swiss, total_rounds};

use crate::match_runner::RunMatch;
use crate::strategy::Strategy;

/// Trait for running tournaments.
pub trait RunTournament {
    fn run<S: Strategy>(&self, strategies: &[S], rng: &mut fastrand::Rng) -> Vec<Standing>;
}

/// Enum for polymorphic tournament dispatch.
#[derive(Debug, Clone)]
pub enum Tournament<R> {
    Swiss(Swiss<R>),
}

impl<R: RunMatch> RunTournament for Tournament<R> {
    fn run<S: Strategy>(&self, strategies: &[S], rng: &mut fastrand::Rng) -> Vec<Standing> {
        match self {
            Tournament::Swiss(t) => t.run(strategies, rng),
        }
    }
}
