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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{ScriptedMatchRunner, StubStrategy, Winner};

    #[test]
    fn swiss_variant_runs_tournament() {
        let match_runner = ScriptedMatchRunner::new().add("a", "b", Winner::Label("a"));
        let tournament = Tournament::Swiss(Swiss::new(match_runner));
        let strategies = vec![StubStrategy::new("a"), StubStrategy::new("b")];
        let mut rng = fastrand::Rng::with_seed(42);

        let standings = tournament.run(&strategies, &mut rng);

        assert_eq!(standings.len(), 2);
    }
}
