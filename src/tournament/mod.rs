mod standing;
mod swiss;

pub use standing::Standing;
pub use swiss::{Swiss, total_rounds};

use crate::match_runner::RunMatch;
use crate::strategy::Strategy;

/// Trait for running tournaments.
pub trait RunTournament {
    fn run<S: Strategy, R: RunMatch>(
        &self,
        strategies: &[S],
        match_runner: &R,
        rng: &mut fastrand::Rng,
    ) -> Vec<Standing>;
}

/// Enum for polymorphic tournament dispatch.
#[derive(Debug, Clone, Copy, Default)]
pub enum Tournament {
    #[default]
    Swiss,
}

impl RunTournament for Tournament {
    fn run<S: Strategy, R: RunMatch>(
        &self,
        strategies: &[S],
        match_runner: &R,
        rng: &mut fastrand::Rng,
    ) -> Vec<Standing> {
        match self {
            Tournament::Swiss => Swiss::new().run(strategies, match_runner, rng),
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
        let tournament = Tournament::Swiss;
        let strategies = vec![StubStrategy::new("a"), StubStrategy::new("b")];
        let mut rng = fastrand::Rng::with_seed(42);

        let standings = tournament.run(&strategies, &match_runner, &mut rng);

        assert_eq!(standings.len(), 2);
    }
}
