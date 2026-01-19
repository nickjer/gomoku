mod standing;
mod swiss;

pub use standing::Standing;
pub use swiss::{Swiss, total_rounds};

use enum_dispatch::enum_dispatch;

use crate::game::Game;
use crate::strategy::Strategy;

/// Trait for running tournaments.
#[enum_dispatch]
pub trait RunTournament {
    fn run<S: Strategy>(
        &self,
        strategies: &[S],
        game: &Game,
        rng: &mut fastrand::Rng,
    ) -> Vec<Standing>;
}

/// Test tournament: returns standings in input order (first strategy ranks first).
#[cfg(test)]
#[derive(Debug, Clone, Copy, Default)]
pub struct InputOrder;

#[cfg(test)]
impl RunTournament for InputOrder {
    fn run<S: Strategy>(
        &self,
        strategies: &[S],
        _game: &Game,
        _rng: &mut fastrand::Rng,
    ) -> Vec<Standing> {
        (0..strategies.len())
            .map(|i| Standing::new(i, 0, 0, 0, 0))
            .collect()
    }
}

/// Test tournament: returns standings in the specified label order.
#[cfg(test)]
#[derive(Debug, Clone)]
pub struct Scripted {
    ranking: Vec<&'static str>,
}

#[cfg(test)]
impl Scripted {
    pub fn new(ranking: Vec<&'static str>) -> Self {
        Self { ranking }
    }
}

#[cfg(test)]
impl RunTournament for Scripted {
    fn run<S: Strategy>(
        &self,
        strategies: &[S],
        _game: &Game,
        _rng: &mut fastrand::Rng,
    ) -> Vec<Standing> {
        self.ranking
            .iter()
            .map(|&label| {
                let index = strategies
                    .iter()
                    .position(|s| s.label() == label)
                    .unwrap_or_else(|| panic!("Strategy with label '{label}' not found"));
                Standing::new(index, 0, 0, 0, 0)
            })
            .collect()
    }
}

/// Enum for polymorphic tournament dispatch.
#[enum_dispatch(RunTournament)]
#[derive(Debug, Clone, strum::Display)]
pub enum Tournament {
    Swiss,
    #[cfg(test)]
    InputOrder,
    #[cfg(test)]
    Scripted,
}

impl Default for Tournament {
    fn default() -> Self {
        #[cfg(not(test))]
        {
            Swiss.into()
        }
        #[cfg(test)]
        {
            InputOrder.into()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Scripted;
    use crate::test_utils::StubStrategy;

    #[test]
    fn swiss_variant_runs_tournament() {
        let game: Game = Scripted::new().add("a", "b", Some("a")).into();
        let tournament: Tournament = Swiss::new().into();
        let strategies = vec![StubStrategy::new("a"), StubStrategy::new("b")];
        let mut rng = fastrand::Rng::with_seed(42);

        let standings = tournament.run(&strategies, &game, &mut rng);

        assert_eq!(standings.len(), 2);
    }
}
