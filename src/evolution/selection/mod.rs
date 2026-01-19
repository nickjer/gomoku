mod tournament;

use enum_dispatch::enum_dispatch;

pub use tournament::{Tournament, TournamentMode};

/// Trait for items that can be selected based on fitness.
pub trait HasFitness {
    fn fitness(&self) -> u32;
}

/// Trait for selection operations.
#[enum_dispatch]
pub trait RunSelection {
    fn select<'a, T: HasFitness>(&self, population: &'a [T], rng: &mut fastrand::Rng) -> &'a T;
}

/// Enum for polymorphic selection dispatch.
#[enum_dispatch(RunSelection)]
#[derive(Debug, Clone, Copy)]
pub enum Selection {
    Tournament,
}

impl Default for Selection {
    fn default() -> Self {
        Tournament::default().into()
    }
}
