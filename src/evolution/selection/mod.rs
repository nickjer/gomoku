mod tournament;

pub use tournament::{Tournament, TournamentMode};

/// Trait for items that can be selected based on fitness.
pub trait HasFitness {
    fn fitness(&self) -> u32;
}

/// Enum for polymorphic selection dispatch.
#[derive(Debug, Clone, Copy)]
pub enum Selection {
    Tournament(Tournament),
}

impl Selection {
    pub fn select<'a, T: HasFitness>(&self, population: &'a [T], rng: &mut fastrand::Rng) -> &'a T {
        match self {
            Selection::Tournament(t) => t.select(population, rng),
        }
    }
}
