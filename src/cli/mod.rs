mod evolvable_strategies;
mod evolve;

pub use evolve::{EvolveArgs, run_evolve};
pub use evolvable_strategies::EvolvableStrategies;

use clap::ValueEnum;
use serde::{Deserialize, Serialize};

use crate::evolution::crossover::{Crossover, Order, Pmx};
use crate::evolution::mutation::{Insert, Inversion, Mutation, Swap};
use crate::evolution::selection::{Selection, Tournament, TournamentMode};
use crate::game::{Freestyle, Game};
use crate::tournament::{Swiss, Tournament as CompetitionTournament};

/// CLI-exposed strategy types for evolution.
#[derive(Debug, Clone, Copy, ValueEnum, Serialize, Deserialize)]
pub enum CliStrategy {
    Nn1,
    Nn2,
    Nn3,
    Nn4,
}

/// CLI-exposed game variants (excludes test-only variants).
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum CliGame {
    Freestyle,
}

impl From<CliGame> for Game {
    fn from(cli: CliGame) -> Self {
        match cli {
            CliGame::Freestyle => Freestyle.into(),
        }
    }
}

/// CLI-exposed tournament variants (excludes test-only variants).
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum CliTournament {
    Swiss,
}

impl From<CliTournament> for CompetitionTournament {
    fn from(cli: CliTournament) -> Self {
        match cli {
            CliTournament::Swiss => Swiss.into(),
        }
    }
}

/// CLI-exposed selection mode for tournament selection.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum CliSelectionMode {
    WithReplacement,
    WithoutReplacement,
}

impl From<CliSelectionMode> for Selection {
    fn from(cli: CliSelectionMode) -> Self {
        let mode = match cli {
            CliSelectionMode::WithReplacement => TournamentMode::WithReplacement,
            CliSelectionMode::WithoutReplacement => TournamentMode::WithoutReplacement,
        };
        Tournament::new(3, mode).into()
    }
}

/// CLI-exposed crossover variants (excludes test-only variants).
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum CliCrossover {
    Order,
    Pmx,
}

impl From<CliCrossover> for Crossover {
    fn from(cli: CliCrossover) -> Self {
        match cli {
            CliCrossover::Order => Order.into(),
            CliCrossover::Pmx => Pmx.into(),
        }
    }
}

/// CLI-exposed mutation variants (excludes test-only variants).
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum CliMutation {
    Swap,
    Insert,
    Inversion,
}

impl From<CliMutation> for Mutation {
    fn from(cli: CliMutation) -> Self {
        match cli {
            CliMutation::Swap => Swap.into(),
            CliMutation::Insert => Insert.into(),
            CliMutation::Inversion => Inversion.into(),
        }
    }
}
