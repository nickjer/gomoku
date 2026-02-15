mod evolvable_strategies;
mod evolve;
mod interactive;
mod play;

pub use evolve::{EvolveCommand, run_evolve};
pub use interactive::{InteractiveArgs, run_interactive};
pub use play::{PlayArgs, run_play};

use anyhow::{Context, Result};
use clap::ValueEnum;
use tracing::info;

/// Sets up logging with the specified log level.
///
/// # Errors
///
/// Returns an error if the log level is invalid.
pub fn setup_logging(log_level: Option<&str>) -> Result<()> {
    use tracing_subscriber::EnvFilter;
    use tracing_subscriber::fmt::format::FmtSpan;

    let filter = match log_level {
        Some(level) => {
            EnvFilter::try_new(level).with_context(|| format!("Invalid log level: {level}"))?
        }
        None => EnvFilter::from_default_env(),
    };

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_span_events(FmtSpan::CLOSE)
        .init();

    Ok(())
}

/// Creates an RNG from an optional seed.
#[must_use]
pub fn create_rng(seed: Option<u64>) -> fastrand::Rng {
    if let Some(seed) = seed {
        info!(seed, "Using provided RNG seed");
        fastrand::Rng::with_seed(seed)
    } else {
        info!("Using random RNG seed");
        fastrand::Rng::new()
    }
}

use crate::evolution::selection::{Selection, Tournament, TournamentMode};
use crate::game::{Freestyle, Game};
use crate::tournament::{Swiss, Tournament as CompetitionTournament};

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
