mod evolvable_strategies;
mod evolve;
mod inspect;
mod interactive;
mod play;

pub use evolve::{EvolveCommand, run_evolve};
pub use inspect::{InspectArgs, run_inspect};
pub use interactive::{InteractiveArgs, run_interactive};
pub use play::{PlayArgs, run_play};

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::cluster::weights::ClusterWeights;
use crate::cluster::{ClusterSmall, ClusterTiny};
use crate::conv::weights::ConvWeights;
use crate::conv::{ConvSmall, ConvTiny};
use crate::strategy::{EvolvableStrategy, Strategy};

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

/// Binary-serializable strategy data (weights only, label comes from filename).
#[derive(Debug, Serialize, Deserialize)]
pub enum StrategyData {
    ConvTiny { weights: ConvWeights<3, 32, 2, 0> },
    ConvSmall { weights: ConvWeights<3, 64, 4, 0> },
    ClusterTiny { weights: ClusterWeights<9, 32, 2> },
    ClusterSmall { weights: ClusterWeights<9, 64, 4> },
}

impl std::fmt::Display for StrategyData {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConvTiny { weights } => write!(formatter, "{weights}"),
            Self::ConvSmall { weights } => write!(formatter, "{weights}"),
            Self::ClusterTiny { weights } => write!(formatter, "{weights}"),
            Self::ClusterSmall { weights } => write!(formatter, "{weights}"),
        }
    }
}

/// Loads raw strategy data from a binary file.
///
/// Returns the label (derived from filename) and the deserialized [`StrategyData`].
///
/// # Errors
///
/// Returns an error if the file cannot be read or deserialization fails.
pub fn load_strategy_data(path: &Path) -> Result<(String, StrategyData)> {
    let cwd = std::env::current_dir().unwrap_or_default();
    let label = path
        .strip_prefix(&cwd)
        .unwrap_or(path)
        .display()
        .to_string();

    let bytes = fs::read(path).with_context(|| format!("Failed to read: {}", path.display()))?;

    let data: StrategyData = postcard::from_bytes(&bytes)
        .with_context(|| format!("Failed to deserialize: {}", path.display()))?;

    Ok((label, data))
}

/// Loads a single strategy from a binary file as a trait object.
///
/// # Errors
///
/// Returns an error if the file cannot be read or deserialization fails.
pub fn load_strategy_from_file(path: &Path) -> Result<Box<dyn Strategy>> {
    let (label, data) = load_strategy_data(path)?;

    Ok(match data {
        StrategyData::ConvTiny { weights } => Box::new(ConvTiny::from_genes(label, weights)),
        StrategyData::ConvSmall { weights } => Box::new(ConvSmall::from_genes(label, weights)),
        StrategyData::ClusterTiny { weights } => Box::new(ClusterTiny::from_genes(label, weights)),
        StrategyData::ClusterSmall { weights } => {
            Box::new(ClusterSmall::from_genes(label, weights))
        }
    })
}
