mod evolvable_strategies;
mod evolve;
mod interactive;
mod play;

pub use evolvable_strategies::EvolvableStrategies;
pub use evolve::{EvolveArgs, run_evolve};
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

use crate::evolution::crossover::Crossover;
use crate::evolution::mutation::Mutation;
use crate::evolution::selection::{Selection, Tournament, TournamentMode};
use crate::game::{Freestyle, Game};
use crate::tournament::{Swiss, Tournament as CompetitionTournament};

/// CLI-exposed strategy types for evolution.
#[derive(Debug, Clone, Copy, ValueEnum)]
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

/// CLI crossover specification.
///
/// Format: `variant`
///
/// Examples:
/// - `order`
/// - `pmx`
/// - `uniform`
#[derive(Debug, Clone, Default)]
pub struct CliCrossover(Crossover);

impl CliCrossover {
    #[must_use]
    pub fn into_crossover(self) -> Crossover {
        self.0
    }
}

impl std::str::FromStr for CliCrossover {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let crossover = match s {
            "order" => Crossover::Order,
            "pmx" => Crossover::Pmx,
            "uniform" => Crossover::Uniform,
            _ => return Err(format!("unknown crossover variant: '{s}'")),
        };
        Ok(Self(crossover))
    }
}

impl std::fmt::Display for CliCrossover {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            Crossover::Order => write!(f, "order"),
            Crossover::Pmx => write!(f, "pmx"),
            Crossover::Uniform => write!(f, "uniform"),
        }
    }
}

/// Default sigma for Gaussian mutation.
const DEFAULT_GAUSSIAN_SIGMA: f32 = 0.01;

/// CLI mutation specification with optional parameters.
///
/// Format: `variant` or `variant:param=value:param2=value2`
///
/// Examples:
/// - `swap`
/// - `gaussian` (uses default sigma=0.01)
/// - `gaussian:sigma=0.05`
#[derive(Debug, Clone, Default)]
pub struct CliMutation(Mutation);

impl CliMutation {
    #[must_use]
    pub fn into_mutation(self) -> Mutation {
        self.0
    }
}

impl std::str::FromStr for CliMutation {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut parts = s.split(':');
        let variant = parts.next().unwrap_or("");

        let mut params = std::collections::HashMap::new();
        for part in parts {
            if let Some((key, value)) = part.split_once('=') {
                params.insert(key, value);
            } else {
                return Err(format!(
                    "invalid parameter format: '{part}', expected key=value"
                ));
            }
        }

        let mutation = match variant {
            "swap" => Mutation::Swap,
            "insert" => Mutation::Insert,
            "inversion" => Mutation::Inversion,
            "gaussian" => {
                let sigma = params
                    .remove("sigma")
                    .map(str::parse::<f32>)
                    .transpose()
                    .map_err(|err| format!("invalid sigma value: {err}"))?
                    .unwrap_or(DEFAULT_GAUSSIAN_SIGMA);
                Mutation::Gaussian { sigma }
            }
            _ => return Err(format!("unknown mutation variant: '{variant}'")),
        };

        if let Some(unknown) = params.keys().next() {
            return Err(format!(
                "unknown parameter '{unknown}' for mutation '{variant}'"
            ));
        }

        Ok(Self(mutation))
    }
}

impl std::fmt::Display for CliMutation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            Mutation::Swap => write!(f, "swap"),
            Mutation::Insert => write!(f, "insert"),
            Mutation::Inversion => write!(f, "inversion"),
            Mutation::Gaussian { sigma } => write!(f, "gaussian:sigma={sigma}"),
        }
    }
}
