mod evolve;
mod inspect;
mod interactive;
mod play;
mod saved_strategy;

pub use evolve::{EvolveArgs, run_evolve};
pub use inspect::{InspectArgs, run_inspect};
pub use interactive::{InteractiveArgs, run_interactive};
pub use play::{PlayArgs, run_play};

use std::path::Path;

use anyhow::{Context, Result, bail};
use tracing::info;

use crate::board::Board;
use crate::minimax::{DEFAULT_DEPTH, MinimaxStrategy};
use crate::outcome::Outcome;
use crate::stone::Stone;
use crate::strategy::Strategy;
use saved_strategy::read_strategy_file;

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

/// Loads a strategy from a specifier: `"minimax"` / `"minimax:DEPTH"` or a file path.
///
/// # Errors
///
/// Returns an error if the specifier is an invalid minimax depth or file loading fails.
pub fn load_strategy(specifier: &Path) -> Result<Box<dyn Strategy>> {
    let specifier_str = specifier.to_str().unwrap_or("");
    if specifier_str == "minimax" {
        return Ok(Box::new(MinimaxStrategy::new(DEFAULT_DEPTH)));
    }
    if let Some(depth_str) = specifier_str.strip_prefix("minimax:") {
        let depth: u32 = depth_str.parse().context("Invalid minimax depth")?;
        if depth == 0 {
            bail!("Minimax depth must be at least 1");
        }
        return Ok(Box::new(MinimaxStrategy::new(depth)));
    }
    Ok(Box::new(read_strategy_file(specifier)?))
}

/// Prints who played which color, the final board, and how the game ended.
pub fn print_game_result(board: &Board, outcome: Outcome, black_label: &str, white_label: &str) {
    println!("{}: {black_label}", Stone::Black);
    println!("{}: {white_label}", Stone::White);
    println!();
    println!("{board}");
    println!();

    match outcome {
        Outcome::Win(winner) => println!("Result: {winner} wins in {} turns", board.move_count()),
        Outcome::Draw => println!("Result: Draw after {} turns", board.move_count()),
    }
}
