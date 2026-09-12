use std::path::PathBuf;

use anyhow::Result;
use clap::Args;

use super::saved_strategy::read_strategy_file;

/// Arguments for the inspect subcommand.
#[derive(Debug, Args)]
pub struct InspectArgs {
    /// Path to strategy (.bin file).
    pub path: PathBuf,
}

/// Runs the inspect subcommand.
///
/// # Errors
///
/// Returns an error if strategy loading fails.
pub fn run_inspect(args: &InspectArgs) -> Result<()> {
    let strategy = read_strategy_file(&args.path)?;

    println!("{strategy}");

    Ok(())
}
