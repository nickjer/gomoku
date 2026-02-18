use std::path::PathBuf;

use anyhow::Result;
use clap::Args;

use super::load_strategy_data;

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
    let (label, data) = load_strategy_data(&args.path)?;

    println!("Label: {label}");
    println!("{data}");

    Ok(())
}
