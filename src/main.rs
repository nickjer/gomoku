use anyhow::Result;
use clap::{Parser, Subcommand};

use gomoku::cli::{EvolveArgs, run_evolve};

#[derive(Parser)]
#[command(name = "gomoku")]
#[command(about = "Evolve strategies for playing Gomoku")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Evolve strategies using a genetic algorithm
    Evolve(EvolveArgs),
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Evolve(ref args) => run_evolve(args),
    }
}
