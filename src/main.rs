use anyhow::Result;
use clap::{Parser, Subcommand};

use gomoku::cli::{
    EvolveCommand, InteractiveArgs, PlayArgs, run_evolve, run_interactive, run_play,
};

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
    #[command(subcommand)]
    Evolve(EvolveCommand),
    /// Play interactively against a strategy
    Interactive(InteractiveArgs),
    /// Play a game between two strategies
    Play(PlayArgs),
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Evolve(ref args) => run_evolve(args),
        Commands::Interactive(ref args) => run_interactive(args),
        Commands::Play(ref args) => run_play(args),
    }
}
