#![warn(clippy::as_conversions)]

mod bitboard;
mod board;
mod cli;
mod evolution;
mod game;
mod interactive_strategy;
mod match_result;
mod minimax;
mod nn;
mod offset;
mod outcome;
mod position;
mod position_id;
mod position_map;
mod stone;
mod strategy;
mod threat;
mod tournament;

#[cfg(test)]
mod test_utils;

use anyhow::Result;
use clap::{Parser, Subcommand};

use crate::cli::{
    EvolveCommand, InspectArgs, InteractiveArgs, PlayArgs, run_evolve, run_inspect,
    run_interactive, run_play,
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
    /// Print strategy summary statistics
    Inspect(InspectArgs),
    /// Play interactively against a strategy
    Interactive(InteractiveArgs),
    /// Play a game between two strategies
    Play(PlayArgs),
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Evolve(ref args) => run_evolve(args),
        Commands::Inspect(ref args) => run_inspect(args),
        Commands::Interactive(ref args) => run_interactive(args),
        Commands::Play(ref args) => run_play(args),
    }
}
