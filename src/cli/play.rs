use std::path::PathBuf;

use anyhow::Result;
use clap::Args;

use super::{create_rng, load_strategy, setup_logging};
use crate::game::{Freestyle, Game, Play as GamePlay, RandomOpening};
use crate::outcome::Outcome;

/// Arguments for the play subcommand.
#[derive(Debug, Args)]
pub struct PlayArgs {
    /// Black strategy: path to .bin file, or "minimax" / "minimax:DEPTH".
    pub black: PathBuf,

    /// White strategy: path to .bin file, or "minimax" / "minimax:DEPTH".
    pub white: PathBuf,

    /// Pre-place N random stones before the game starts (0 = empty board).
    #[arg(long, default_value = "0")]
    pub opening_moves: u32,

    /// RNG seed for reproducibility.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Log level (error, warn, info, debug, trace).
    #[arg(short, long, value_name = "LEVEL")]
    pub log_level: Option<String>,
}

/// Runs the play subcommand.
///
/// # Errors
///
/// Returns an error if logging setup fails or strategy loading fails.
pub fn run_play(args: &PlayArgs) -> Result<()> {
    setup_logging(args.log_level.as_deref())?;

    let mut rng = create_rng(args.seed);

    let black = load_strategy(&args.black)?;
    let white = load_strategy(&args.white)?;

    let game: Game = if args.opening_moves > 0 {
        RandomOpening {
            moves: args.opening_moves,
        }
        .into()
    } else {
        Freestyle.into()
    };

    let result = game.play(black.as_ref(), white.as_ref(), &mut rng);

    println!("Black (X): {}", result.black_label());
    println!("White (O): {}", result.white_label());
    println!();
    println!("{}", result.board_state());
    println!();

    match result.outcome() {
        Outcome::BlackWins => println!("Result: Black (X) wins in {} turns", result.turn_count()),
        Outcome::WhiteWins => println!("Result: White (O) wins in {} turns", result.turn_count()),
        Outcome::Draw => println!("Result: Draw after {} turns", result.turn_count()),
    }

    Ok(())
}
