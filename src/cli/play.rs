use std::path::PathBuf;

use anyhow::Result;
use clap::Args;

use super::{create_rng, load_strategy, print_game_result, setup_logging};
use crate::board::Board;
use crate::game::{Freestyle, Game, NoOpObserver, Play as GamePlay};

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

    let game: Game = Freestyle {
        opening_moves: args.opening_moves,
    }
    .into();

    let mut board = Board::new();
    let outcome = game
        .play_from(
            &mut board,
            black.as_ref(),
            white.as_ref(),
            &mut NoOpObserver,
            &mut rng,
        )
        .expect("NoOpObserver never breaks early");

    print_game_result(&board, outcome, black.label(), white.label());

    Ok(())
}
