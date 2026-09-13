use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, ValueEnum};

use super::{create_rng, load_strategy, print_game_result, setup_logging};
use crate::board::Board;
use crate::game::{Freestyle, Game, NoOpObserver, Play as GamePlay};
use crate::interactive_strategy::InteractiveStrategy;
use crate::strategy::Strategy;

/// Which color the human player uses.
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub enum PlayerColor {
    #[default]
    Black,
    White,
}

/// Arguments for the interactive subcommand.
#[derive(Debug, Args)]
pub struct InteractiveArgs {
    /// Opponent strategy: path to .bin file, or "minimax" / "minimax:DEPTH".
    pub strategy: PathBuf,

    /// Which color to play as (black moves first).
    #[arg(long, default_value = "black")]
    pub play_as: PlayerColor,

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

/// Runs the interactive subcommand.
///
/// # Errors
///
/// Returns an error if logging setup fails or strategy loading fails.
pub fn run_interactive(args: &InteractiveArgs) -> Result<()> {
    setup_logging(args.log_level.as_deref())?;

    let mut rng = create_rng(args.seed);
    let opponent = load_strategy(&args.strategy)?;

    let game: Game = Freestyle {
        opening_moves: args.opening_moves,
    }
    .into();

    let terminal = ratatui::init();
    let human = InteractiveStrategy::new(terminal);

    let (black, white): (&dyn Strategy, &dyn Strategy) = match args.play_as {
        PlayerColor::Black => (&human, opponent.as_ref()),
        PlayerColor::White => (opponent.as_ref(), &human),
    };

    let mut board = Board::new();
    let outcome = game
        .play_from(&mut board, black, white, &mut NoOpObserver, &mut rng)
        .expect("NoOpObserver never breaks early");

    ratatui::restore();

    print_game_result(&board, outcome, black.label(), white.label());

    Ok(())
}
