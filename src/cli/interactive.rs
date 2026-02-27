use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, ValueEnum};

use super::{create_rng, load_strategy, setup_logging};
use crate::game::{Freestyle, Game, Play as GamePlay, RandomOpening};
use crate::interactive_strategy::InteractiveStrategy;
use crate::outcome::Outcome;

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

    let game: Game = if args.opening_moves > 0 {
        RandomOpening {
            moves: args.opening_moves,
        }
        .into()
    } else {
        Freestyle.into()
    };

    let terminal = ratatui::init();
    let human = InteractiveStrategy::new(terminal);

    let result = match args.play_as {
        PlayerColor::Black => game.play(&human, opponent.as_ref(), &mut rng),
        PlayerColor::White => game.play(opponent.as_ref(), &human, &mut rng),
    };

    ratatui::restore();

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
