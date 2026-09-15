use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use clap::Args;

use super::saved_strategy::read_strategy_file;
use crate::board::Board;
use crate::position_id::PositionId;
use crate::stone::Stone;

/// Arguments for the inspect subcommand.
#[derive(Debug, Args)]
pub struct InspectArgs {
    /// Path to strategy (.bin file).
    pub path: PathBuf,

    /// Score the empty positions of the board drawn in this file (`X`, `O`,
    /// and `.` or `·` for empty, whitespace ignored) for the side to move.
    #[arg(long)]
    pub board: Option<PathBuf>,
}

/// Runs the inspect subcommand.
///
/// # Errors
///
/// Returns an error if strategy loading fails or the board file does not
/// draw a whole board.
pub fn run_inspect(args: &InspectArgs) -> Result<()> {
    let strategy = read_strategy_file(&args.path)?;

    println!("{strategy}");

    if let Some(board_path) = &args.board {
        let board = read_board_file(board_path)?;
        println!();
        print_scores(
            &strategy.score_positions(board.stone_to_move(), &board),
            &board,
        );
    }

    Ok(())
}

/// Reads a board drawn as `X`, `O`, and `.` or `·`, whitespace ignored.
fn read_board_file(path: &Path) -> Result<Board> {
    let text =
        fs::read_to_string(path).with_context(|| format!("Failed to read: {}", path.display()))?;
    let symbols: Vec<char> = text
        .chars()
        .filter(|symbol| !symbol.is_whitespace())
        .collect();
    ensure!(
        symbols.len() == PositionId::COUNT,
        "expected {} positions in {}, found {}",
        PositionId::COUNT,
        path.display(),
        symbols.len()
    );

    let mut board = Board::new();
    for (position, symbol) in PositionId::iter().zip(symbols) {
        match symbol {
            'X' => board.place_unchecked(position, Stone::Black),
            'O' => board.place_unchecked(position, Stone::White),
            '.' | '·' => {}
            other => bail!("unexpected stone {other:?} in {}", path.display()),
        }
    }
    Ok(board)
}

/// Prints the board with every empty position replaced by its rank (1 is
/// the network's favorite), then the ten favorites with their scores.
fn print_scores(scores: &crate::position_map::PositionMap<f32, 1>, board: &Board) {
    let mut ranked_positions = board.empty_position_ids();
    ranked_positions
        .sort_by(|&earlier, &later| scores.get(later)[0].total_cmp(&scores.get(earlier)[0]));

    println!(
        "{} to move; empty positions ranked by score:",
        board.stone_to_move()
    );
    for row in PositionId::rows() {
        let cells: Vec<String> = row
            .into_iter()
            .map(|position| match board.stone(position) {
                Some(Stone::Black) => "  X".to_string(),
                Some(Stone::White) => "  O".to_string(),
                None => {
                    let rank = ranked_positions
                        .iter()
                        .position(|&ranked| ranked == position)
                        .expect("every empty position is ranked")
                        + 1;
                    if rank < 100 {
                        format!("{rank:3}")
                    } else {
                        "  .".to_string()
                    }
                }
            })
            .collect();
        println!("{}", cells.join(""));
    }

    println!();
    for (index, &position) in ranked_positions.iter().take(10).enumerate() {
        println!(
            "{:2}. ({:2}, {:2}) {:10.4}",
            index + 1,
            position.row(),
            position.col(),
            scores.get(position)[0]
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;

    fn write_board(text: &str) -> tempfile::NamedTempFile {
        let file = tempfile::NamedTempFile::new().unwrap();
        fs::write(file.path(), text).unwrap();
        file
    }

    #[test]
    fn reads_a_drawn_board_with_either_empty_symbol() {
        let mut rows = vec![". ".repeat(PositionId::WIDTH); PositionId::WIDTH];
        rows[3] = "· · X · · · · · · · · · · O ·".to_string();
        let file = write_board(&rows.join("\n"));

        let board = read_board_file(file.path()).unwrap();

        assert_eq!(
            board.stone(PositionId::from_position(Position::new(3, 2))),
            Some(Stone::Black)
        );
        assert_eq!(
            board.stone(PositionId::from_position(Position::new(3, 13))),
            Some(Stone::White)
        );
        assert_eq!(board.move_count(), 2);
        assert_eq!(board.stone_to_move(), Stone::Black);
    }

    #[test]
    fn rejects_a_drawing_with_the_wrong_number_of_positions() {
        let file = write_board(&".".repeat(PositionId::COUNT - 1));

        let error = read_board_file(file.path()).unwrap_err();

        assert!(error.to_string().contains("expected 225"), "{error}");
    }

    #[test]
    fn rejects_an_unknown_symbol() {
        let file = write_board(&format!("?{}", ".".repeat(PositionId::COUNT - 1)));

        let error = read_board_file(file.path()).unwrap_err();

        assert!(error.to_string().contains("unexpected stone"), "{error}");
    }
}
