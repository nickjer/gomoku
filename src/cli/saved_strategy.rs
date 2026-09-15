use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::board::Board;
use crate::nn::{ClusterSmall, ClusterTiny, ConvSmall, ConvTiny};
use crate::position_id::PositionId;
use crate::position_map::PositionMap;
use crate::stone::Stone;
use crate::strategy::Strategy;

/// A strategy as saved to a `.bin` file, tagged with which kind it is.
///
/// This list is the registry of strategy kinds the CLI knows. Each variant's
/// doc line is the help text for `evolve <KIND>`.
#[derive(
    Debug,
    Serialize,
    Deserialize,
    derive_more::Display,
    derive_more::From,
    derive_more::TryInto,
    strum::EnumDiscriminants,
)]
#[strum_discriminants(name(StrategyKind), derive(clap::ValueEnum))]
pub enum SavedStrategy {
    /// 3x3 squares, 32 channels, 2 layers, ~10K parameters
    ConvTiny(ConvTiny),
    /// 3x3 squares, 64 channels, 4 layers, ~112K parameters
    ConvSmall(ConvSmall),
    /// Cluster expansion, 32 channels, 2 layers, ~10K parameters
    ClusterTiny(ClusterTiny),
    /// Cluster expansion, 64 channels, 4 layers, ~112K parameters
    ClusterSmall(ClusterSmall),
}

impl SavedStrategy {
    fn as_strategy(&self) -> &dyn Strategy {
        match self {
            Self::ConvTiny(strategy) => strategy,
            Self::ConvSmall(strategy) => strategy,
            Self::ClusterTiny(strategy) => strategy,
            Self::ClusterSmall(strategy) => strategy,
        }
    }

    /// The network's score for every position, reading the board as it lies
    /// from the current player's side.
    #[must_use]
    pub fn score_positions(&self, current_stone: Stone, board: &Board) -> PositionMap<f32, 1> {
        match self {
            Self::ConvTiny(strategy) => strategy.score_positions(current_stone, board),
            Self::ConvSmall(strategy) => strategy.score_positions(current_stone, board),
            Self::ClusterTiny(strategy) => strategy.score_positions(current_stone, board),
            Self::ClusterSmall(strategy) => strategy.score_positions(current_stone, board),
        }
    }
}

impl Strategy for SavedStrategy {
    fn choose_move(
        &self,
        current_stone: Stone,
        board: &Board,
        rng: &mut fastrand::Rng,
    ) -> PositionId {
        self.as_strategy().choose_move(current_stone, board, rng)
    }

    fn label(&self) -> &str {
        self.as_strategy().label()
    }
}

/// Reads one strategy file.
///
/// # Errors
///
/// Returns an error if the file cannot be read or does not hold a strategy.
pub fn read_strategy_file(path: &Path) -> Result<SavedStrategy> {
    let bytes = fs::read(path).with_context(|| format!("Failed to read: {}", path.display()))?;
    postcard::from_bytes(&bytes)
        .with_context(|| format!("Failed to deserialize: {}", path.display()))
}

/// Reads every `.bin` file in a directory, sorted by filename.
///
/// # Errors
///
/// Returns an error if the directory cannot be read, holds no `.bin` files,
/// or one of the files does not hold a strategy.
pub fn read_strategy_directory(dir: &Path) -> Result<Vec<SavedStrategy>> {
    let mut paths: Vec<_> = fs::read_dir(dir)
        .with_context(|| format!("Failed to read directory: {}", dir.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "bin"))
        .collect();

    if paths.is_empty() {
        bail!("No .bin files found in: {}", dir.display());
    }

    paths.sort();
    paths.iter().map(|path| read_strategy_file(path)).collect()
}

/// Writes strategies to a directory as `{rank}_{label}.bin`, rank starting
/// at 1 and zero-padded so the files sort in rank order.
///
/// # Errors
///
/// Returns an error if the directory cannot be created or a file cannot be
/// written.
pub fn write_strategy_directory(dir: &Path, strategies: &[SavedStrategy]) -> Result<()> {
    fs::create_dir_all(dir)
        .with_context(|| format!("Failed to create directory: {}", dir.display()))?;

    let width = strategies.len().to_string().len();
    for (index, strategy) in strategies.iter().enumerate() {
        let bytes = postcard::to_allocvec(strategy).context("Failed to serialize strategy")?;
        let rank = index + 1;
        let path = dir.join(format!("{rank:0width$}_{}.bin", strategy.label()));
        fs::write(&path, bytes).with_context(|| format!("Failed to write: {}", path.display()))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::EvolvableStrategy;

    fn temp_dir() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn write_then_read_directory_round_trips_strategies() {
        let dir = temp_dir();
        let mut rng = fastrand::Rng::with_seed(42);
        let originals = [
            ConvTiny::random("c0", &mut rng),
            ConvTiny::random("c1", &mut rng),
        ];
        let saved: Vec<SavedStrategy> = originals.iter().cloned().map(Into::into).collect();

        write_strategy_directory(dir.path(), &saved).unwrap();
        let loaded: Vec<ConvTiny> = read_strategy_directory(dir.path())
            .unwrap()
            .into_iter()
            .map(|saved| saved.try_into().unwrap())
            .collect();

        assert_eq!(loaded.len(), 2);
        for (loaded, original) in loaded.iter().zip(&originals) {
            assert_eq!(loaded.label(), original.label());
            assert_eq!(loaded.genes(), original.genes());
        }
    }

    #[test]
    fn filenames_have_padded_ranks() {
        let dir = temp_dir();
        let mut rng = fastrand::Rng::with_seed(42);
        let saved: Vec<SavedStrategy> = (0..12)
            .map(|i| ConvTiny::random(format!("s{i}"), &mut rng).into())
            .collect();

        write_strategy_directory(dir.path(), &saved).unwrap();

        let mut files: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .filter_map(Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().to_string())
            .collect();
        files.sort();

        assert_eq!(files[0], "01_s0.bin");
        assert_eq!(files[9], "10_s9.bin");
        assert_eq!(files[11], "12_s11.bin");
    }

    #[test]
    fn read_empty_directory_fails() {
        let dir = temp_dir();

        let error = read_strategy_directory(dir.path()).unwrap_err();

        assert!(error.to_string().contains("No .bin files"));
    }

    #[test]
    fn read_missing_directory_fails() {
        let error = read_strategy_directory(Path::new("does/not/exist")).unwrap_err();

        assert!(error.to_string().contains("Failed to read directory"));
    }

    #[test]
    fn read_missing_file_fails() {
        let error = read_strategy_file(Path::new("does/not/exist.bin")).unwrap_err();

        assert!(error.to_string().contains("Failed to read"));
    }

    #[test]
    fn read_file_that_is_not_a_strategy_fails() {
        let dir = temp_dir();
        let path = dir.path().join("junk.bin");
        fs::write(&path, b"not a strategy").unwrap();

        let error = read_strategy_file(&path).unwrap_err();

        assert!(error.to_string().contains("Failed to deserialize"));
    }

    #[test]
    fn converting_to_another_kind_fails() {
        let mut rng = fastrand::Rng::with_seed(42);
        let saved: SavedStrategy = ConvTiny::random("c0", &mut rng).into();

        let wrong_kind = ConvSmall::try_from(saved);

        let error = wrong_kind.unwrap_err();
        assert_eq!(StrategyKind::from(&error.input), StrategyKind::ConvTiny);
    }

    #[test]
    fn strategy_kind_names_the_variant() {
        let mut rng = fastrand::Rng::with_seed(42);
        let saved: SavedStrategy = ClusterSmall::random("k", &mut rng).into();

        assert_eq!(StrategyKind::from(&saved), StrategyKind::ClusterSmall);
    }

    #[test]
    fn saved_strategy_plays_like_the_strategy_it_holds() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = ConvTiny::random("c0", &mut rng);
        let saved: SavedStrategy = strategy.clone().into();
        let board = Board::new();

        let mut rng1 = fastrand::Rng::with_seed(7);
        let mut rng2 = fastrand::Rng::with_seed(7);
        assert_eq!(saved.label(), "c0");
        assert_eq!(
            saved.choose_move(Stone::Black, &board, &mut rng1),
            strategy.choose_move(Stone::Black, &board, &mut rng2)
        );
    }

    #[test]
    fn every_kind_reports_its_label() {
        let mut rng = fastrand::Rng::with_seed(42);
        let saved: [SavedStrategy; 4] = [
            ConvTiny::random("a", &mut rng).into(),
            ConvSmall::random("b", &mut rng).into(),
            ClusterTiny::random("c", &mut rng).into(),
            ClusterSmall::random("d", &mut rng).into(),
        ];

        let labels: Vec<&str> = saved.iter().map(Strategy::label).collect();

        assert_eq!(labels, ["a", "b", "c", "d"]);
    }

    #[test]
    fn display_shows_label_then_network() {
        let mut rng = fastrand::Rng::with_seed(42);
        let saved: SavedStrategy = ConvTiny::random("shown", &mut rng).into();

        let text = saved.to_string();

        assert!(text.starts_with("Label: shown\n"), "{text}");
        assert!(text.contains("Neural network"), "{text}");
    }
}
