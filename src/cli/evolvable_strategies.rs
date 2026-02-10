use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::conv::{ConvSmall, ConvTiny};
use crate::strategy::{EvolvableStrategy, Strategy};

/// Binary-serializable strategy data (genes only, label comes from filename).
#[derive(Debug, Serialize, Deserialize)]
pub enum StrategyData {
    ConvTiny { genes: Vec<f32> },
    ConvSmall { genes: Vec<f32> },
}

/// A homogeneous collection of strategies that can be evolved together.
#[derive(Debug)]
pub enum EvolvableStrategies {
    ConvTiny { strategies: Vec<ConvTiny> },
    ConvSmall { strategies: Vec<ConvSmall> },
}

/// Saves strategies to a directory as individual binary files.
///
/// Each strategy is saved as `{padded_rank}_{label}.bin` where rank starts at 1.
///
/// # Errors
///
/// Returns an error if directory creation, serialization, or file I/O fails.
pub fn save_strategies_to_directory(dir: &Path, strategies: &EvolvableStrategies) -> Result<()> {
    fs::create_dir_all(dir)
        .with_context(|| format!("Failed to create directory: {}", dir.display()))?;

    match strategies {
        EvolvableStrategies::ConvTiny { strategies } => {
            save_each(dir, strategies, |genes| StrategyData::ConvTiny { genes })
        }
        EvolvableStrategies::ConvSmall { strategies } => {
            save_each(dir, strategies, |genes| StrategyData::ConvSmall { genes })
        }
    }
}

fn save_each<S: EvolvableStrategy, T>(
    dir: &Path,
    strategies: &[S],
    wrap: fn(Vec<T>) -> StrategyData,
) -> Result<()>
where
    S::Genes: Into<Vec<T>>,
{
    let width = strategies.len().to_string().len();

    for (i, strategy) in strategies.iter().enumerate() {
        let data = wrap(strategy.genes().clone().into());
        let bytes = postcard::to_allocvec(&data).context("Failed to serialize strategy")?;

        let rank = i + 1;
        let path = dir.join(format!("{rank:0width$}_{}.bin", strategy.label()));
        fs::write(&path, bytes).with_context(|| format!("Failed to write: {}", path.display()))?;
    }

    Ok(())
}

/// Loads a single strategy from a binary file.
///
/// # Errors
///
/// Returns an error if the file cannot be read or deserialization fails.
pub fn load_strategy_from_file(path: &Path) -> Result<Box<dyn Strategy>> {
    let (label, data) = load_strategy_data(path)?;

    Ok(match data {
        StrategyData::ConvTiny { genes } => Box::new(ConvTiny::from_genes(label, genes.into())),
        StrategyData::ConvSmall { genes } => Box::new(ConvSmall::from_genes(label, genes.into())),
    })
}

/// Loads strategies from a directory of binary files.
///
/// Reads all `.bin` files sorted by filename. Labels are derived from file paths.
///
/// # Errors
///
/// Returns an error if no `.bin` files found, deserialization fails, or types are mixed.
pub fn load_strategies_from_directory(dir: &Path) -> Result<EvolvableStrategies> {
    let mut entries: Vec<_> = fs::read_dir(dir)
        .with_context(|| format!("Failed to read directory: {}", dir.display()))?
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "bin"))
        .collect();

    if entries.is_empty() {
        bail!("No .bin files found in: {}", dir.display());
    }

    entries.sort_by_key(std::fs::DirEntry::file_name);

    let mut loaded = Vec::with_capacity(entries.len());
    for entry in entries {
        loaded.push(load_strategy_data(&entry.path())?);
    }

    build_strategies(loaded)
}

fn load_strategy_data(path: &Path) -> Result<(String, StrategyData)> {
    let cwd = std::env::current_dir().unwrap_or_default();
    let label = path
        .strip_prefix(&cwd)
        .unwrap_or(path)
        .display()
        .to_string();

    let bytes = fs::read(path).with_context(|| format!("Failed to read: {}", path.display()))?;

    let data: StrategyData = postcard::from_bytes(&bytes)
        .with_context(|| format!("Failed to deserialize: {}", path.display()))?;

    Ok((label, data))
}

fn build_strategies(loaded: Vec<(String, StrategyData)>) -> Result<EvolvableStrategies> {
    let first_discriminant = std::mem::discriminant(&loaded[0].1);

    for (label, data) in &loaded[1..] {
        if std::mem::discriminant(data) != first_discriminant {
            bail!("Mixed strategy types: {label} has different type");
        }
    }

    match &loaded[0].1 {
        StrategyData::ConvTiny { .. } => Ok(EvolvableStrategies::ConvTiny {
            strategies: loaded
                .into_iter()
                .map(|(label, data)| match data {
                    StrategyData::ConvTiny { genes } => ConvTiny::from_genes(label, genes.into()),
                    StrategyData::ConvSmall { .. } => unreachable!(),
                })
                .collect(),
        }),
        StrategyData::ConvSmall { .. } => Ok(EvolvableStrategies::ConvSmall {
            strategies: loaded
                .into_iter()
                .map(|(label, data)| match data {
                    StrategyData::ConvSmall { genes } => ConvSmall::from_genes(label, genes.into()),
                    StrategyData::ConvTiny { .. } => unreachable!(),
                })
                .collect(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn save_and_load_conv_tiny_round_trip() {
        let dir = temp_dir();
        let mut rng = fastrand::Rng::with_seed(42);
        let strategies = vec![
            ConvTiny::random("c0", &mut rng),
            ConvTiny::random("c1", &mut rng),
        ];
        let original_genes: Vec<_> = strategies
            .iter()
            .map(|s| s.genes().as_ref().to_vec())
            .collect();
        let original = EvolvableStrategies::ConvTiny { strategies };

        save_strategies_to_directory(dir.path(), &original).unwrap();
        let loaded = load_strategies_from_directory(dir.path()).unwrap();

        match loaded {
            EvolvableStrategies::ConvTiny { strategies } => {
                assert_eq!(strategies.len(), 2);
                assert_eq!(strategies[0].genes().as_ref(), &original_genes[0][..]);
                assert_eq!(strategies[1].genes().as_ref(), &original_genes[1][..]);
            }
            _ => panic!("Expected ConvTiny"),
        }
    }

    #[test]
    fn filenames_have_padded_ranks() {
        let dir = temp_dir();
        let mut rng = fastrand::Rng::with_seed(42);
        let strategies = EvolvableStrategies::ConvTiny {
            strategies: (0..12)
                .map(|i| ConvTiny::random(format!("s{i}"), &mut rng))
                .collect(),
        };

        save_strategies_to_directory(dir.path(), &strategies).unwrap();

        let mut files: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        files.sort();

        assert_eq!(files[0], "01_s0.bin");
        assert_eq!(files[9], "10_s9.bin");
        assert_eq!(files[11], "12_s11.bin");
    }

    #[test]
    fn load_empty_directory_fails() {
        let dir = temp_dir();

        let result = load_strategies_from_directory(dir.path());

        assert!(result.unwrap_err().to_string().contains("No .bin files"));
    }
}
