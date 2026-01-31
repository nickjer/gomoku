use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::cluster::strategy::{NN1, NN2, NN3, NN4};
use crate::conv::ConvTiny;
use crate::gene::Gene;
use crate::strategy::{EvolvableStrategy, Strategy};

/// Binary-serializable strategy data (genes only, label comes from filename).
#[derive(Debug, Serialize, Deserialize)]
pub enum StrategyData {
    Nn1 { genes: Vec<Gene> },
    Nn2 { genes: Vec<Gene> },
    Nn3 { genes: Vec<Gene> },
    Nn4 { genes: Vec<Gene> },
    ConvTiny { genes: Vec<f32> },
}

/// A homogeneous collection of strategies that can be evolved together.
#[derive(Debug)]
pub enum EvolvableStrategies {
    Nn1 { strategies: Vec<NN1> },
    Nn2 { strategies: Vec<NN2> },
    Nn3 { strategies: Vec<NN3> },
    Nn4 { strategies: Vec<NN4> },
    ConvTiny { strategies: Vec<ConvTiny> },
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
        EvolvableStrategies::Nn1 { strategies } => {
            save_each(dir, strategies, |genes| StrategyData::Nn1 { genes })
        }
        EvolvableStrategies::Nn2 { strategies } => {
            save_each(dir, strategies, |genes| StrategyData::Nn2 { genes })
        }
        EvolvableStrategies::Nn3 { strategies } => {
            save_each(dir, strategies, |genes| StrategyData::Nn3 { genes })
        }
        EvolvableStrategies::Nn4 { strategies } => {
            save_each(dir, strategies, |genes| StrategyData::Nn4 { genes })
        }
        EvolvableStrategies::ConvTiny { strategies } => {
            save_each(dir, strategies, |genes| StrategyData::ConvTiny { genes })
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
        StrategyData::Nn1 { genes } => Box::new(NN1::from_genes(label, genes)),
        StrategyData::Nn2 { genes } => Box::new(NN2::from_genes(label, genes)),
        StrategyData::Nn3 { genes } => Box::new(NN3::from_genes(label, genes)),
        StrategyData::Nn4 { genes } => Box::new(NN4::from_genes(label, genes)),
        StrategyData::ConvTiny { genes } => Box::new(ConvTiny::from_genes(label, genes.into())),
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
        StrategyData::Nn1 { .. } => Ok(EvolvableStrategies::Nn1 {
            strategies: loaded
                .into_iter()
                .map(|(label, data)| match data {
                    StrategyData::Nn1 { genes } => NN1::from_genes(label, genes),
                    _ => unreachable!(),
                })
                .collect(),
        }),
        StrategyData::Nn2 { .. } => Ok(EvolvableStrategies::Nn2 {
            strategies: loaded
                .into_iter()
                .map(|(label, data)| match data {
                    StrategyData::Nn2 { genes } => NN2::from_genes(label, genes),
                    _ => unreachable!(),
                })
                .collect(),
        }),
        StrategyData::Nn3 { .. } => Ok(EvolvableStrategies::Nn3 {
            strategies: loaded
                .into_iter()
                .map(|(label, data)| match data {
                    StrategyData::Nn3 { genes } => NN3::from_genes(label, genes),
                    _ => unreachable!(),
                })
                .collect(),
        }),
        StrategyData::Nn4 { .. } => Ok(EvolvableStrategies::Nn4 {
            strategies: loaded
                .into_iter()
                .map(|(label, data)| match data {
                    StrategyData::Nn4 { genes } => NN4::from_genes(label, genes),
                    _ => unreachable!(),
                })
                .collect(),
        }),
        StrategyData::ConvTiny { .. } => Ok(EvolvableStrategies::ConvTiny {
            strategies: loaded
                .into_iter()
                .map(|(label, data)| match data {
                    StrategyData::ConvTiny { genes } => ConvTiny::from_genes(label, genes.into()),
                    _ => unreachable!(),
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
    fn save_and_load_nn1_round_trip() {
        let dir = temp_dir();
        let mut rng = fastrand::Rng::with_seed(42);
        let strategies = vec![NN1::random("s0", &mut rng), NN1::random("s1", &mut rng)];
        let original_genes: Vec<_> = strategies.iter().map(|s| s.genes().clone()).collect();
        let original = EvolvableStrategies::Nn1 { strategies };

        save_strategies_to_directory(dir.path(), &original).unwrap();
        let loaded = load_strategies_from_directory(dir.path()).unwrap();

        match loaded {
            EvolvableStrategies::Nn1 { strategies } => {
                assert_eq!(strategies.len(), 2);
                assert_eq!(strategies[0].genes(), &original_genes[0]);
                assert_eq!(strategies[1].genes(), &original_genes[1]);
            }
            _ => panic!("Expected Nn1"),
        }
    }

    #[test]
    fn save_and_load_nn4_round_trip() {
        let dir = temp_dir();
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = NN4::random("test", &mut rng);
        let original_genes = strategy.genes().clone();
        let original = EvolvableStrategies::Nn4 {
            strategies: vec![strategy],
        };

        save_strategies_to_directory(dir.path(), &original).unwrap();
        let loaded = load_strategies_from_directory(dir.path()).unwrap();

        match loaded {
            EvolvableStrategies::Nn4 { strategies } => {
                assert_eq!(strategies[0].genes(), &original_genes);
            }
            _ => panic!("Expected Nn4"),
        }
    }

    #[test]
    fn filenames_have_padded_ranks() {
        let dir = temp_dir();
        let mut rng = fastrand::Rng::with_seed(42);
        let strategies = EvolvableStrategies::Nn1 {
            strategies: (0..12)
                .map(|i| NN1::random(format!("s{i}"), &mut rng))
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
}
