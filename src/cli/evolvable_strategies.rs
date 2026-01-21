use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::cluster::strategy::{NN1, NN2, NN3, NN4};
use crate::gene::Gene;
use crate::strategy::EvolvableStrategy;

/// Binary-serializable strategy data (genes only, label comes from filename).
#[derive(Debug, Serialize, Deserialize)]
pub enum StrategyData {
    Nn1 { genes: Vec<Gene> },
    Nn2 { genes: Vec<Gene> },
    Nn3 { genes: Vec<Gene> },
    Nn4 { genes: Vec<Gene> },
}

/// A homogeneous collection of strategies that can be evolved together.
#[derive(Debug)]
pub enum EvolvableStrategies {
    Nn1 { strategies: Vec<NN1> },
    Nn2 { strategies: Vec<NN2> },
    Nn3 { strategies: Vec<NN3> },
    Nn4 { strategies: Vec<NN4> },
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
    }
}

fn save_each<S: EvolvableStrategy>(
    dir: &Path,
    strategies: &[S],
    wrap: fn(Vec<Gene>) -> StrategyData,
) -> Result<()> {
    let width = strategies.len().to_string().len();

    for (i, strategy) in strategies.iter().enumerate() {
        let data = wrap(strategy.genes().to_vec());
        let bytes = postcard::to_allocvec(&data).context("Failed to serialize strategy")?;

        let rank = i + 1;
        let path = dir.join(format!("{rank:0width$}_{}.bin", strategy.label()));
        fs::write(&path, bytes).with_context(|| format!("Failed to write: {}", path.display()))?;
    }

    Ok(())
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

    let cwd = std::env::current_dir().unwrap_or_default();
    let mut loaded = Vec::with_capacity(entries.len());

    for entry in entries {
        let path = entry.path();
        let label = path
            .strip_prefix(&cwd)
            .unwrap_or(&path)
            .display()
            .to_string();

        let bytes =
            fs::read(&path).with_context(|| format!("Failed to read: {}", path.display()))?;

        let data: StrategyData = postcard::from_bytes(&bytes)
            .with_context(|| format!("Failed to deserialize: {}", path.display()))?;

        loaded.push((label, data));
    }

    build_strategies(loaded)
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
        let original_genes: Vec<_> = strategies.iter().map(|s| s.genes().to_vec()).collect();
        let original = EvolvableStrategies::Nn1 { strategies };

        save_strategies_to_directory(dir.path(), &original).unwrap();
        let loaded = load_strategies_from_directory(dir.path()).unwrap();

        match loaded {
            EvolvableStrategies::Nn1 { strategies } => {
                assert_eq!(strategies.len(), 2);
                assert_eq!(strategies[0].genes(), original_genes[0].as_slice());
                assert_eq!(strategies[1].genes(), original_genes[1].as_slice());
            }
            _ => panic!("Expected Nn1"),
        }
    }

    #[test]
    fn save_and_load_nn4_round_trip() {
        let dir = temp_dir();
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = NN4::random("test", &mut rng);
        let original_genes = strategy.genes().to_vec();
        let original = EvolvableStrategies::Nn4 {
            strategies: vec![strategy],
        };

        save_strategies_to_directory(dir.path(), &original).unwrap();
        let loaded = load_strategies_from_directory(dir.path()).unwrap();

        match loaded {
            EvolvableStrategies::Nn4 { strategies } => {
                assert_eq!(strategies[0].genes(), original_genes.as_slice());
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
}
