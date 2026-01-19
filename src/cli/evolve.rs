use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Args;
use tracing::info;

use super::evolvable_strategies::EvolvableStrategies;
use super::{CliCrossover, CliMutation, CliStrategy};
use crate::evolution::{Evolver, Population};
use crate::game::Freestyle;
use crate::strategy::EvolvableStrategy;
use crate::tournament::Swiss;

/// Arguments for the evolve subcommand.
#[derive(Debug, Args)]
pub struct EvolveArgs {
    /// Strategy type (nn1, nn2, nn3, nn4). Required when generating random strategies.
    #[arg(value_name = "STRATEGY")]
    pub strategy_type: Option<CliStrategy>,

    /// Number of random strategies to generate. Required with STRATEGY.
    #[arg(value_name = "POPULATION")]
    pub population: Option<usize>,

    /// Load strategies from RON file (conflicts with STRATEGY/POPULATION).
    #[arg(short, long, value_name = "FILE", conflicts_with_all = ["strategy_type", "population"])]
    pub input: Option<PathBuf>,

    /// Output file for evolved strategies.
    #[arg(short, long, value_name = "FILE")]
    pub output: PathBuf,

    /// Number of generations to evolve.
    #[arg(short, long, default_value = "10")]
    pub generations: u32,

    /// Number of top performers preserved each generation.
    #[arg(short, long, default_value = "2")]
    pub elitism: usize,

    /// Crossover operator.
    #[arg(short, long, default_value = "order")]
    pub crossover: CliCrossover,

    /// Crossover probability (0.0 to 1.0).
    #[arg(long, default_value = "0.8")]
    pub crossover_rate: f64,

    /// Mutation operator.
    #[arg(short, long, default_value = "swap")]
    pub mutation: CliMutation,

    /// Mutation probability (0.0 to 1.0).
    #[arg(long, default_value = "0.1")]
    pub mutation_rate: f64,

    /// RNG seed for reproducibility.
    #[arg(long)]
    pub seed: Option<u64>,

    /// Log level (error, warn, info, debug, trace).
    #[arg(short, long, value_name = "LEVEL")]
    pub log_level: Option<String>,
}

/// Runs the evolve subcommand.
///
/// # Errors
///
/// Returns an error if logging setup fails, strategy loading/generation fails,
/// serialization fails, or file I/O fails.
pub fn run_evolve(args: &EvolveArgs) -> Result<()> {
    setup_logging(args.log_level.as_deref())?;

    let mut rng = if let Some(seed) = args.seed {
        info!(seed, "Using provided RNG seed");
        fastrand::Rng::with_seed(seed)
    } else {
        info!("Using random RNG seed");
        fastrand::Rng::new()
    };

    let strategies = load_or_generate(args, &mut rng)?;

    let evolver = Evolver::new()
        .generations(args.generations)
        .elitism(args.elitism)
        .crossover(args.crossover.into())
        .crossover_rate(args.crossover_rate)
        .mutation(args.mutation.into())
        .mutation_rate(args.mutation_rate)
        .tournament(Swiss.into())
        .game(Freestyle.into());

    let output_strategies = match strategies {
        EvolvableStrategies::Nn1 { strategies } => {
            info!(
                strategy_type = "Nn1",
                population = strategies.len(),
                "Starting evolution"
            );
            EvolvableStrategies::Nn1 {
                strategies: run_evolution(&evolver, strategies, &mut rng),
            }
        }
        EvolvableStrategies::Nn2 { strategies } => {
            info!(
                strategy_type = "Nn2",
                population = strategies.len(),
                "Starting evolution"
            );
            EvolvableStrategies::Nn2 {
                strategies: run_evolution(&evolver, strategies, &mut rng),
            }
        }
        EvolvableStrategies::Nn3 { strategies } => {
            info!(
                strategy_type = "Nn3",
                population = strategies.len(),
                "Starting evolution"
            );
            EvolvableStrategies::Nn3 {
                strategies: run_evolution(&evolver, strategies, &mut rng),
            }
        }
        EvolvableStrategies::Nn4 { strategies } => {
            info!(
                strategy_type = "Nn4",
                population = strategies.len(),
                "Starting evolution"
            );
            EvolvableStrategies::Nn4 {
                strategies: run_evolution(&evolver, strategies, &mut rng),
            }
        }
    };

    let config = ron::ser::PrettyConfig::default()
        .depth_limit(2)
        .compact_arrays(true)
        .separator(String::new());
    let output = ron::ser::to_string_pretty(&output_strategies, config)
        .context("Failed to serialize strategies")?;

    fs::write(&args.output, output)
        .with_context(|| format!("Failed to write output file: {}", args.output.display()))?;

    info!(path = %args.output.display(), "Saved strategies");

    Ok(())
}

fn setup_logging(log_level: Option<&str>) -> Result<()> {
    use tracing_subscriber::EnvFilter;
    use tracing_subscriber::fmt::format::FmtSpan;

    let filter = match log_level {
        Some(level) => {
            EnvFilter::try_new(level).with_context(|| format!("Invalid log level: {level}"))?
        }
        None => EnvFilter::from_default_env(),
    };

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_span_events(FmtSpan::CLOSE)
        .init();

    Ok(())
}

fn load_or_generate(args: &EvolveArgs, rng: &mut fastrand::Rng) -> Result<EvolvableStrategies> {
    if let Some(ref input_path) = args.input {
        let content = fs::read_to_string(input_path)
            .with_context(|| format!("Failed to read input file: {}", input_path.display()))?;

        let strategies: EvolvableStrategies = ron::from_str(&content).with_context(|| {
            format!("Failed to parse strategies from: {}", input_path.display())
        })?;

        info!(path = %input_path.display(), "Loaded strategies from file");
        Ok(strategies)
    } else {
        let strategy_type = args.strategy_type.ok_or_else(|| {
            anyhow::anyhow!("STRATEGY argument is required when not using --input")
        })?;

        let population = args.population.ok_or_else(|| {
            anyhow::anyhow!("POPULATION argument is required when not using --input")
        })?;

        info!(count = population, "Generating random strategies");

        Ok(match strategy_type {
            CliStrategy::Nn1 => EvolvableStrategies::Nn1 {
                strategies: generate_random(population, rng),
            },
            CliStrategy::Nn2 => EvolvableStrategies::Nn2 {
                strategies: generate_random(population, rng),
            },
            CliStrategy::Nn3 => EvolvableStrategies::Nn3 {
                strategies: generate_random(population, rng),
            },
            CliStrategy::Nn4 => EvolvableStrategies::Nn4 {
                strategies: generate_random(population, rng),
            },
        })
    }
}

fn generate_random<S: EvolvableStrategy>(population: usize, rng: &mut fastrand::Rng) -> Vec<S> {
    (0..population)
        .map(|i| S::random(format!("gen0_{i}"), rng))
        .collect()
}

fn run_evolution<S: EvolvableStrategy>(
    evolver: &Evolver,
    strategies: Vec<S>,
    rng: &mut fastrand::Rng,
) -> Vec<S> {
    let population: Population<S> = evolver.evolve(strategies, rng);
    info!(generation = population.generation(), "Evolution complete");
    population.into_strategies()
}
