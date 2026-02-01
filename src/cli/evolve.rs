use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::{Args, Subcommand};
use tracing::info;

use super::evolvable_strategies::{
    EvolvableStrategies, load_strategies_from_directory, save_strategies_to_directory,
};
use super::{PermutationCrossover, PermutationMutation, create_rng, setup_logging};
use crate::cluster::strategy::{NN1, NN2, NN3, NN4};
use crate::conv::{ConvSmall, ConvTiny};
use crate::evolution::crossover::Crossover;
use crate::evolution::mutation::Mutation;
use crate::evolution::{Evolver, Population};
use crate::game::Freestyle;
use crate::strategy::EvolvableStrategy;
use crate::tournament::Swiss;

/// Strategy subcommands for evolution.
#[derive(Debug, Subcommand)]
pub enum EvolveCommand {
    /// Evolve NN1 strategies (70 fingerprints, 4 orthogonal neighbors)
    Nn1(PermutationArgs),
    /// Evolve NN2 strategies (~1,100 fingerprints, 8 neighbors)
    Nn2(PermutationArgs),
    /// Evolve NN3 strategies (~10,000 fingerprints)
    Nn3(PermutationArgs),
    /// Evolve NN4 strategies (~100,000 fingerprints)
    Nn4(PermutationArgs),
    /// Evolve `ConvTiny` strategies (CNN with 3x3 kernels, 32 channels, ~10K params)
    ConvTiny(ConvArgs),
    /// Evolve `ConvSmall` strategies (CNN with 3x3 kernels, 64 channels, ~112K params)
    ConvSmall(ConvArgs),
}

/// Common evolution parameters shared by all strategies.
#[derive(Debug, Args)]
pub struct CommonArgs {
    /// Load strategies from directory (skips random generation)
    #[arg(short, long, value_name = "DIR")]
    pub input: Option<PathBuf>,

    /// Number of random strategies to generate (required without --input)
    #[arg(short, long, value_name = "N")]
    pub population: Option<usize>,

    /// Output directory for evolved strategies
    #[arg(short, long, value_name = "DIR")]
    pub output: PathBuf,

    /// Number of generations to evolve
    #[arg(short, long, default_value = "10")]
    pub generations: u32,

    /// Number of top performers preserved each generation
    #[arg(short, long, default_value = "2")]
    pub elitism: usize,

    /// Crossover probability (0.0 to 1.0)
    #[arg(long, default_value = "0.8")]
    pub crossover_rate: f64,

    /// Mutation probability (0.0 to 1.0)
    #[arg(long, default_value = "0.1")]
    pub mutation_rate: f64,

    /// RNG seed for reproducibility
    #[arg(long)]
    pub seed: Option<u64>,

    /// Log level (error, warn, info, debug, trace)
    #[arg(short, long, value_name = "LEVEL")]
    pub log_level: Option<String>,
}

/// Arguments for evolving permutation-based strategies (NN1-4).
#[derive(Debug, Args)]
pub struct PermutationArgs {
    #[command(flatten)]
    pub common: CommonArgs,

    /// Crossover operator
    #[arg(short, long, default_value = "order")]
    pub crossover: PermutationCrossover,

    /// Mutation operator
    #[arg(short, long, default_value = "swap")]
    pub mutation: PermutationMutation,
}

/// Arguments for evolving convolutional strategies (`ConvTiny`, `ConvSmall`, etc.).
#[derive(Debug, Args)]
pub struct ConvArgs {
    #[command(flatten)]
    pub common: CommonArgs,

    /// Gaussian mutation sigma
    #[arg(long, default_value = "0.01")]
    pub sigma: f32,
}

/// Runs the evolve subcommand.
///
/// # Errors
///
/// Returns an error if logging setup fails, strategy loading/generation fails,
/// serialization fails, or file I/O fails.
pub fn run_evolve(cmd: &EvolveCommand) -> Result<()> {
    match cmd {
        EvolveCommand::Nn1(args) => run_permutation::<NN1>(args, "Nn1", extract_nn1, wrap_nn1),
        EvolveCommand::Nn2(args) => run_permutation::<NN2>(args, "Nn2", extract_nn2, wrap_nn2),
        EvolveCommand::Nn3(args) => run_permutation::<NN3>(args, "Nn3", extract_nn3, wrap_nn3),
        EvolveCommand::Nn4(args) => run_permutation::<NN4>(args, "Nn4", extract_nn4, wrap_nn4),
        EvolveCommand::ConvTiny(args) => {
            run_conv::<ConvTiny>(args, "ConvTiny", extract_conv_tiny, wrap_conv_tiny)
        }
        EvolveCommand::ConvSmall(args) => {
            run_conv::<ConvSmall>(args, "ConvSmall", extract_conv_small, wrap_conv_small)
        }
    }
}

fn run_permutation<S: EvolvableStrategy>(
    args: &PermutationArgs,
    type_name: &str,
    extract: fn(EvolvableStrategies) -> Result<Vec<S>>,
    wrap: fn(Vec<S>) -> EvolvableStrategies,
) -> Result<()> {
    setup_logging(args.common.log_level.as_deref())?;
    let mut rng = create_rng(args.common.seed);

    let strategies = load_or_generate(&args.common, extract, &mut rng)?;
    info!(
        strategy_type = type_name,
        population = strategies.len(),
        "Starting evolution"
    );

    let evolver = create_evolver(&args.common, args.crossover.into(), args.mutation.into());
    let population: Population<S> = evolver.evolve(strategies, &mut rng);
    info!(generation = population.generation(), "Evolution complete");

    save_strategies_to_directory(&args.common.output, &wrap(population.into_strategies()))?;
    info!(path = %args.common.output.display(), "Saved strategies");

    Ok(())
}

fn run_conv<S: EvolvableStrategy>(
    args: &ConvArgs,
    type_name: &str,
    extract: fn(EvolvableStrategies) -> Result<Vec<S>>,
    wrap: fn(Vec<S>) -> EvolvableStrategies,
) -> Result<()> {
    setup_logging(args.common.log_level.as_deref())?;
    let mut rng = create_rng(args.common.seed);

    let strategies = load_or_generate(&args.common, extract, &mut rng)?;
    info!(
        strategy_type = type_name,
        population = strategies.len(),
        "Starting evolution"
    );

    let evolver = create_evolver(
        &args.common,
        Crossover::Uniform,
        Mutation::Gaussian { sigma: args.sigma },
    );
    let population: Population<S> = evolver.evolve(strategies, &mut rng);
    info!(generation = population.generation(), "Evolution complete");

    save_strategies_to_directory(&args.common.output, &wrap(population.into_strategies()))?;
    info!(path = %args.common.output.display(), "Saved strategies");

    Ok(())
}

fn create_evolver(common: &CommonArgs, crossover: Crossover, mutation: Mutation) -> Evolver {
    Evolver::new()
        .generations(common.generations)
        .elitism(common.elitism)
        .crossover(crossover)
        .crossover_rate(common.crossover_rate)
        .mutation(mutation)
        .mutation_rate(common.mutation_rate)
        .tournament(Swiss.into())
        .game(Freestyle.into())
}

fn load_or_generate<S: EvolvableStrategy>(
    common: &CommonArgs,
    extract: fn(EvolvableStrategies) -> Result<Vec<S>>,
    rng: &mut fastrand::Rng,
) -> Result<Vec<S>> {
    if let Some(ref input_dir) = common.input {
        let strategies = load_strategies_from_directory(input_dir)?;
        info!(path = %input_dir.display(), "Loaded strategies from directory");
        extract(strategies)
    } else {
        let population = common
            .population
            .ok_or_else(|| anyhow::anyhow!("--population required when not using --input"))?;
        info!(count = population, "Generating random strategies");
        Ok(generate_random(population, rng))
    }
}

fn generate_random<S: EvolvableStrategy>(population: usize, rng: &mut fastrand::Rng) -> Vec<S> {
    (0..population)
        .map(|i| S::random(format!("gen0_{i}"), rng))
        .collect()
}

// Extract functions for each strategy type
fn extract_nn1(strategies: EvolvableStrategies) -> Result<Vec<NN1>> {
    match strategies {
        EvolvableStrategies::Nn1 { strategies } => Ok(strategies),
        _ => bail!("Expected NN1 strategies, found different type"),
    }
}

fn extract_nn2(strategies: EvolvableStrategies) -> Result<Vec<NN2>> {
    match strategies {
        EvolvableStrategies::Nn2 { strategies } => Ok(strategies),
        _ => bail!("Expected NN2 strategies, found different type"),
    }
}

fn extract_nn3(strategies: EvolvableStrategies) -> Result<Vec<NN3>> {
    match strategies {
        EvolvableStrategies::Nn3 { strategies } => Ok(strategies),
        _ => bail!("Expected NN3 strategies, found different type"),
    }
}

fn extract_nn4(strategies: EvolvableStrategies) -> Result<Vec<NN4>> {
    match strategies {
        EvolvableStrategies::Nn4 { strategies } => Ok(strategies),
        _ => bail!("Expected NN4 strategies, found different type"),
    }
}

fn extract_conv_tiny(strategies: EvolvableStrategies) -> Result<Vec<ConvTiny>> {
    match strategies {
        EvolvableStrategies::ConvTiny { strategies } => Ok(strategies),
        _ => bail!("Expected ConvTiny strategies, found different type"),
    }
}

fn extract_conv_small(strategies: EvolvableStrategies) -> Result<Vec<ConvSmall>> {
    match strategies {
        EvolvableStrategies::ConvSmall { strategies } => Ok(strategies),
        _ => bail!("Expected ConvSmall strategies, found different type"),
    }
}

// Wrap functions for each NN strategy type
fn wrap_nn1(strategies: Vec<NN1>) -> EvolvableStrategies {
    EvolvableStrategies::Nn1 { strategies }
}

fn wrap_nn2(strategies: Vec<NN2>) -> EvolvableStrategies {
    EvolvableStrategies::Nn2 { strategies }
}

fn wrap_nn3(strategies: Vec<NN3>) -> EvolvableStrategies {
    EvolvableStrategies::Nn3 { strategies }
}

fn wrap_nn4(strategies: Vec<NN4>) -> EvolvableStrategies {
    EvolvableStrategies::Nn4 { strategies }
}

fn wrap_conv_tiny(strategies: Vec<ConvTiny>) -> EvolvableStrategies {
    EvolvableStrategies::ConvTiny { strategies }
}

fn wrap_conv_small(strategies: Vec<ConvSmall>) -> EvolvableStrategies {
    EvolvableStrategies::ConvSmall { strategies }
}
