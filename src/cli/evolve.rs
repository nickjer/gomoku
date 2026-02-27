use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use clap::{Args, Subcommand};
use tracing::{info, warn};

use super::evolvable_strategies::{
    EvolvableStrategies, load_strategies_from_directory, save_strategies_to_directory,
};
use super::{create_rng, setup_logging};
use crate::cluster::{ClusterSmall, ClusterTiny};
use crate::conv::{ConvSmall, ConvTiny};
use crate::evolution::crossover::Crossover;
use crate::evolution::fitness_weight::FitnessWeight;
use crate::evolution::mutation::Mutation;
use crate::evolution::{
    Evolver, MinimaxFitness, Population, ThreatDefenseFitness, TournamentFitness,
};
use crate::game::Freestyle;
use crate::strategy::EvolvableStrategy;
use crate::tournament::Swiss;

/// Strategy subcommands for evolution.
#[derive(Debug, Subcommand)]
pub enum EvolveCommand {
    /// Evolve `ConvTiny` strategies (CNN with 3x3 kernels, 32 channels, ~10K params)
    ConvTiny(EvolutionArgs),
    /// Evolve `ConvSmall` strategies (CNN with 3x3 kernels, 64 channels, ~112K params)
    ConvSmall(EvolutionArgs),
    /// Evolve `ClusterTiny` strategies (cluster with 9 features, 32 channels, ~10K params)
    ClusterTiny(EvolutionArgs),
    /// Evolve `ClusterSmall` strategies (cluster with 9 features, 64 channels, ~112K params)
    ClusterSmall(EvolutionArgs),
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

    /// Tournament evaluator weight (0 to disable)
    #[arg(long, default_value = "1.0")]
    pub tournament_weight: f32,

    /// Threat defense evaluator weight (0 to disable)
    #[arg(long, default_value = "0.0")]
    pub defense_weight: f32,

    /// Minimax evaluator weight (0 to disable)
    #[arg(long, default_value = "0.0")]
    pub minimax_weight: f32,

    /// Minimax search depths (evaluated smallest to largest with early cutoff)
    #[arg(long, default_value = "4", num_args = 1..)]
    pub minimax_depth: Vec<u32>,

    /// Depth for per-move minimax scoring (enables move scoring mode when set)
    #[arg(long)]
    pub minimax_scoring_depth: Option<u32>,

    /// Save a checkpoint every N generations (0 to disable)
    #[arg(long, default_value = "0", value_name = "N")]
    pub checkpoint_every: u32,

    /// Log level (error, warn, info, debug, trace)
    #[arg(short, long, value_name = "LEVEL")]
    pub log_level: Option<String>,
}

/// Arguments for evolving strategies.
#[derive(Debug, Args)]
pub struct EvolutionArgs {
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
        EvolveCommand::ConvTiny(args) => {
            run_evolution::<ConvTiny>(args, "ConvTiny", extract_conv_tiny, wrap_conv_tiny)
        }
        EvolveCommand::ConvSmall(args) => {
            run_evolution::<ConvSmall>(args, "ConvSmall", extract_conv_small, wrap_conv_small)
        }
        EvolveCommand::ClusterTiny(args) => run_evolution::<ClusterTiny>(
            args,
            "ClusterTiny",
            extract_cluster_tiny,
            wrap_cluster_tiny,
        ),
        EvolveCommand::ClusterSmall(args) => run_evolution::<ClusterSmall>(
            args,
            "ClusterSmall",
            extract_cluster_small,
            wrap_cluster_small,
        ),
    }
}

fn run_evolution<S: EvolvableStrategy>(
    args: &EvolutionArgs,
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

    let output = &args.common.output;
    let generations = args.common.generations;
    let checkpoint_every = args.common.checkpoint_every;
    let gen_width = generations.max(1).to_string().len();

    let population: Population<S> = evolver.evolve(strategies, &mut rng, |population| {
        let generation = population.generation();
        if checkpoint_every > 0 && generation % checkpoint_every == 0 {
            let dir = output.join(format!("gen_{generation:0gen_width$}"));
            if let Err(err) = save_population(&dir, population, wrap) {
                warn!(%err, "Failed to save checkpoint");
            }
        }
    });
    info!(generation = population.generation(), "Evolution complete");

    let final_gen = population.generation();
    let already_checkpointed = checkpoint_every > 0 && final_gen.is_multiple_of(checkpoint_every);
    if !already_checkpointed {
        let dir = output.join(format!("gen_{final_gen:0gen_width$}"));
        save_population(&dir, &population, wrap)?;
    }

    Ok(())
}

fn save_population<S: EvolvableStrategy>(
    dir: &Path,
    population: &Population<S>,
    wrap: fn(Vec<S>) -> EvolvableStrategies,
) -> Result<()> {
    let strategies: Vec<S> = population
        .individuals()
        .iter()
        .map(|ind| {
            let strategy = ind.strategy();
            S::from_genes(strategy.label().to_string(), strategy.genes().clone())
        })
        .collect();
    save_strategies_to_directory(dir, &wrap(strategies))?;
    info!(path = %dir.display(), "Saved strategies");
    Ok(())
}

fn create_evolver(common: &CommonArgs, crossover: Crossover, mutation: Mutation) -> Evolver {
    let mut evaluators = Vec::new();
    if common.tournament_weight > 0.0 {
        evaluators.push((
            TournamentFitness::new(Swiss.into(), Freestyle.into()).into(),
            FitnessWeight::new(common.tournament_weight),
        ));
    }
    if common.defense_weight > 0.0 {
        evaluators.push((
            ThreatDefenseFitness.into(),
            FitnessWeight::new(common.defense_weight),
        ));
    }
    if common.minimax_weight > 0.0 {
        evaluators.push((
            MinimaxFitness::new(common.minimax_depth.clone(), common.minimax_scoring_depth).into(),
            FitnessWeight::new(common.minimax_weight),
        ));
    }

    Evolver::new()
        .evaluators(evaluators)
        .generations(common.generations)
        .elitism(common.elitism)
        .crossover(crossover)
        .crossover_rate(common.crossover_rate)
        .mutation(mutation)
        .mutation_rate(common.mutation_rate)
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
        .map(|i| S::random(format!("{i}"), rng))
        .collect()
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

fn extract_cluster_tiny(strategies: EvolvableStrategies) -> Result<Vec<ClusterTiny>> {
    match strategies {
        EvolvableStrategies::ClusterTiny { strategies } => Ok(strategies),
        _ => bail!("Expected ClusterTiny strategies, found different type"),
    }
}

fn extract_cluster_small(strategies: EvolvableStrategies) -> Result<Vec<ClusterSmall>> {
    match strategies {
        EvolvableStrategies::ClusterSmall { strategies } => Ok(strategies),
        _ => bail!("Expected ClusterSmall strategies, found different type"),
    }
}

fn wrap_conv_tiny(strategies: Vec<ConvTiny>) -> EvolvableStrategies {
    EvolvableStrategies::ConvTiny { strategies }
}

fn wrap_conv_small(strategies: Vec<ConvSmall>) -> EvolvableStrategies {
    EvolvableStrategies::ConvSmall { strategies }
}

fn wrap_cluster_tiny(strategies: Vec<ClusterTiny>) -> EvolvableStrategies {
    EvolvableStrategies::ClusterTiny { strategies }
}

fn wrap_cluster_small(strategies: Vec<ClusterSmall>) -> EvolvableStrategies {
    EvolvableStrategies::ClusterSmall { strategies }
}
