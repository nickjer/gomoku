use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};
use clap::Args;
use tracing::{info, warn};

use super::saved_strategy::{
    SavedStrategy, StrategyKind, read_strategy_directory, write_strategy_directory,
};
use super::{create_rng, setup_logging};
use crate::evolution::crossover::Crossover;
use crate::evolution::fitness_weight::FitnessWeight;
use crate::evolution::mutation::Mutation;
use crate::evolution::selection::{Tournament as TournamentSelection, TournamentMode};
use crate::evolution::{
    Evolver, MinimaxFitness, Population, ThreatDefenseFitness, TournamentFitness,
};
use crate::game::{Freestyle, Game, RandomOpening};
use crate::nn::{ClusterSmall, ClusterTiny, ConvSmall, ConvTiny};
use crate::strategy::{EvolvableStrategy, Strategy};
use crate::tournament::Swiss;

/// Arguments for evolving strategies.
#[derive(Debug, Args)]
pub struct EvolveArgs {
    /// Kind of strategy to evolve (read from the --input files when omitted)
    #[arg(value_enum, required_unless_present = "input")]
    pub kind: Option<StrategyKind>,

    /// Load strategies from directory (skips random generation)
    #[arg(short, long, value_name = "DIR")]
    pub input: Option<PathBuf>,

    /// Number of random strategies to generate
    #[arg(
        short,
        long,
        value_name = "N",
        required_unless_present = "input",
        conflicts_with = "input"
    )]
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

    /// How tournament selection samples the individuals it compares
    #[arg(long, value_enum, default_value_t = TournamentMode::WithReplacement)]
    pub selection: TournamentMode,

    /// Crossover probability (0.0 to 1.0)
    #[arg(long, default_value = "0.8")]
    pub crossover_rate: f64,

    /// Mutation probability (0.0 to 1.0)
    #[arg(long, default_value = "0.1")]
    pub mutation_rate: f64,

    /// Gaussian mutation sigma
    #[arg(long, default_value = "0.01")]
    pub sigma: f32,

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

    /// Pre-place N random stones before each game (0 = start from empty board)
    #[arg(long, default_value = "0")]
    pub opening_moves: u32,

    /// Save a checkpoint every N generations (0 to disable)
    #[arg(long, default_value = "0", value_name = "N")]
    pub checkpoint_every: u32,

    /// Log level (error, warn, info, debug, trace)
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

    let loaded = match &args.input {
        Some(dir) => {
            let loaded = read_strategy_directory(dir)?;
            info!(path = %dir.display(), count = loaded.len(), "Loaded strategies from directory");
            Some(loaded)
        }
        None => None,
    };

    // An explicit kind wins; otherwise the loaded files say what they hold.
    let kind = match (args.kind, &loaded) {
        (Some(kind), _) => kind,
        (None, Some(loaded)) => StrategyKind::from(&loaded[0]),
        (None, None) => unreachable!("clap requires a kind or an --input directory"),
    };
    info!(?kind, "Starting evolution");

    match kind {
        StrategyKind::ConvTiny => run_evolution::<ConvTiny>(args, kind, loaded),
        StrategyKind::ConvSmall => run_evolution::<ConvSmall>(args, kind, loaded),
        StrategyKind::ClusterTiny => run_evolution::<ClusterTiny>(args, kind, loaded),
        StrategyKind::ClusterSmall => run_evolution::<ClusterSmall>(args, kind, loaded),
    }
}

fn run_evolution<S>(
    args: &EvolveArgs,
    kind: StrategyKind,
    loaded: Option<Vec<SavedStrategy>>,
) -> Result<()>
where
    S: EvolvableStrategy + Clone + Into<SavedStrategy> + TryFrom<SavedStrategy>,
{
    let mut rng = create_rng(args.seed);

    let strategies: Vec<S> = if let Some(loaded) = loaded {
        loaded
            .into_iter()
            .map(|saved| {
                let found = StrategyKind::from(&saved);
                let label = saved.label().to_owned();
                S::try_from(saved).map_err(|_| anyhow!("{label} holds a {found:?}, not a {kind:?}"))
            })
            .collect::<Result<_>>()?
    } else {
        let population = args
            .population
            .expect("clap requires --population without --input");
        info!(count = population, "Generating random strategies");
        (0..population)
            .map(|i| S::random(format!("{i}"), &mut rng))
            .collect()
    };

    let evolver = create_evolver(args);

    let output = &args.output;
    let generations = args.generations;
    let checkpoint_every = args.checkpoint_every;
    let gen_width = generations.max(1).to_string().len();

    let population: Population<S> = evolver.evolve(strategies, &mut rng, |population| {
        let generation = population.generation();
        if checkpoint_every > 0 && generation % checkpoint_every == 0 {
            let dir = output.join(format!("gen_{generation:0gen_width$}"));
            if let Err(err) = save_population(&dir, population) {
                warn!(%err, "Failed to save checkpoint");
            }
        }
    });
    info!(generation = population.generation(), "Evolution complete");

    let final_gen = population.generation();
    let already_checkpointed = checkpoint_every > 0 && final_gen.is_multiple_of(checkpoint_every);
    if !already_checkpointed {
        let dir = output.join(format!("gen_{final_gen:0gen_width$}"));
        save_population(&dir, &population)?;
    }

    Ok(())
}

fn save_population<S: EvolvableStrategy + Clone + Into<SavedStrategy>>(
    dir: &Path,
    population: &Population<S>,
) -> Result<()> {
    let saved: Vec<SavedStrategy> = population
        .individuals()
        .iter()
        .map(|individual| individual.strategy().clone().into())
        .collect();
    write_strategy_directory(dir, &saved)?;
    info!(path = %dir.display(), "Saved strategies");
    Ok(())
}

fn create_evolver(args: &EvolveArgs) -> Evolver {
    let game: Game = if args.opening_moves > 0 {
        RandomOpening {
            moves: args.opening_moves,
        }
        .into()
    } else {
        Freestyle.into()
    };

    let mut evaluators = Vec::new();
    if args.tournament_weight > 0.0 {
        evaluators.push((
            TournamentFitness::new(Swiss.into(), game.clone()).into(),
            FitnessWeight::new(args.tournament_weight),
        ));
    }
    if args.defense_weight > 0.0 {
        evaluators.push((
            ThreatDefenseFitness.into(),
            FitnessWeight::new(args.defense_weight),
        ));
    }
    if args.minimax_weight > 0.0 {
        evaluators.push((
            MinimaxFitness::new(
                game.clone(),
                args.minimax_depth.clone(),
                args.minimax_scoring_depth,
            )
            .into(),
            FitnessWeight::new(args.minimax_weight),
        ));
    }

    Evolver::new()
        .evaluators(evaluators)
        .generations(args.generations)
        .elitism(args.elitism)
        .selection(TournamentSelection::new(3, args.selection).into())
        .crossover(Crossover::Uniform)
        .crossover_rate(args.crossover_rate)
        .mutation(Mutation::Gaussian { sigma: args.sigma })
        .mutation_rate(args.mutation_rate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct TestCli {
        #[command(flatten)]
        args: EvolveArgs,
    }

    #[test]
    fn kind_and_population_are_required_without_input() {
        assert!(TestCli::try_parse_from(["gomoku", "-p", "4", "-o", "out"]).is_err());
        assert!(TestCli::try_parse_from(["gomoku", "conv-tiny", "-o", "out"]).is_err());

        let cli = TestCli::try_parse_from(["gomoku", "conv-tiny", "-p", "4", "-o", "out"]).unwrap();

        assert_eq!(cli.args.kind, Some(StrategyKind::ConvTiny));
        assert_eq!(cli.args.population, Some(4));
    }

    #[test]
    fn kind_is_optional_with_input() {
        let cli = TestCli::try_parse_from(["gomoku", "-i", "dir", "-o", "out"]).unwrap();

        assert_eq!(cli.args.kind, None);
        assert_eq!(cli.args.input, Some(PathBuf::from("dir")));
    }

    #[test]
    fn population_conflicts_with_input() {
        assert!(TestCli::try_parse_from(["gomoku", "-i", "dir", "-p", "4", "-o", "out"]).is_err());
    }

    #[test]
    fn selection_defaults_to_with_replacement() {
        let cli = TestCli::try_parse_from(["gomoku", "-i", "dir", "-o", "out"]).unwrap();
        assert_eq!(cli.args.selection, TournamentMode::WithReplacement);

        let cli = TestCli::try_parse_from([
            "gomoku",
            "-i",
            "dir",
            "-o",
            "out",
            "--selection",
            "without-replacement",
        ])
        .unwrap();
        assert_eq!(cli.args.selection, TournamentMode::WithoutReplacement);
    }
}
