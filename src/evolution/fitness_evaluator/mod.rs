mod minimax;
mod scored_board;
mod threat_defense;
mod tournament;

pub use minimax::MinimaxFitness;
pub use scored_board::ScoredBoardFitness;
pub use threat_defense::ThreatDefenseFitness;
pub use tournament::TournamentFitness;

use enum_dispatch::enum_dispatch;

use crate::strategy::EvolvableStrategy;

use super::fitness_score::FitnessScore;

/// Computes fitness scores for a population of strategies.
#[enum_dispatch]
pub trait EvaluateFitness {
    fn evaluate<S: EvolvableStrategy>(
        &self,
        strategies: &[S],
        rng: &mut fastrand::Rng,
    ) -> Vec<FitnessScore>;
}

/// Polymorphic dispatch over all fitness evaluator variants.
#[enum_dispatch(EvaluateFitness)]
#[derive(strum::Display)]
#[allow(clippy::enum_variant_names)]
pub enum FitnessEvaluator {
    TournamentFitness,
    ThreatDefenseFitness,
    MinimaxFitness,
    ScoredBoardFitness,
}
