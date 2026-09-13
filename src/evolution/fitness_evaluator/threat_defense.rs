use tracing::instrument;

use crate::evolution::fitness_score::FitnessScore;
use crate::strategy::EvolvableStrategy;
use crate::threat::{count_correct_moves, generate_threat_scenarios};

use super::EvaluateFitness;

/// Evaluates fitness by testing how well each strategy blocks pre-generated threats.
pub struct ThreatDefenseFitness;

impl EvaluateFitness for ThreatDefenseFitness {
    #[instrument(name = "ThreatDefenseFitness", skip_all)]
    fn evaluate<S: EvolvableStrategy>(
        &self,
        strategies: &[S],
        rng: &mut fastrand::Rng,
    ) -> Vec<FitnessScore> {
        let scenarios = generate_threat_scenarios(rng);
        strategies
            .iter()
            .map(|strategy| FitnessScore::new(count_correct_moves(strategy, &scenarios, rng)))
            .collect()
    }
}
