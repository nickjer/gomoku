use tracing::instrument;

use crate::evolution::fitness_score::FitnessScore;
use crate::strategy::Strategy;
use crate::threat::{generate_threat_scenarios, test_defense};

use super::EvaluateFitness;

/// Evaluates fitness by testing how well each strategy blocks pre-generated threats.
pub struct ThreatDefenseFitness;

impl EvaluateFitness for ThreatDefenseFitness {
    #[instrument(name = "ThreatDefenseFitness", skip_all)]
    fn evaluate<S: Strategy>(
        &self,
        strategies: &[S],
        rng: &mut fastrand::Rng,
    ) -> Vec<FitnessScore> {
        let scenarios = generate_threat_scenarios(rng);
        strategies
            .iter()
            .map(|strategy| FitnessScore::new(test_defense(strategy, &scenarios, rng)))
            .collect()
    }
}
