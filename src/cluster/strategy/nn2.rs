use serde::{Deserialize, Serialize};

use crate::board::Board;
use crate::cache_id::CacheId;
use crate::cache_repository::CacheRepository;
use crate::cluster::BestPositionSelector;
use crate::cluster::fingerprint::FingerprintNN2;
use crate::position_id::PositionId;
use crate::stone::Stone;
use crate::strategy::{EvolvableStrategy, Strategy};

/// A strategy based on NN2 fingerprints with evolvable gene priority.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(from = "NN2Raw", into = "NN2Raw")]
pub struct NN2 {
    label: String,
    genes: Vec<FingerprintNN2>,
    selector: BestPositionSelector<FingerprintNN2>,
}

#[derive(Serialize, Deserialize)]
struct NN2Raw {
    label: String,
    genes: Vec<FingerprintNN2>,
}

impl From<NN2Raw> for NN2 {
    fn from(raw: NN2Raw) -> Self {
        Self::new(raw.label, raw.genes)
    }
}

impl From<NN2> for NN2Raw {
    fn from(strategy: NN2) -> Self {
        Self {
            label: strategy.label,
            genes: strategy.genes,
        }
    }
}

impl NN2 {
    #[must_use]
    pub fn new(label: impl Into<String>, genes: Vec<FingerprintNN2>) -> Self {
        let selector = BestPositionSelector::new(&genes);
        Self {
            label: label.into(),
            genes,
            selector,
        }
    }
}

impl EvolvableStrategy for NN2 {
    type Gene = FingerprintNN2;

    fn random_genes(rng: &mut fastrand::Rng) -> Vec<Self::Gene> {
        let mut genes = FingerprintNN2::all().to_vec();
        rng.shuffle(&mut genes);
        genes
    }

    fn from_genes(label: impl Into<String>, genes: Vec<Self::Gene>) -> Self {
        Self::new(label, genes)
    }

    fn genes(&self) -> &[Self::Gene] {
        &self.genes
    }
}

impl Strategy for NN2 {
    fn cache_dependencies(&self) -> &[CacheId] {
        &[CacheId::NeighborNN1, CacheId::NeighborNN2]
    }

    fn choose_move(
        &self,
        current_stone: Stone,
        board: &Board,
        cache_repo: &CacheRepository,
        rng: &mut fastrand::Rng,
    ) -> PositionId {
        let position_fingerprints: Vec<_> = board
            .empty_position_ids()
            .iter()
            .map(|&pos| {
                (
                    pos,
                    FingerprintNN2::calculate(pos, current_stone, cache_repo),
                )
            })
            .collect();

        self.selector.select(&position_fingerprints, rng)
    }

    fn label(&self) -> &str {
        &self.label
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::NeighborCounts;
    use crate::position::Position;
    use std::collections::HashSet;

    fn pos(row: u8, col: u8) -> PositionId {
        PositionId::from_position(Position::new(row, col))
    }

    fn counts(player: u8, opponent: u8, empty: u8) -> NeighborCounts {
        NeighborCounts::new(player, opponent, empty)
    }

    fn fp2(nn1: NeighborCounts, nn2: NeighborCounts) -> FingerprintNN2 {
        FingerprintNN2::new(nn1, nn2)
    }

    struct TestContext {
        cache_repo: CacheRepository,
        rng: fastrand::Rng,
    }

    impl TestContext {
        fn new() -> Self {
            let mut cache_repo = CacheRepository::new();
            cache_repo.activate(CacheId::NeighborNN1);
            cache_repo.activate(CacheId::NeighborNN2);
            Self {
                cache_repo,
                rng: fastrand::Rng::new(),
            }
        }

        fn place(&mut self, board: &mut Board, position_id: PositionId, stone: Stone) {
            board.place(position_id, stone).unwrap();
            self.cache_repo.place(position_id, stone);
        }
    }

    #[test]
    fn choose_move_returns_empty_position() {
        let mut ctx = TestContext::new();
        let strategy = NN2::new("test", FingerprintNN2::all().to_vec());
        let board = Board::new();

        let result = strategy.choose_move(Stone::Black, &board, &ctx.cache_repo, &mut ctx.rng);

        assert!(board.empty_position_ids().contains(&result));
    }

    #[test]
    fn choose_move_prefers_edge_mixed_when_prioritized() {
        // Board setup (top-left corner):
        //
        //        col 0   col 1
        //      +-------+-------+
        // row 0|   B   |   B   |
        //      +-------+-------+
        // row 1|  (*)  |       |   (*) = position being evaluated
        //      +-------+-------+
        // row 2|   W   |   W   |
        //      +-------+-------+
        //
        // Position (1,0):
        //   NN1: (0,0)=B, (2,0)=W, (1,1)=empty → self=1, opp=1, empty=1
        //   NN2: (0,1)=B, (2,1)=W             → self=1, opp=1, empty=0
        let mut ctx = TestContext::new();
        let edge_mixed = fp2(counts(1, 1, 1), counts(1, 1, 0));
        let mut priority = vec![edge_mixed];
        priority.extend(FingerprintNN2::all().iter().filter(|&&f| f != edge_mixed));

        let strategy = NN2::new("test", priority);
        let mut board = Board::new();

        ctx.place(&mut board, pos(0, 0), Stone::Black);
        ctx.place(&mut board, pos(2, 0), Stone::White);
        ctx.place(&mut board, pos(0, 1), Stone::Black);
        ctx.place(&mut board, pos(2, 1), Stone::White);

        let result = strategy.choose_move(Stone::Black, &board, &ctx.cache_repo, &mut ctx.rng);
        let fingerprint = FingerprintNN2::calculate(result, Stone::Black, &ctx.cache_repo);

        assert_eq!(result, pos(1, 0));
        assert_eq!(fingerprint, edge_mixed);
    }

    #[test]
    #[should_panic(expected = "Unknown fingerprint in priority map")]
    fn choose_move_panics_on_unknown_fingerprint() {
        let mut ctx = TestContext::new();
        let incomplete_priority = vec![fp2(counts(0, 0, 4), counts(0, 0, 4))];

        let strategy = NN2::new("test", incomplete_priority);
        let mut board = Board::new();
        ctx.place(&mut board, PositionId::center(), Stone::Black);

        strategy.choose_move(Stone::White, &board, &ctx.cache_repo, &mut ctx.rng);
    }

    #[test]
    fn label_returns_provided_label() {
        let strategy = NN2::new("my_nn2_strategy", FingerprintNN2::all().to_vec());

        assert_eq!(strategy.label(), "my_nn2_strategy");
    }

    #[test]
    fn genes_returns_fingerprint_priority() {
        let priority: Vec<_> = FingerprintNN2::all().to_vec();
        let strategy = NN2::new("test", priority.clone());

        assert_eq!(strategy.genes(), priority.as_slice());
    }

    #[test]
    fn random_genes_has_correct_size() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = NN2::random("test", &mut rng);

        assert_eq!(strategy.genes().len(), FingerprintNN2::all().len());
    }

    #[test]
    fn random_genes_contains_all_fingerprints() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = NN2::random("test", &mut rng);

        let genes_set: HashSet<_> = strategy.genes().iter().collect();
        let all_set: HashSet<_> = FingerprintNN2::all().iter().collect();

        assert_eq!(genes_set, all_set);
    }

    #[test]
    fn random_genes_is_shuffled() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = NN2::random("test", &mut rng);

        assert_ne!(strategy.genes(), FingerprintNN2::all());
    }

    #[test]
    fn random_genes_deterministic_with_same_seed() {
        let strategy1 = NN2::random("test", &mut fastrand::Rng::with_seed(42));
        let strategy2 = NN2::random("test", &mut fastrand::Rng::with_seed(42));

        assert_eq!(strategy1.genes(), strategy2.genes());
    }

    #[test]
    fn random_genes_different_with_different_seeds() {
        let strategy1 = NN2::random("test", &mut fastrand::Rng::with_seed(1));
        let strategy2 = NN2::random("test", &mut fastrand::Rng::with_seed(2));

        assert_ne!(strategy1.genes(), strategy2.genes());
    }
}
