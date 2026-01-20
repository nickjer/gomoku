use serde::{Deserialize, Serialize};

use crate::board::Board;
use crate::cache_id::CacheId;
use crate::cache_repository::CacheRepository;
use crate::cluster::fingerprint::FingerprintNN1;
use crate::cluster::select_best_position;
use crate::position_id::PositionId;
use crate::stone::Stone;
use crate::strategy::{EvolvableStrategy, Strategy};

/// A strategy based on NN1 fingerprints with evolvable gene priority.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(from = "NN1Raw", into = "NN1Raw")]
pub struct NN1 {
    label: String,
    /// Fingerprint indices in rank order (index 0 = highest priority).
    ranked_fingerprints: Vec<u32>,
    /// Reverse lookup: `fingerprint_positions[fp.index()]` = position in `ranked_fingerprints`.
    fingerprint_positions: Vec<u32>,
}

#[derive(Serialize, Deserialize)]
struct NN1Raw {
    label: String,
    ranked_fingerprints: Vec<u32>,
}

impl From<NN1Raw> for NN1 {
    fn from(raw: NN1Raw) -> Self {
        Self::new(raw.label, raw.ranked_fingerprints)
    }
}

impl From<NN1> for NN1Raw {
    fn from(strategy: NN1) -> Self {
        Self {
            label: strategy.label,
            ranked_fingerprints: strategy.ranked_fingerprints,
        }
    }
}

impl NN1 {
    /// Creates a new NN1 strategy from fingerprint indices in priority order.
    ///
    /// # Panics
    ///
    /// Panics if `ranked_fingerprints` contains indices outside the valid range.
    #[must_use]
    pub fn new(label: impl Into<String>, ranked_fingerprints: Vec<u32>) -> Self {
        let mut fingerprint_positions = vec![0u32; ranked_fingerprints.len()];
        for (position, &fp_index) in ranked_fingerprints.iter().enumerate() {
            let idx = usize::try_from(fp_index).unwrap();
            fingerprint_positions[idx] = u32::try_from(position).unwrap();
        }
        Self {
            label: label.into(),
            ranked_fingerprints,
            fingerprint_positions,
        }
    }
}

impl EvolvableStrategy for NN1 {
    type Gene = u32;

    fn random_genes(rng: &mut fastrand::Rng) -> Vec<Self::Gene> {
        let len = u32::try_from(FingerprintNN1::all().len()).unwrap();
        let mut genes: Vec<u32> = (0..len).collect();
        rng.shuffle(&mut genes);
        genes
    }

    fn from_genes(label: impl Into<String>, genes: Vec<Self::Gene>) -> Self {
        Self::new(label, genes)
    }

    fn genes(&self) -> &[Self::Gene] {
        &self.ranked_fingerprints
    }
}

impl Strategy for NN1 {
    fn cache_dependencies(&self) -> &[CacheId] {
        &[CacheId::NeighborNN1]
    }

    fn choose_move(
        &self,
        current_stone: Stone,
        board: &Board,
        cache_repo: &CacheRepository,
        rng: &mut fastrand::Rng,
    ) -> PositionId {
        select_best_position(
            board.empty_position_ids(),
            &self.fingerprint_positions,
            |pos| FingerprintNN1::calculate(pos, current_stone, cache_repo).index(),
            rng,
        )
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

    fn fp(player: u8, opponent: u8, empty: u8) -> FingerprintNN1 {
        FingerprintNN1::new(NeighborCounts::new(player, opponent, empty))
    }

    fn all_indices() -> Vec<u32> {
        (0..FingerprintNN1::all().len() as u32).collect()
    }

    struct TestContext {
        cache_repo: CacheRepository,
        rng: fastrand::Rng,
    }

    impl TestContext {
        fn new() -> Self {
            let mut cache_repo = CacheRepository::new();
            cache_repo.activate(CacheId::NeighborNN1);
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
        let strategy = NN1::new("test", all_indices());
        let board = Board::new();

        let result = strategy.choose_move(Stone::Black, &board, &ctx.cache_repo, &mut ctx.rng);

        assert!(board.empty_position_ids().contains(&result));
    }

    #[test]
    fn choose_move_prefers_higher_priority_fingerprint() {
        let mut ctx = TestContext::new();
        let all_empty = fp(0, 0, 4);
        let all_empty_index = all_empty.index();
        let mut priority = vec![all_empty_index];
        priority.extend((0..FingerprintNN1::all().len() as u32).filter(|&i| i != all_empty_index));

        let strategy = NN1::new("test", priority);
        let mut board = Board::new();
        ctx.place(&mut board, PositionId::center(), Stone::Black);

        let result = strategy.choose_move(Stone::White, &board, &ctx.cache_repo, &mut ctx.rng);
        let fingerprint = FingerprintNN1::calculate(result, Stone::White, &ctx.cache_repo);

        assert_eq!(fingerprint, all_empty);
    }

    #[test]
    fn choose_move_prefers_self_neighbors_when_prioritized() {
        let mut ctx = TestContext::new();
        let self_neighbor = fp(1, 0, 3);
        let self_neighbor_index = self_neighbor.index();
        let mut priority = vec![self_neighbor_index];
        priority
            .extend((0..FingerprintNN1::all().len() as u32).filter(|&i| i != self_neighbor_index));

        let strategy = NN1::new("test", priority);
        let mut board = Board::new();
        ctx.place(&mut board, PositionId::center(), Stone::Black);

        let result = strategy.choose_move(Stone::Black, &board, &ctx.cache_repo, &mut ctx.rng);
        let fingerprint = FingerprintNN1::calculate(result, Stone::Black, &ctx.cache_repo);

        assert_eq!(fingerprint, self_neighbor);
    }

    #[test]
    fn choose_move_prefers_edge_mixed_when_prioritized() {
        let mut ctx = TestContext::new();
        let edge_mixed = fp(1, 1, 1);
        let edge_mixed_index = edge_mixed.index();
        let mut priority = vec![edge_mixed_index];
        priority.extend((0..FingerprintNN1::all().len() as u32).filter(|&i| i != edge_mixed_index));

        let strategy = NN1::new("test", priority);
        let mut board = Board::new();

        // Edge position (1,0) has 3 neighbors: (0,0), (2,0), (1,1)
        // Place black at (0,0), white at (2,0)
        ctx.place(&mut board, pos(0, 0), Stone::Black);
        ctx.place(&mut board, pos(2, 0), Stone::White);

        let result = strategy.choose_move(Stone::Black, &board, &ctx.cache_repo, &mut ctx.rng);
        let fingerprint = FingerprintNN1::calculate(result, Stone::Black, &ctx.cache_repo);

        assert_eq!(result, pos(1, 0));
        assert_eq!(fingerprint, edge_mixed);
    }

    #[test]
    fn label_returns_provided_label() {
        let strategy = NN1::new("my_nn1_strategy", all_indices());

        assert_eq!(strategy.label(), "my_nn1_strategy");
    }

    #[test]
    fn genes_returns_ranked_fingerprints() {
        let priority = all_indices();
        let strategy = NN1::new("test", priority.clone());

        assert_eq!(strategy.genes(), priority.as_slice());
    }

    #[test]
    fn random_genes_has_correct_size() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = NN1::random("test", &mut rng);

        assert_eq!(strategy.genes().len(), FingerprintNN1::all().len());
    }

    #[test]
    fn random_genes_contains_all_indices() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = NN1::random("test", &mut rng);

        let genes_set: HashSet<_> = strategy.genes().iter().copied().collect();
        let all_set: HashSet<_> = (0..FingerprintNN1::all().len() as u32).collect();

        assert_eq!(genes_set, all_set);
    }

    #[test]
    fn random_genes_is_shuffled() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = NN1::random("test", &mut rng);

        let sequential: Vec<u32> = (0..FingerprintNN1::all().len() as u32).collect();
        assert_ne!(strategy.genes(), sequential.as_slice());
    }

    #[test]
    fn random_genes_deterministic_with_same_seed() {
        let strategy1 = NN1::random("test", &mut fastrand::Rng::with_seed(42));
        let strategy2 = NN1::random("test", &mut fastrand::Rng::with_seed(42));

        assert_eq!(strategy1.genes(), strategy2.genes());
    }

    #[test]
    fn random_genes_different_with_different_seeds() {
        let strategy1 = NN1::random("test", &mut fastrand::Rng::with_seed(1));
        let strategy2 = NN1::random("test", &mut fastrand::Rng::with_seed(2));

        assert_ne!(strategy1.genes(), strategy2.genes());
    }
}
