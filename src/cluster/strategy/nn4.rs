use serde::{Deserialize, Serialize};

use crate::board::Board;
use crate::cache_id::CacheId;
use crate::cache_repository::CacheRepository;
use crate::cluster::fingerprint::FingerprintNN4;
use crate::cluster::select_best_position;
use crate::position_id::PositionId;
use crate::stone::Stone;
use crate::strategy::{EvolvableStrategy, Strategy};

/// A strategy based on NN4 fingerprints with evolvable gene priority.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(from = "NN4Raw", into = "NN4Raw")]
pub struct NN4 {
    label: String,
    /// Fingerprint indices in rank order (index 0 = highest priority).
    ranked_fingerprints: Vec<u32>,
    /// Reverse lookup: `fingerprint_positions[fp.index()]` = position in `ranked_fingerprints`.
    fingerprint_positions: Vec<u32>,
}

#[derive(Serialize, Deserialize)]
struct NN4Raw {
    label: String,
    ranked_fingerprints: Vec<u32>,
}

impl From<NN4Raw> for NN4 {
    fn from(raw: NN4Raw) -> Self {
        Self::new(raw.label, raw.ranked_fingerprints)
    }
}

impl From<NN4> for NN4Raw {
    fn from(strategy: NN4) -> Self {
        Self {
            label: strategy.label,
            ranked_fingerprints: strategy.ranked_fingerprints,
        }
    }
}

impl NN4 {
    /// Creates a new NN4 strategy from fingerprint indices in priority order.
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

impl EvolvableStrategy for NN4 {
    type Gene = u32;

    fn random_genes(rng: &mut fastrand::Rng) -> Vec<Self::Gene> {
        let len = u32::try_from(FingerprintNN4::all().len()).unwrap();
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

impl Strategy for NN4 {
    fn cache_dependencies(&self) -> &[CacheId] {
        &[
            CacheId::NeighborNN1,
            CacheId::NeighborNN2,
            CacheId::NeighborNN3,
            CacheId::NeighborNN4,
        ]
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
            |pos| FingerprintNN4::calculate(pos, current_stone, cache_repo).index(),
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

    fn counts(player: u8, opponent: u8, empty: u8) -> NeighborCounts {
        NeighborCounts::new(player, opponent, empty)
    }

    fn fp4(
        nn1: NeighborCounts,
        nn2: NeighborCounts,
        nn3: NeighborCounts,
        nn4: NeighborCounts,
    ) -> FingerprintNN4 {
        FingerprintNN4::new(nn1, nn2, nn3, nn4)
    }

    fn all_indices() -> Vec<u32> {
        (0..FingerprintNN4::all().len() as u32).collect()
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
            cache_repo.activate(CacheId::NeighborNN3);
            cache_repo.activate(CacheId::NeighborNN4);
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
        let strategy = NN4::new("test", all_indices());
        let board = Board::new();

        let result = strategy.choose_move(Stone::Black, &board, &ctx.cache_repo, &mut ctx.rng);

        assert!(board.empty_position_ids().contains(&result));
    }

    #[test]
    fn choose_move_prefers_nn4_pattern_when_prioritized() {
        //       4   5   6   7   8   9  10
        //     +---+---+---+---+---+---+---+
        //  4  |   |   |   |   |   |   |   |
        //     +---+---+---+---+---+---+---+
        //  5  |   | W | B | W | B |   |   |
        //     +---+---+---+---+---+---+---+
        //  6  |   | B | B | W |   |   |   |
        //     +---+---+---+---+---+---+---+
        //  7  |   | W | B |(*)| B |   |   |
        //     +---+---+---+---+---+---+---+
        //  8  |   |   | W | B |   |   |   |
        //     +---+---+---+---+---+---+---+
        //  9  |   |   |   |   |   |   |   |
        //     +---+---+---+---+---+---+---+
        //
        // (*) = center (7,7), playing as Black
        // NN1: (6,7)=W, (8,7)=B, (7,6)=B, (7,8)=B → self=3, opp=1, empty=0
        // NN2: (6,6)=B, (8,8)=empty, (6,8)=empty, (8,6)=W → self=1, opp=1, empty=2
        // NN3: (5,7)=W, (9,7)=empty, (7,5)=W, (7,9)=empty → self=0, opp=2, empty=2
        // NN4: (5,6)=B, (5,8)=B, (6,5)=B, (6,9)=empty, (8,5)=empty, (8,9)=empty,
        //      (9,6)=W, (9,8)=empty → self=3, opp=1, empty=4
        let mut ctx = TestContext::new();
        let specific_fp = fp4(
            counts(3, 1, 0),
            counts(1, 1, 2),
            counts(0, 2, 2),
            counts(3, 1, 4),
        );
        let specific_fp_index = specific_fp.index();

        // Put the specific fingerprint first (highest priority), then all others
        let mut priority = vec![specific_fp_index];
        priority
            .extend((0..FingerprintNN4::all().len() as u32).filter(|&i| i != specific_fp_index));

        let strategy = NN4::new("test", priority);
        let mut board = Board::new();

        // NN1 neighbors
        ctx.place(&mut board, pos(6, 7), Stone::White);
        ctx.place(&mut board, pos(8, 7), Stone::Black);
        ctx.place(&mut board, pos(7, 6), Stone::Black);
        ctx.place(&mut board, pos(7, 8), Stone::Black);

        // NN2 neighbors
        ctx.place(&mut board, pos(6, 6), Stone::Black);
        ctx.place(&mut board, pos(8, 6), Stone::White);

        // NN3 neighbors
        ctx.place(&mut board, pos(5, 7), Stone::White);
        ctx.place(&mut board, pos(7, 5), Stone::White);

        // NN4 neighbors (knight's move: (±2,±1) and (±1,±2) from center)
        ctx.place(&mut board, pos(5, 6), Stone::Black);
        ctx.place(&mut board, pos(5, 8), Stone::Black);
        ctx.place(&mut board, pos(6, 5), Stone::Black);
        ctx.place(&mut board, pos(9, 6), Stone::White);

        let result = strategy.choose_move(Stone::Black, &board, &ctx.cache_repo, &mut ctx.rng);
        let fingerprint = FingerprintNN4::calculate(result, Stone::Black, &ctx.cache_repo);

        assert_eq!(result, pos(7, 7)); // center
        assert_eq!(fingerprint, specific_fp);
    }

    #[test]
    fn label_returns_provided_label() {
        let strategy = NN4::new("my_nn4_strategy", all_indices());

        assert_eq!(strategy.label(), "my_nn4_strategy");
    }

    #[test]
    fn genes_returns_ranked_fingerprints() {
        let priority = all_indices();
        let strategy = NN4::new("test", priority.clone());

        assert_eq!(strategy.genes(), priority.as_slice());
    }

    #[test]
    fn random_genes_has_correct_size() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = NN4::random("test", &mut rng);

        assert_eq!(strategy.genes().len(), FingerprintNN4::all().len());
    }

    #[test]
    fn random_genes_contains_all_indices() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = NN4::random("test", &mut rng);

        let genes_set: HashSet<_> = strategy.genes().iter().copied().collect();
        let all_set: HashSet<_> = (0..FingerprintNN4::all().len() as u32).collect();

        assert_eq!(genes_set, all_set);
    }

    #[test]
    fn random_genes_is_shuffled() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = NN4::random("test", &mut rng);

        let sequential: Vec<u32> = (0..FingerprintNN4::all().len() as u32).collect();
        assert_ne!(strategy.genes(), sequential.as_slice());
    }

    #[test]
    fn random_genes_deterministic_with_same_seed() {
        let strategy1 = NN4::random("test", &mut fastrand::Rng::with_seed(42));
        let strategy2 = NN4::random("test", &mut fastrand::Rng::with_seed(42));

        assert_eq!(strategy1.genes(), strategy2.genes());
    }

    #[test]
    fn random_genes_different_with_different_seeds() {
        let strategy1 = NN4::random("test", &mut fastrand::Rng::with_seed(1));
        let strategy2 = NN4::random("test", &mut fastrand::Rng::with_seed(2));

        assert_ne!(strategy1.genes(), strategy2.genes());
    }
}
