use crate::board::Board;
use crate::cache_id::CacheId;
use crate::cache_repository::CacheRepository;
use crate::cluster::fingerprint::FingerprintNN3;
use crate::cluster::select_best_position;
use crate::gene::Gene;
use crate::position_id::PositionId;
use crate::stone::Stone;
use crate::strategy::{EvolvableStrategy, Strategy};

const CACHE_DEPENDENCIES: &[CacheId] = &[CacheId::FingerprintNN3Index];

/// A strategy based on NN3 fingerprints with evolvable gene priority.
#[derive(Debug, Clone)]
pub struct NN3 {
    label: String,
    /// Fingerprint indices in rank order (index 0 = highest priority).
    ranked_fingerprints: Vec<Gene>,
    /// Reverse lookup: `fingerprint_positions[fp.index()]` = position in `ranked_fingerprints`.
    fingerprint_positions: Vec<usize>,
}

impl NN3 {
    /// Creates a new NN3 strategy from fingerprint indices in priority order.
    #[must_use]
    pub fn new(label: impl Into<String>, ranked_fingerprints: Vec<Gene>) -> Self {
        let mut fingerprint_positions = vec![0usize; ranked_fingerprints.len()];
        for (position, &gene) in ranked_fingerprints.iter().enumerate() {
            fingerprint_positions[gene.index()] = position;
        }
        Self {
            label: label.into(),
            ranked_fingerprints,
            fingerprint_positions,
        }
    }
}

impl EvolvableStrategy for NN3 {
    type Genes = Vec<Gene>;

    fn random(label: impl Into<String>, rng: &mut fastrand::Rng) -> Self {
        let len = FingerprintNN3::all().len();
        let mut genes: Vec<Gene> = (0..len).map(Gene::new).collect();
        rng.shuffle(&mut genes);
        Self::new(label, genes)
    }

    fn genes(&self) -> &Self::Genes {
        &self.ranked_fingerprints
    }

    fn from_genes(label: impl Into<String>, genes: Self::Genes) -> Self {
        Self::new(label, genes)
    }
}

impl Strategy for NN3 {
    fn cache_dependencies(&self) -> &[CacheId] {
        CACHE_DEPENDENCIES
    }

    fn choose_move(
        &self,
        current_stone: Stone,
        board: &Board,
        cache_repo: &CacheRepository,
        rng: &mut fastrand::Rng,
    ) -> PositionId {
        let index_cache = cache_repo
            .fingerprint_nn3_index()
            .expect("NN3 requires fingerprint_nn3_index cache");

        select_best_position(
            board.empty_position_ids(),
            &self.fingerprint_positions,
            |pos| index_cache.get(pos, current_stone),
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
    use crate::gene::Gene;
    use crate::position::Position;
    use std::collections::HashSet;

    fn pos(row: u8, col: u8) -> PositionId {
        PositionId::from_position(Position::new(row, col))
    }

    fn counts(player: u8, opponent: u8, empty: u8) -> NeighborCounts {
        NeighborCounts::new(player, opponent, empty)
    }

    fn fp3(nn1: NeighborCounts, nn2: NeighborCounts, nn3: NeighborCounts) -> FingerprintNN3 {
        FingerprintNN3::new(nn1, nn2, nn3)
    }

    fn all_genes() -> Vec<Gene> {
        (0..FingerprintNN3::all().len()).map(Gene::new).collect()
    }

    fn genes_with_first(first: FingerprintNN3) -> Vec<Gene> {
        let first_idx = first.index();
        let mut priority = vec![Gene::new(first_idx)];
        priority.extend(
            (0..FingerprintNN3::all().len())
                .filter(|&i| i != first_idx)
                .map(Gene::new),
        );
        priority
    }

    struct TestContext {
        cache_repo: CacheRepository,
        rng: fastrand::Rng,
    }

    impl TestContext {
        fn new() -> Self {
            let mut cache_repo = CacheRepository::new();
            cache_repo.activate(CacheId::FingerprintNN3Index);
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
        let strategy = NN3::new("test", all_genes());
        let board = Board::new();

        let result = strategy.choose_move(Stone::Black, &board, &ctx.cache_repo, &mut ctx.rng);

        assert!(board.empty_position_ids().contains(&result));
    }

    #[test]
    fn choose_move_prefers_nn3_pattern_when_prioritized() {
        //       5   6   7   8   9
        //     +---+---+---+---+---+
        //  5  |   |   |   |   |   |
        //     +---+---+---+---+---+
        //  6  |   | B | W |   |   |
        //     +---+---+---+---+---+
        //  7  |   | B |(*)| B |   |
        //     +---+---+---+---+---+
        //  8  |   |   | W | W |   |
        //     +---+---+---+---+---+
        //  9  |   |   |   |   |   |
        //     +---+---+---+---+---+
        //
        // (*) = center (7,7), playing as Black
        // NN1: (6,7)=W, (8,7)=W, (7,6)=B, (7,8)=B → self=2, opp=2, empty=0
        // NN2: (6,6)=B, (8,8)=W                   → self=1, opp=1, empty=2
        // NN3: all empty                          → self=0, opp=0, empty=4
        let mut ctx = TestContext::new();
        let specific_fp = fp3(counts(2, 2, 0), counts(1, 1, 2), counts(0, 0, 4));
        let strategy = NN3::new("test", genes_with_first(specific_fp));
        let mut board = Board::new();

        // NN1 neighbors
        ctx.place(&mut board, pos(6, 7), Stone::White); // up
        ctx.place(&mut board, pos(8, 7), Stone::White); // down
        ctx.place(&mut board, pos(7, 6), Stone::Black); // left
        ctx.place(&mut board, pos(7, 8), Stone::Black); // right

        // NN2 neighbors
        ctx.place(&mut board, pos(6, 6), Stone::Black); // up-left
        ctx.place(&mut board, pos(8, 8), Stone::White); // down-right

        let result = strategy.choose_move(Stone::Black, &board, &ctx.cache_repo, &mut ctx.rng);

        assert_eq!(result, pos(7, 7)); // center
    }

    #[test]
    fn label_returns_provided_label() {
        let strategy = NN3::new("my_nn3_strategy", all_genes());

        assert_eq!(strategy.label(), "my_nn3_strategy");
    }

    #[test]
    fn genes_returns_ranked_fingerprints() {
        let priority = all_genes();
        let strategy = NN3::new("test", priority.clone());

        assert_eq!(strategy.genes(), priority.as_slice());
    }

    #[test]
    fn random_genes_has_correct_size() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = NN3::random("test", &mut rng);

        assert_eq!(strategy.genes().len(), FingerprintNN3::all().len());
    }

    #[test]
    fn random_genes_contains_all_indices() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = NN3::random("test", &mut rng);

        let genes_set: HashSet<_> = strategy.genes().iter().copied().collect();
        let all_set: HashSet<_> = all_genes().into_iter().collect();

        assert_eq!(genes_set, all_set);
    }

    #[test]
    fn random_genes_is_shuffled() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = NN3::random("test", &mut rng);

        let sequential = all_genes();
        assert_ne!(strategy.genes(), sequential.as_slice());
    }

    #[test]
    fn random_genes_deterministic_with_same_seed() {
        let strategy1 = NN3::random("test", &mut fastrand::Rng::with_seed(42));
        let strategy2 = NN3::random("test", &mut fastrand::Rng::with_seed(42));

        assert_eq!(strategy1.genes(), strategy2.genes());
    }

    #[test]
    fn random_genes_different_with_different_seeds() {
        let strategy1 = NN3::random("test", &mut fastrand::Rng::with_seed(1));
        let strategy2 = NN3::random("test", &mut fastrand::Rng::with_seed(2));

        assert_ne!(strategy1.genes(), strategy2.genes());
    }
}
