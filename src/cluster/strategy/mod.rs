use std::marker::PhantomData;

use crate::board::Board;
use crate::cache_id::CacheId;
use crate::cache_repository::CacheRepository;
use crate::cluster::FingerprintIndexCache;
use crate::cluster::fingerprint::{
    Fingerprint, FingerprintNN1, FingerprintNN2, FingerprintNN3, FingerprintNN4,
};
use crate::cluster::select_best_position;
use crate::gene::Gene;
use crate::position_id::PositionId;
use crate::stone::Stone;
use crate::strategy::{EvolvableStrategy, Strategy};

/// A strategy based on fingerprint priority ordering.
#[derive(Debug, Clone)]
pub struct FingerprintStrategy<F: Fingerprint> {
    label: String,
    ranked_fingerprints: Vec<Gene>,
    fingerprint_positions: Vec<usize>,
    _marker: PhantomData<F>,
}

impl<F: Fingerprint> FingerprintStrategy<F> {
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
            _marker: PhantomData,
        }
    }
}

impl<F: Fingerprint> EvolvableStrategy for FingerprintStrategy<F> {
    type Genes = Vec<Gene>;

    fn random(label: impl Into<String>, rng: &mut fastrand::Rng) -> Self {
        let len = F::all().len();
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

impl<F: Fingerprint> Strategy for FingerprintStrategy<F> {
    fn cache_dependencies(&self) -> &[CacheId] {
        F::CACHE_DEPENDENCIES
    }

    fn choose_move(
        &self,
        current_stone: Stone,
        board: &Board,
        cache_repo: &CacheRepository,
        rng: &mut fastrand::Rng,
    ) -> PositionId {
        let index_cache =
            F::get_cache(cache_repo).expect("Strategy requires its fingerprint index cache");

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

pub type NN1 = FingerprintStrategy<FingerprintNN1>;
pub type NN2 = FingerprintStrategy<FingerprintNN2>;
pub type NN3 = FingerprintStrategy<FingerprintNN3>;
pub type NN4 = FingerprintStrategy<FingerprintNN4>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::NeighborCounts;
    use crate::position::Position;
    use std::collections::HashSet;

    fn pos(row: usize, col: usize) -> PositionId {
        PositionId::from_position(Position::new(row, col))
    }

    fn counts(player: u8, opponent: u8, empty: u8) -> NeighborCounts {
        NeighborCounts::new(player, opponent, empty)
    }

    fn all_genes<F: Fingerprint>() -> Vec<Gene> {
        (0..F::all().len()).map(Gene::new).collect()
    }

    fn genes_with_first<F: Fingerprint>(first_idx: usize) -> Vec<Gene> {
        let mut priority = vec![Gene::new(first_idx)];
        priority.extend(
            (0..F::all().len())
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
        fn new<F: Fingerprint>() -> Self {
            let mut cache_repo = CacheRepository::new();
            cache_repo.activate(F::CACHE_ID);
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

    mod nn1_tests {
        use super::*;

        fn fp(player: u8, opponent: u8, empty: u8) -> FingerprintNN1 {
            FingerprintNN1::new(counts(player, opponent, empty))
        }

        #[test]
        fn choose_move_returns_empty_position() {
            let mut ctx = TestContext::new::<FingerprintNN1>();
            let strategy = NN1::new("test", all_genes::<FingerprintNN1>());
            let board = Board::new();

            let result = strategy.choose_move(Stone::Black, &board, &ctx.cache_repo, &mut ctx.rng);

            assert!(board.empty_position_ids().contains(&result));
        }

        #[test]
        fn choose_move_prefers_higher_priority_fingerprint() {
            let mut ctx = TestContext::new::<FingerprintNN1>();
            let all_empty = fp(0, 0, 4);
            let strategy = NN1::new(
                "test",
                genes_with_first::<FingerprintNN1>(all_empty.index()),
            );
            let mut board = Board::new();
            ctx.place(&mut board, PositionId::center(), Stone::Black);

            let result = strategy.choose_move(Stone::White, &board, &ctx.cache_repo, &mut ctx.rng);

            assert!(board.empty_position_ids().contains(&result));
        }

        #[test]
        fn choose_move_prefers_self_neighbors_when_prioritized() {
            let mut ctx = TestContext::new::<FingerprintNN1>();
            let self_neighbor = fp(1, 0, 3);
            let strategy = NN1::new(
                "test",
                genes_with_first::<FingerprintNN1>(self_neighbor.index()),
            );
            let mut board = Board::new();
            ctx.place(&mut board, PositionId::center(), Stone::Black);

            let result = strategy.choose_move(Stone::Black, &board, &ctx.cache_repo, &mut ctx.rng);

            let center_neighbors = [pos(6, 7), pos(8, 7), pos(7, 6), pos(7, 8)];
            assert!(center_neighbors.contains(&result));
        }

        #[test]
        fn choose_move_prefers_edge_mixed_when_prioritized() {
            let mut ctx = TestContext::new::<FingerprintNN1>();
            let edge_mixed = fp(1, 1, 1);
            let strategy = NN1::new(
                "test",
                genes_with_first::<FingerprintNN1>(edge_mixed.index()),
            );
            let mut board = Board::new();

            ctx.place(&mut board, pos(0, 0), Stone::Black);
            ctx.place(&mut board, pos(2, 0), Stone::White);

            let result = strategy.choose_move(Stone::Black, &board, &ctx.cache_repo, &mut ctx.rng);

            assert_eq!(result, pos(1, 0));
        }

        #[test]
        fn label_returns_provided_label() {
            let strategy = NN1::new("my_nn1_strategy", all_genes::<FingerprintNN1>());
            assert_eq!(strategy.label(), "my_nn1_strategy");
        }

        #[test]
        fn genes_returns_ranked_fingerprints() {
            let priority = all_genes::<FingerprintNN1>();
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

            let genes_set: HashSet<usize> = strategy.genes().iter().map(|g| g.index()).collect();
            let all_set: HashSet<usize> = (0..FingerprintNN1::all().len()).collect();
            assert_eq!(genes_set, all_set);
        }

        #[test]
        fn random_genes_is_shuffled() {
            let mut rng = fastrand::Rng::with_seed(42);
            let strategy = NN1::random("test", &mut rng);

            let sequential: Vec<Gene> = (0..FingerprintNN1::all().len()).map(Gene::new).collect();
            assert_ne!(strategy.genes(), sequential.as_slice());
        }

        #[test]
        fn random_genes_deterministic_with_same_seed() {
            let s1 = NN1::random("test", &mut fastrand::Rng::with_seed(42));
            let s2 = NN1::random("test", &mut fastrand::Rng::with_seed(42));
            assert_eq!(s1.genes(), s2.genes());
        }

        #[test]
        fn random_genes_different_with_different_seeds() {
            let s1 = NN1::random("test", &mut fastrand::Rng::with_seed(1));
            let s2 = NN1::random("test", &mut fastrand::Rng::with_seed(2));
            assert_ne!(s1.genes(), s2.genes());
        }
    }

    mod nn2_tests {
        use super::*;

        fn fp2(nn1: NeighborCounts, nn2: NeighborCounts) -> FingerprintNN2 {
            FingerprintNN2::new(nn1, nn2)
        }

        #[test]
        fn choose_move_returns_empty_position() {
            let mut ctx = TestContext::new::<FingerprintNN2>();
            let strategy = NN2::new("test", all_genes::<FingerprintNN2>());
            let board = Board::new();

            let result = strategy.choose_move(Stone::Black, &board, &ctx.cache_repo, &mut ctx.rng);

            assert!(board.empty_position_ids().contains(&result));
        }

        #[test]
        fn choose_move_prefers_edge_mixed_when_prioritized() {
            let mut ctx = TestContext::new::<FingerprintNN2>();
            let edge_mixed = fp2(counts(1, 1, 1), counts(1, 1, 0));
            let strategy = NN2::new(
                "test",
                genes_with_first::<FingerprintNN2>(edge_mixed.index()),
            );
            let mut board = Board::new();

            ctx.place(&mut board, pos(0, 0), Stone::Black);
            ctx.place(&mut board, pos(2, 0), Stone::White);
            ctx.place(&mut board, pos(0, 1), Stone::Black);
            ctx.place(&mut board, pos(2, 1), Stone::White);

            let result = strategy.choose_move(Stone::Black, &board, &ctx.cache_repo, &mut ctx.rng);

            assert_eq!(result, pos(1, 0));
        }

        #[test]
        fn label_returns_provided_label() {
            let strategy = NN2::new("my_nn2_strategy", all_genes::<FingerprintNN2>());
            assert_eq!(strategy.label(), "my_nn2_strategy");
        }

        #[test]
        fn genes_returns_ranked_fingerprints() {
            let priority = all_genes::<FingerprintNN2>();
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
        fn random_genes_contains_all_indices() {
            let mut rng = fastrand::Rng::with_seed(42);
            let strategy = NN2::random("test", &mut rng);

            let genes_set: HashSet<_> = strategy.genes().iter().copied().collect();
            let all_set: HashSet<_> = all_genes::<FingerprintNN2>().into_iter().collect();
            assert_eq!(genes_set, all_set);
        }

        #[test]
        fn random_genes_is_shuffled() {
            let mut rng = fastrand::Rng::with_seed(42);
            let strategy = NN2::random("test", &mut rng);
            assert_ne!(strategy.genes(), all_genes::<FingerprintNN2>().as_slice());
        }

        #[test]
        fn random_genes_deterministic_with_same_seed() {
            let s1 = NN2::random("test", &mut fastrand::Rng::with_seed(42));
            let s2 = NN2::random("test", &mut fastrand::Rng::with_seed(42));
            assert_eq!(s1.genes(), s2.genes());
        }

        #[test]
        fn random_genes_different_with_different_seeds() {
            let s1 = NN2::random("test", &mut fastrand::Rng::with_seed(1));
            let s2 = NN2::random("test", &mut fastrand::Rng::with_seed(2));
            assert_ne!(s1.genes(), s2.genes());
        }
    }

    mod nn3_tests {
        use super::*;

        fn fp3(nn1: NeighborCounts, nn2: NeighborCounts, nn3: NeighborCounts) -> FingerprintNN3 {
            FingerprintNN3::new(nn1, nn2, nn3)
        }

        #[test]
        fn choose_move_returns_empty_position() {
            let mut ctx = TestContext::new::<FingerprintNN3>();
            let strategy = NN3::new("test", all_genes::<FingerprintNN3>());
            let board = Board::new();

            let result = strategy.choose_move(Stone::Black, &board, &ctx.cache_repo, &mut ctx.rng);

            assert!(board.empty_position_ids().contains(&result));
        }

        #[test]
        fn choose_move_prefers_nn3_pattern_when_prioritized() {
            let mut ctx = TestContext::new::<FingerprintNN3>();
            let specific_fp = fp3(counts(2, 2, 0), counts(1, 1, 2), counts(0, 0, 4));
            let strategy = NN3::new(
                "test",
                genes_with_first::<FingerprintNN3>(specific_fp.index()),
            );
            let mut board = Board::new();

            ctx.place(&mut board, pos(6, 7), Stone::White);
            ctx.place(&mut board, pos(8, 7), Stone::White);
            ctx.place(&mut board, pos(7, 6), Stone::Black);
            ctx.place(&mut board, pos(7, 8), Stone::Black);
            ctx.place(&mut board, pos(6, 6), Stone::Black);
            ctx.place(&mut board, pos(8, 8), Stone::White);

            let result = strategy.choose_move(Stone::Black, &board, &ctx.cache_repo, &mut ctx.rng);

            assert_eq!(result, pos(7, 7));
        }

        #[test]
        fn label_returns_provided_label() {
            let strategy = NN3::new("my_nn3_strategy", all_genes::<FingerprintNN3>());
            assert_eq!(strategy.label(), "my_nn3_strategy");
        }

        #[test]
        fn genes_returns_ranked_fingerprints() {
            let priority = all_genes::<FingerprintNN3>();
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
            let all_set: HashSet<_> = all_genes::<FingerprintNN3>().into_iter().collect();
            assert_eq!(genes_set, all_set);
        }

        #[test]
        fn random_genes_is_shuffled() {
            let mut rng = fastrand::Rng::with_seed(42);
            let strategy = NN3::random("test", &mut rng);
            assert_ne!(strategy.genes(), all_genes::<FingerprintNN3>().as_slice());
        }

        #[test]
        fn random_genes_deterministic_with_same_seed() {
            let s1 = NN3::random("test", &mut fastrand::Rng::with_seed(42));
            let s2 = NN3::random("test", &mut fastrand::Rng::with_seed(42));
            assert_eq!(s1.genes(), s2.genes());
        }

        #[test]
        fn random_genes_different_with_different_seeds() {
            let s1 = NN3::random("test", &mut fastrand::Rng::with_seed(1));
            let s2 = NN3::random("test", &mut fastrand::Rng::with_seed(2));
            assert_ne!(s1.genes(), s2.genes());
        }
    }

    mod nn4_tests {
        use super::*;

        fn fp4(
            nn1: NeighborCounts,
            nn2: NeighborCounts,
            nn3: NeighborCounts,
            nn4: NeighborCounts,
        ) -> FingerprintNN4 {
            FingerprintNN4::new(nn1, nn2, nn3, nn4)
        }

        #[test]
        fn choose_move_returns_empty_position() {
            let mut ctx = TestContext::new::<FingerprintNN4>();
            let strategy = NN4::new("test", all_genes::<FingerprintNN4>());
            let board = Board::new();

            let result = strategy.choose_move(Stone::Black, &board, &ctx.cache_repo, &mut ctx.rng);

            assert!(board.empty_position_ids().contains(&result));
        }

        #[test]
        fn choose_move_prefers_nn4_pattern_when_prioritized() {
            let mut ctx = TestContext::new::<FingerprintNN4>();
            let specific_fp = fp4(
                counts(3, 1, 0),
                counts(1, 1, 2),
                counts(0, 2, 2),
                counts(3, 1, 4),
            );
            let strategy = NN4::new(
                "test",
                genes_with_first::<FingerprintNN4>(specific_fp.index()),
            );
            let mut board = Board::new();

            ctx.place(&mut board, pos(6, 7), Stone::White);
            ctx.place(&mut board, pos(8, 7), Stone::Black);
            ctx.place(&mut board, pos(7, 6), Stone::Black);
            ctx.place(&mut board, pos(7, 8), Stone::Black);
            ctx.place(&mut board, pos(6, 6), Stone::Black);
            ctx.place(&mut board, pos(8, 6), Stone::White);
            ctx.place(&mut board, pos(5, 7), Stone::White);
            ctx.place(&mut board, pos(7, 5), Stone::White);
            ctx.place(&mut board, pos(5, 6), Stone::Black);
            ctx.place(&mut board, pos(5, 8), Stone::Black);
            ctx.place(&mut board, pos(6, 5), Stone::Black);
            ctx.place(&mut board, pos(9, 6), Stone::White);

            let result = strategy.choose_move(Stone::Black, &board, &ctx.cache_repo, &mut ctx.rng);

            assert_eq!(result, pos(7, 7));
        }

        #[test]
        fn label_returns_provided_label() {
            let strategy = NN4::new("my_nn4_strategy", all_genes::<FingerprintNN4>());
            assert_eq!(strategy.label(), "my_nn4_strategy");
        }

        #[test]
        fn genes_returns_ranked_fingerprints() {
            let priority = all_genes::<FingerprintNN4>();
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
            let all_set: HashSet<_> = all_genes::<FingerprintNN4>().into_iter().collect();
            assert_eq!(genes_set, all_set);
        }

        #[test]
        fn random_genes_is_shuffled() {
            let mut rng = fastrand::Rng::with_seed(42);
            let strategy = NN4::random("test", &mut rng);
            assert_ne!(strategy.genes(), all_genes::<FingerprintNN4>().as_slice());
        }

        #[test]
        fn random_genes_deterministic_with_same_seed() {
            let s1 = NN4::random("test", &mut fastrand::Rng::with_seed(42));
            let s2 = NN4::random("test", &mut fastrand::Rng::with_seed(42));
            assert_eq!(s1.genes(), s2.genes());
        }

        #[test]
        fn random_genes_different_with_different_seeds() {
            let s1 = NN4::random("test", &mut fastrand::Rng::with_seed(1));
            let s2 = NN4::random("test", &mut fastrand::Rng::with_seed(2));
            assert_ne!(s1.genes(), s2.genes());
        }
    }
}
