use crate::board::Board;
use crate::cache_id::CacheId;
use crate::cache_repository::CacheRepository;
use crate::pair::BestPositionSelector;
use crate::pair::fingerprint::FingerprintNN1;
use crate::position_id::PositionId;
use crate::stone::Stone;
use crate::strategy::Strategy;

/// A strategy based on NN1 fingerprints with evolvable gene priority.
pub struct NN1 {
    label: String,
    genes: Vec<FingerprintNN1>,
    selector: BestPositionSelector<FingerprintNN1>,
}

impl NN1 {
    #[must_use]
    pub fn new(label: impl Into<String>, genes: Vec<FingerprintNN1>) -> Self {
        let selector = BestPositionSelector::new(&genes);
        Self {
            label: label.into(),
            genes,
            selector,
        }
    }

    #[must_use]
    pub fn random(label: impl Into<String>, rng: &mut fastrand::Rng) -> Self {
        let mut genes: Vec<_> = FingerprintNN1::all().to_vec();
        rng.shuffle(&mut genes);
        Self::new(label, genes)
    }

    #[must_use]
    pub fn genes(&self) -> &[FingerprintNN1] {
        &self.genes
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
        let position_fingerprints: Vec<_> = board
            .empty_position_ids()
            .iter()
            .map(|&pos| {
                (
                    pos,
                    FingerprintNN1::calculate(pos, current_stone, cache_repo),
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
    use crate::pair::NeighborCounts;
    use crate::position::Position;
    use std::collections::HashSet;

    fn pos(row: u8, col: u8) -> PositionId {
        PositionId::from_position(Position::new(row, col))
    }

    fn fp(player: u8, opponent: u8, empty: u8) -> FingerprintNN1 {
        FingerprintNN1::new(NeighborCounts::new(player, opponent, empty))
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
        let strategy = NN1::new("test", FingerprintNN1::all().to_vec());
        let board = Board::new();

        let result = strategy.choose_move(Stone::Black, &board, &ctx.cache_repo, &mut ctx.rng);

        assert!(board.empty_position_ids().contains(&result));
    }

    #[test]
    fn choose_move_prefers_higher_priority_fingerprint() {
        let mut ctx = TestContext::new();
        let all_empty = fp(0, 0, 4);
        let mut priority = vec![all_empty];
        priority.extend(FingerprintNN1::all().iter().filter(|&&f| f != all_empty));

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
        let mut priority = vec![self_neighbor];
        priority.extend(
            FingerprintNN1::all()
                .iter()
                .filter(|&&f| f != self_neighbor),
        );

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
        let mut priority = vec![edge_mixed];
        priority.extend(FingerprintNN1::all().iter().filter(|&&f| f != edge_mixed));

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
    #[should_panic(expected = "Unknown fingerprint in priority map")]
    fn choose_move_panics_on_unknown_fingerprint() {
        let mut ctx = TestContext::new();
        let incomplete_priority = vec![fp(0, 0, 4)];

        let strategy = NN1::new("test", incomplete_priority);
        let mut board = Board::new();
        ctx.place(&mut board, PositionId::center(), Stone::Black);

        strategy.choose_move(Stone::White, &board, &ctx.cache_repo, &mut ctx.rng);
    }

    #[test]
    fn label_returns_provided_label() {
        let strategy = NN1::new("my_nn1_strategy", FingerprintNN1::all().to_vec());

        assert_eq!(strategy.label(), "my_nn1_strategy");
    }

    #[test]
    fn genes_returns_fingerprint_priority() {
        let priority: Vec<_> = FingerprintNN1::all().to_vec();
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
    fn random_genes_contains_all_fingerprints() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = NN1::random("test", &mut rng);

        let genes_set: HashSet<_> = strategy.genes().iter().collect();
        let all_set: HashSet<_> = FingerprintNN1::all().iter().collect();

        assert_eq!(genes_set, all_set);
    }

    #[test]
    fn random_genes_is_shuffled() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = NN1::random("test", &mut rng);

        assert_ne!(strategy.genes(), FingerprintNN1::all());
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
