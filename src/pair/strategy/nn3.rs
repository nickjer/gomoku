use crate::board::Board;
use crate::cache_id::CacheId;
use crate::cache_repository::CacheRepository;
use crate::pair::BestPositionSelector;
use crate::pair::fingerprint::FingerprintNN3;
use crate::position_id::PositionId;
use crate::stone::Stone;
use crate::strategy::{EvolvableStrategy, Strategy};

/// A strategy based on NN3 fingerprints with evolvable gene priority.
pub struct NN3 {
    label: String,
    genes: Vec<FingerprintNN3>,
    selector: BestPositionSelector<FingerprintNN3>,
}

impl NN3 {
    #[must_use]
    pub fn new(label: impl Into<String>, genes: Vec<FingerprintNN3>) -> Self {
        let selector = BestPositionSelector::new(&genes);
        Self {
            label: label.into(),
            genes,
            selector,
        }
    }
}

impl EvolvableStrategy for NN3 {
    type Gene = FingerprintNN3;

    fn random_genes(rng: &mut fastrand::Rng) -> Vec<Self::Gene> {
        let mut genes = FingerprintNN3::all().to_vec();
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

impl Strategy for NN3 {
    fn cache_dependencies(&self) -> &[CacheId] {
        &[
            CacheId::NeighborNN1,
            CacheId::NeighborNN2,
            CacheId::NeighborNN3,
        ]
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
                    FingerprintNN3::calculate(pos, current_stone, cache_repo),
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

    fn counts(player: u8, opponent: u8, empty: u8) -> NeighborCounts {
        NeighborCounts::new(player, opponent, empty)
    }

    fn fp3(nn1: NeighborCounts, nn2: NeighborCounts, nn3: NeighborCounts) -> FingerprintNN3 {
        FingerprintNN3::new(nn1, nn2, nn3)
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
        let strategy = NN3::new("test", FingerprintNN3::all().to_vec());
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
        let mut priority = vec![specific_fp];
        priority.extend(FingerprintNN3::all().iter().filter(|&&f| f != specific_fp));

        let strategy = NN3::new("test", priority);
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
        let fingerprint = FingerprintNN3::calculate(result, Stone::Black, &ctx.cache_repo);

        assert_eq!(result, pos(7, 7)); // center
        assert_eq!(fingerprint, specific_fp);
    }

    #[test]
    #[should_panic(expected = "Unknown fingerprint in priority map")]
    fn choose_move_panics_on_unknown_fingerprint() {
        let mut ctx = TestContext::new();
        let incomplete_priority = vec![fp3(counts(0, 0, 4), counts(0, 0, 4), counts(0, 0, 4))];

        let strategy = NN3::new("test", incomplete_priority);
        let mut board = Board::new();
        ctx.place(&mut board, PositionId::center(), Stone::Black);

        strategy.choose_move(Stone::White, &board, &ctx.cache_repo, &mut ctx.rng);
    }

    #[test]
    fn label_returns_provided_label() {
        let strategy = NN3::new("my_nn3_strategy", FingerprintNN3::all().to_vec());

        assert_eq!(strategy.label(), "my_nn3_strategy");
    }

    #[test]
    fn genes_returns_fingerprint_priority() {
        let priority: Vec<_> = FingerprintNN3::all().to_vec();
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
    fn random_genes_contains_all_fingerprints() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = NN3::random("test", &mut rng);

        let genes_set: HashSet<_> = strategy.genes().iter().collect();
        let all_set: HashSet<_> = FingerprintNN3::all().iter().collect();

        assert_eq!(genes_set, all_set);
    }

    #[test]
    fn random_genes_is_shuffled() {
        let mut rng = fastrand::Rng::with_seed(42);
        let strategy = NN3::random("test", &mut rng);

        assert_ne!(strategy.genes(), FingerprintNN3::all());
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
