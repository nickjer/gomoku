use crate::board::Board;
use crate::match_result::MatchResult;
use crate::stone::Stone;
use crate::strategy::Strategy;

use super::{GameObserver, Play, run_from};

/// Freestyle Gomoku with a random opening: places `moves` stones alternating Black/White on
/// random empty positions before the strategies begin play.
///
/// If the board reaches a finished state during setup (unlikely but possible with a large
/// `moves` count), setup stops early and the game resolves immediately.
#[derive(Debug, Clone, Copy, Default)]
pub struct RandomOpening {
    pub moves: u32,
}

impl Play for RandomOpening {
    fn play_from(
        &self,
        board: &mut Board,
        black_strategy: &dyn Strategy,
        white_strategy: &dyn Strategy,
        observer: &mut dyn GameObserver,
        rng: &mut fastrand::Rng,
    ) -> Option<MatchResult> {
        for _ in 0..self.moves {
            if board.is_finished() {
                break;
            }
            let stone = if board.move_count().is_multiple_of(2) {
                Stone::Black
            } else {
                Stone::White
            };
            let empty = board.empty_position_ids();
            let position = empty[rng.usize(0..empty.len())];
            board
                .place(position, stone)
                .expect("position chosen from empty list");
        }
        run_from(board, black_strategy, white_strategy, observer, rng)
    }
}

#[cfg(test)]
mod tests {
    use std::ops::ControlFlow;

    use super::*;
    use crate::board::Board;
    use crate::game::{Freestyle, GameObserver, NoOpObserver};
    use crate::position_id::PositionId;
    use crate::stone::Stone;
    use crate::test_utils::ScriptedStrategy;

    /// Observer that breaks on the very first call, used to isolate opening stone count.
    struct BreakImmediately;

    impl GameObserver for BreakImmediately {
        fn on_move(&mut self, _: Stone, _: PositionId, _: &Board) -> ControlFlow<()> {
            ControlFlow::Break(())
        }
    }

    #[test]
    fn board_has_correct_stone_count_after_opening() {
        let game = RandomOpening { moves: 4 };
        let (black, white) = ScriptedStrategy::black_wins();
        let mut board = Board::new();
        let mut rng = fastrand::Rng::with_seed(42);

        // Break before any strategy move to inspect only the opening stones.
        // With 4 opening stones on a 15×15 board, 5-in-a-row is impossible,
        // so the game cannot finish during setup.
        game.play_from(&mut board, &black, &white, &mut BreakImmediately, &mut rng);

        assert_eq!(board.move_count(), 4);
    }

    #[test]
    fn zero_opening_moves_produces_same_result_as_freestyle() {
        let (black_a, white_a) = ScriptedStrategy::black_wins();
        let (black_b, white_b) = ScriptedStrategy::black_wins();
        let mut rng_a = fastrand::Rng::with_seed(42);
        let mut rng_b = fastrand::Rng::with_seed(42);

        let result_random = RandomOpening { moves: 0 }.play(&black_a, &white_a, &mut rng_a);
        let result_freestyle = Freestyle.play(&black_b, &white_b, &mut rng_b);

        assert_eq!(result_random.outcome(), result_freestyle.outcome());
        assert_eq!(result_random.turn_count(), result_freestyle.turn_count());
    }

    #[test]
    fn opening_setup_does_not_panic_across_seeds() {
        // Verify that random opening setup never panics for a range of seeds.
        // Uses BreakImmediately so no strategy moves are attempted after the opening.
        // Strategies are created fresh per iteration since ScriptedStrategy tracks an index.
        for seed in 0..50 {
            let (black, white) = ScriptedStrategy::black_wins();
            let game = RandomOpening { moves: 10 };
            let mut board = Board::new();
            let mut rng = fastrand::Rng::with_seed(seed);
            game.play_from(&mut board, &black, &white, &mut BreakImmediately, &mut rng);
            assert_eq!(board.move_count(), 10);
        }
    }

    #[test]
    fn turn_count_in_result_includes_opening_stones() {
        // Black wins in 9 turns from empty. With 4 opening stones pre-placed at positions
        // that don't overlap with the scripted moves, total turn_count should be 13.
        // We place the opening stones at rows 5–6 (far from row 0–1 used by scripted moves).
        let (black, white) = ScriptedStrategy::black_wins();
        let mut board = Board::new();
        let mut rng = fastrand::Rng::with_seed(0);

        // Pre-seed the board with 4 specific non-conflicting stones, then run the scripted
        // strategies from that position. Rows 5–6 don't overlap with the scripted moves
        // (which use rows 0–1), so the strategies play normally.
        use crate::position::Position;
        use crate::position_id::PositionId;
        board
            .place(PositionId::from_position(Position::new(5, 0)), Stone::Black)
            .unwrap();
        board
            .place(PositionId::from_position(Position::new(5, 1)), Stone::White)
            .unwrap();
        board
            .place(PositionId::from_position(Position::new(6, 0)), Stone::Black)
            .unwrap();
        board
            .place(PositionId::from_position(Position::new(6, 1)), Stone::White)
            .unwrap();

        let result = Freestyle.play_from(&mut board, &black, &white, &mut NoOpObserver, &mut rng);

        assert_eq!(result.unwrap().turn_count(), 9 + 4);
    }
}
