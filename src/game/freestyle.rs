use tracing::instrument;

use crate::board::Board;
use crate::outcome::Outcome;
use crate::stone::Stone;
use crate::strategy::Strategy;

use super::{GameObserver, Play};

/// Freestyle Gomoku: 5+ in a row wins, no restrictions.
///
/// `opening_moves` stones are placed on random empty positions, alternating Black/White,
/// before the strategies begin play. Zero means the strategies start from the board as given.
#[derive(Debug, Clone, Copy, Default)]
pub struct Freestyle {
    pub opening_moves: u32,
}

impl Play for Freestyle {
    #[instrument(level = "debug", skip_all, fields(black = black_strategy.label(), white = white_strategy.label()))]
    fn play_from(
        &self,
        board: &mut Board,
        black_strategy: &dyn Strategy,
        white_strategy: &dyn Strategy,
        observer: &mut dyn GameObserver,
        rng: &mut fastrand::Rng,
    ) -> Option<Outcome> {
        // A large opening could finish the game by itself; stop placing if it does.
        for _ in 0..self.opening_moves {
            if board.is_finished() {
                break;
            }
            let empty = board.empty_position_ids();
            let position = empty[rng.usize(0..empty.len())];
            board
                .place(position, board.stone_to_move())
                .expect("position chosen from empty list");
        }

        while !board.is_finished() {
            let stone = board.stone_to_move();
            let strategy = match stone {
                Stone::Black => black_strategy,
                Stone::White => white_strategy,
            };

            let position = strategy.choose_move(stone, board, rng);

            if observer.on_move(stone, position, board).is_break() {
                return None;
            }

            board
                .place(position, stone)
                .expect("strategy returned invalid move");
        }

        Some(board.outcome().expect("game finished without outcome"))
    }
}

#[cfg(test)]
mod tests {
    use std::ops::ControlFlow;

    use super::*;
    use crate::game::NoOpObserver;
    use crate::position::Position;
    use crate::position_id::PositionId;
    use crate::test_utils::ScriptedStrategy;

    /// Observer that breaks on the very first call, used to isolate the opening stones.
    struct BreakImmediately;

    impl GameObserver for BreakImmediately {
        fn on_move(&mut self, _: Stone, _: PositionId, _: &Board) -> ControlFlow<()> {
            ControlFlow::Break(())
        }
    }

    #[test]
    fn black_five_in_a_row_wins() {
        let (black, white) = ScriptedStrategy::black_wins();
        let mut rng = fastrand::Rng::new();

        let outcome = Freestyle::default().play(&black, &white, &mut rng);

        assert_eq!(outcome, Outcome::Win(Stone::Black));
    }

    #[test]
    fn white_five_in_a_row_wins() {
        let (black, white) = ScriptedStrategy::white_wins();
        let mut rng = fastrand::Rng::new();

        let outcome = Freestyle::default().play(&black, &white, &mut rng);

        assert_eq!(outcome, Outcome::Win(Stone::White));
    }

    #[test]
    fn full_board_without_five_is_a_draw() {
        let (black, white) = ScriptedStrategy::draw();
        let mut rng = fastrand::Rng::new();

        let outcome = Freestyle::default().play(&black, &white, &mut rng);

        assert_eq!(outcome, Outcome::Draw);
    }

    #[test]
    fn board_holds_every_move_after_play() {
        let (black, white) = ScriptedStrategy::black_wins();
        let mut board = Board::new();
        let mut rng = fastrand::Rng::new();

        Freestyle::default().play_from(&mut board, &black, &white, &mut NoOpObserver, &mut rng);

        // Black plays 5 moves, white plays 4 moves = 9 total
        assert_eq!(board.move_count(), 9);
    }

    #[test]
    fn board_has_correct_stone_count_after_opening() {
        let game = Freestyle { opening_moves: 4 };
        let (black, white) = ScriptedStrategy::black_wins();
        let mut board = Board::new();
        let mut rng = fastrand::Rng::with_seed(42);

        // Break before any strategy move to inspect only the opening stones.
        game.play_from(&mut board, &black, &white, &mut BreakImmediately, &mut rng);

        assert_eq!(board.move_count(), 4);
    }

    #[test]
    fn opening_alternates_colours_starting_with_black() {
        let game = Freestyle { opening_moves: 5 };
        let (black, white) = ScriptedStrategy::black_wins();
        let mut board = Board::new();
        let mut rng = fastrand::Rng::with_seed(42);

        game.play_from(&mut board, &black, &white, &mut BreakImmediately, &mut rng);

        let black_count = PositionId::iter()
            .filter(|&pos| board.stone(pos) == Some(Stone::Black))
            .count();
        assert_eq!(black_count, 3);
        assert_eq!(board.stone_to_move(), Stone::White);
    }

    #[test]
    fn opening_setup_does_not_panic_across_seeds() {
        // Strategies are created fresh per iteration since ScriptedStrategy tracks an index.
        for seed in 0..50 {
            let (black, white) = ScriptedStrategy::black_wins();
            let game = Freestyle { opening_moves: 10 };
            let mut board = Board::new();
            let mut rng = fastrand::Rng::with_seed(seed);
            game.play_from(&mut board, &black, &white, &mut BreakImmediately, &mut rng);
            assert_eq!(board.move_count(), 10);
        }
    }

    #[test]
    fn play_continues_from_stones_already_on_the_board() {
        // Black wins in 9 turns from empty. Four stones pre-placed at rows 5–6 do not overlap
        // the scripted moves at rows 0–1, so the total is 13.
        let (black, white) = ScriptedStrategy::black_wins();
        let mut board = Board::new();
        let mut rng = fastrand::Rng::with_seed(0);
        for (row, col, stone) in [
            (5, 0, Stone::Black),
            (5, 1, Stone::White),
            (6, 0, Stone::Black),
            (6, 1, Stone::White),
        ] {
            board
                .place(PositionId::from_position(Position::new(row, col)), stone)
                .unwrap();
        }

        let outcome =
            Freestyle::default().play_from(&mut board, &black, &white, &mut NoOpObserver, &mut rng);

        assert_eq!(outcome, Some(Outcome::Win(Stone::Black)));
        assert_eq!(board.move_count(), 9 + 4);
    }
}
