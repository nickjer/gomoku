//! Board positions with one right answer, used to test whether a strategy can finish
//! its own four and block the opponent's fours and threes.
//!
//! Every scenario is stamped from a pattern: a short line of cells laid along one of the
//! four directions, with a few random filler stones for the other side so the board
//! looks like a game in progress.

use crate::board::Board;
use crate::offset::Offset;
use crate::position::Position;
use crate::position_id::PositionId;
use crate::stone::Stone;
use crate::strategy::Strategy;

/// The four directions to check for lines.
const DIRECTIONS: [Offset; 4] = [
    Offset::new(0, 1),  // horizontal
    Offset::new(1, 0),  // vertical
    Offset::new(1, 1),  // diagonal down-right
    Offset::new(1, -1), // diagonal down-left
];

/// What one cell along the line holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cell {
    /// A stone of the player to move.
    Mover,
    /// A stone of the opponent.
    Opponent,
    /// Stays empty so the shape reads as intended; playing here is not the answer.
    Empty,
    /// Stays empty; playing here is a correct move.
    Answer,
}

use Cell::{Answer, Empty, Mover, Opponent};

/// Four own stones in a row: complete the five at either end.
const WINNING_MOVE: &[Cell] = &[Answer, Mover, Mover, Mover, Mover, Answer];

/// Four opponent stones in a row: block either end or lose next move.
const FOUR_IN_A_ROW: &[Cell] = &[Answer, Opponent, Opponent, Opponent, Opponent, Answer];

/// Three opponent stones with both ends open: block an end before it becomes an open four.
const OPEN_THREE: &[Cell] = &[Answer, Opponent, Opponent, Opponent, Answer];

/// Two opponent stones, a gap, one more, both ends open: only the gap stops the open four.
const SPLIT_THREE: &[Cell] = &[Empty, Opponent, Opponent, Answer, Opponent, Empty];

const PATTERNS: [&[Cell]; 4] = [WINNING_MOVE, FOUR_IN_A_ROW, OPEN_THREE, SPLIT_THREE];

/// How many scenarios each pattern gets per colour.
const SCENARIOS_PER_PATTERN_AND_COLOUR: usize = 10;

/// A board where the player to move has a small set of correct moves.
#[derive(Debug, Clone)]
pub struct ThreatScenario {
    board: Board,
    answers: Vec<PositionId>,
    player_to_move: Stone,
}

impl ThreatScenario {
    /// Lays `pattern` along a random direction from a random start that keeps every cell on
    /// the board, then adds two to four filler stones for the side with fewer pattern stones.
    fn from_pattern(pattern: &[Cell], player_to_move: Stone, rng: &mut fastrand::Rng) -> Self {
        let direction = DIRECTIONS[rng.usize(..DIRECTIONS.len())];

        // The last cell is `span` steps from the start, so shrink the start range on each
        // axis the direction moves along.
        let span = pattern.len() - 1;
        let start_range = |delta: isize| match delta.signum() {
            1 => 0..PositionId::WIDTH - span,
            -1 => span..PositionId::WIDTH,
            _ => 0..PositionId::WIDTH,
        };
        let start = PositionId::from_position(Position::new(
            rng.usize(start_range(direction.row_delta())),
            rng.usize(start_range(direction.col_delta())),
        ));

        let mut board = Board::new();
        let mut line = Vec::with_capacity(pattern.len());
        let mut answers = Vec::new();
        let cells = std::iter::successors(Some(start), |position| position.offset(direction));
        for (position, &cell) in cells.zip(pattern) {
            line.push(position);
            match cell {
                Mover => board
                    .place(position, player_to_move)
                    .expect("pattern cells are distinct"),
                Opponent => board
                    .place(position, player_to_move.opponent())
                    .expect("pattern cells are distinct"),
                Answer => answers.push(position),
                Empty => {}
            }
        }
        debug_assert_eq!(
            line.len(),
            pattern.len(),
            "start chosen so the pattern fits"
        );

        let mover_stones = pattern.iter().filter(|&&cell| cell == Mover).count();
        let opponent_stones = pattern.iter().filter(|&&cell| cell == Opponent).count();
        let filler = if mover_stones < opponent_stones {
            player_to_move
        } else {
            player_to_move.opponent()
        };
        let mut remaining = rng.usize(2..=4);
        while remaining > 0 {
            let position = PositionId::from_index(rng.usize(..PositionId::COUNT));
            if board.is_empty(position) && !line.contains(&position) {
                board
                    .place(position, filler)
                    .expect("position checked empty");
                remaining -= 1;
            }
        }

        Self {
            board,
            answers,
            player_to_move,
        }
    }

    /// Returns true if `position` is one of the correct moves for this scenario.
    #[must_use]
    pub fn is_correct_move(&self, position: PositionId) -> bool {
        self.answers.contains(&position)
    }
}

/// Generates scenarios for every pattern, for both colours to move.
pub fn generate_threat_scenarios(rng: &mut fastrand::Rng) -> Vec<ThreatScenario> {
    let mut scenarios = Vec::with_capacity(PATTERNS.len() * 2 * SCENARIOS_PER_PATTERN_AND_COLOUR);
    for pattern in PATTERNS {
        for _ in 0..SCENARIOS_PER_PATTERN_AND_COLOUR {
            for player_to_move in [Stone::Black, Stone::White] {
                scenarios.push(ThreatScenario::from_pattern(pattern, player_to_move, rng));
            }
        }
    }
    scenarios
}

/// Counts the scenarios in which the strategy plays a correct move.
///
/// # Panics
///
/// Panics if the scenario count exceeds `u16::MAX`.
pub fn count_correct_moves<S: Strategy>(
    strategy: &S,
    scenarios: &[ThreatScenario],
    rng: &mut fastrand::Rng,
) -> f32 {
    let correct = scenarios
        .iter()
        .filter(|scenario| {
            let chosen = strategy.choose_move(scenario.player_to_move, &scenario.board, rng);
            scenario.is_correct_move(chosen)
        })
        .count();
    f32::from(u16::try_from(correct).expect("scenario count fits in u16"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitboard::winning_threats;
    use crate::test_utils::ScriptedStrategy;

    fn pos(row: usize, col: usize) -> PositionId {
        PositionId::from_position(Position::new(row, col))
    }

    fn scenario_with_stones(
        stones: &[(usize, usize)],
        stone: Stone,
        answers: &[(usize, usize)],
        player_to_move: Stone,
    ) -> ThreatScenario {
        let mut board = Board::new();
        for &(row, col) in stones {
            board.place(pos(row, col), stone).unwrap();
        }
        ThreatScenario {
            board,
            answers: answers.iter().map(|&(row, col)| pos(row, col)).collect(),
            player_to_move,
        }
    }

    fn stone_count(board: &Board, stone: Stone) -> usize {
        PositionId::iter()
            .filter(|&position| board.stone(position) == Some(stone))
            .count()
    }

    #[test]
    fn generates_every_pattern_for_both_colours() {
        let mut rng = fastrand::Rng::with_seed(42);

        let scenarios = generate_threat_scenarios(&mut rng);

        assert_eq!(
            scenarios.len(),
            PATTERNS.len() * 2 * SCENARIOS_PER_PATTERN_AND_COLOUR
        );
        let black_to_move = scenarios
            .iter()
            .filter(|scenario| scenario.player_to_move == Stone::Black)
            .count();
        assert_eq!(black_to_move, scenarios.len() / 2);
    }

    #[test]
    fn answers_are_empty_in_every_generated_scenario() {
        for seed in 0..20 {
            let mut rng = fastrand::Rng::with_seed(seed);
            for scenario in generate_threat_scenarios(&mut rng) {
                assert!(!scenario.answers.is_empty());
                for &answer in &scenario.answers {
                    assert!(scenario.board.is_empty(answer));
                }
            }
        }
    }

    #[test]
    fn filler_stones_go_to_the_side_with_fewer_pattern_stones() {
        let mut rng = fastrand::Rng::with_seed(7);

        let winning = ThreatScenario::from_pattern(WINNING_MOVE, Stone::Black, &mut rng);
        assert_eq!(stone_count(&winning.board, Stone::Black), 4);
        assert!((2..=4).contains(&stone_count(&winning.board, Stone::White)));

        let blocking = ThreatScenario::from_pattern(OPEN_THREE, Stone::Black, &mut rng);
        assert_eq!(stone_count(&blocking.board, Stone::White), 3);
        assert!((2..=4).contains(&stone_count(&blocking.board, Stone::Black)));
    }

    #[test]
    fn four_stone_answers_agree_with_bitboard_winning_threats() {
        for seed in 0..50 {
            let mut rng = fastrand::Rng::with_seed(seed);
            for (pattern, owner) in [(WINNING_MOVE, Stone::Black), (FOUR_IN_A_ROW, Stone::White)] {
                let scenario = ThreatScenario::from_pattern(pattern, Stone::Black, &mut rng);
                let threats = winning_threats(*scenario.board.bitboard(owner));

                assert_eq!(scenario.answers.len(), 2);
                assert_eq!(threats.iter_set().count(), 2);
                for &answer in &scenario.answers {
                    assert!(threats.is_set(answer));
                }
            }
        }
    }

    #[test]
    fn split_three_answer_sits_between_the_stones() {
        for seed in 0..50 {
            let mut rng = fastrand::Rng::with_seed(seed);
            let scenario = ThreatScenario::from_pattern(SPLIT_THREE, Stone::Black, &mut rng);

            let [gap] = scenario.answers[..] else {
                panic!("split three has exactly one answer");
            };
            let white_neighbours = Offset::CENTER_AND_NEIGHBORS[1..]
                .iter()
                .filter_map(|&offset| gap.offset(offset))
                .filter(|&neighbour| scenario.board.stone(neighbour) == Some(Stone::White))
                .count();
            assert!(white_neighbours >= 2, "seed {seed}");
        }
    }

    #[test]
    fn four_in_a_row_scenario_identifies_blocking_moves() {
        let scenario = scenario_with_stones(
            &[(7, 5), (7, 6), (7, 7), (7, 8)],
            Stone::White,
            &[(7, 4), (7, 9)],
            Stone::Black,
        );

        assert!(scenario.is_correct_move(pos(7, 4)));
        assert!(scenario.is_correct_move(pos(7, 9)));
        assert!(!scenario.is_correct_move(pos(7, 7))); // occupied
        assert!(!scenario.is_correct_move(pos(0, 0))); // not blocking
    }

    #[test]
    fn split_three_scenario_identifies_only_the_gap() {
        let scenario = scenario_with_stones(
            &[(7, 5), (7, 6), (7, 8)],
            Stone::White,
            &[(7, 7)],
            Stone::Black,
        );

        assert!(scenario.is_correct_move(pos(7, 7)));
        assert!(!scenario.is_correct_move(pos(7, 4))); // open end, not a block
        assert!(!scenario.is_correct_move(pos(7, 9))); // open end, not a block
    }

    #[test]
    fn count_correct_moves_scores_blocking_and_winning_strategies() {
        let block = scenario_with_stones(
            &[(7, 5), (7, 6), (7, 7), (7, 8)],
            Stone::White,
            &[(7, 4), (7, 9)],
            Stone::Black,
        );
        let win = scenario_with_stones(
            &[(3, 5), (3, 6), (3, 7), (3, 8)],
            Stone::Black,
            &[(3, 4), (3, 9)],
            Stone::Black,
        );
        let scenarios = [block, win];
        let mut rng = fastrand::Rng::with_seed(42);

        let right = ScriptedStrategy::with_positions("right", &[(7, 4), (3, 9)]);
        assert_eq!(count_correct_moves(&right, &scenarios, &mut rng), 2.0);

        let half = ScriptedStrategy::with_positions("half", &[(7, 4), (0, 0)]);
        assert_eq!(count_correct_moves(&half, &scenarios, &mut rng), 1.0);

        let wrong = ScriptedStrategy::with_positions("wrong", &[(0, 0), (0, 1)]);
        assert_eq!(count_correct_moves(&wrong, &scenarios, &mut rng), 0.0);
    }
}
