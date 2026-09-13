use std::io::BufRead;

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use tracing::{info, instrument};

use crate::board::Board;
use crate::evolution::fitness_score::FitnessScore;
use crate::position_id::PositionId;
use crate::position_map::PositionArray;
use crate::stone::Stone;
use crate::strategy::EvolvableStrategy;

use super::EvaluateFitness;

/// Evaluates fitness by how closely each strategy's ranking of the empty
/// positions follows minimax's scores on boards from real games.
///
/// Every strategy in a generation is measured on the same sampled boards.
/// A board's error weights the shortfall of the strategy's k-th ranked
/// position by `rank_decay^(k-1)`, with the weights normalized to sum to
/// one; fitness is minus the mean error over the sample.
pub struct ScoredBoardFitness {
    boards: Vec<ScoredBoard>,
    boards_per_generation: usize,
    rank_decay: f32,
    cap: f32,
}

/// A board with, for every empty position, how far minimax's score for it
/// falls behind the board's best position, as a share of the cap.
struct ScoredBoard {
    board: Board,
    shortfall: PositionArray<Option<f32>>,
}

/// One line of the scored-board file: every position as `.`, `X` or `O`,
/// and every position's raw minimax score for the side to move, `null`
/// under a stone.
#[derive(Deserialize)]
struct ScoredBoardRecord {
    stones: String,
    scores: Vec<Option<f32>>,
}

/// A pick at least this far behind the best is counted as a blunder: a
/// losing move on a quiet board costs about the whole cap.
const BLUNDER_SHORTFALL: f32 = 0.9;

impl ScoredBoardFitness {
    /// Reads scored boards from JSON Lines. Scores are clamped to `[-cap, cap]`
    /// before the shortfalls are taken, and boards where every move ties are dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if the cap is not positive, a line does not parse or
    /// does not describe a whole board, or no board is left.
    pub fn read(
        reader: impl BufRead,
        cap: f32,
        boards_per_generation: usize,
        rank_decay: f32,
    ) -> Result<Self> {
        ensure!(cap > 0.0, "the score cap must be positive");
        let mut boards = Vec::new();
        let mut tied_boards = 0;
        for (line_index, line) in reader.lines().enumerate() {
            let line_number = line_index + 1;
            let record: ScoredBoardRecord = serde_json::from_str(&line?)
                .with_context(|| format!("line {line_number} is not a scored board"))?;
            ensure!(
                record.stones.len() == PositionId::COUNT
                    && record.scores.len() == PositionId::COUNT,
                "line {line_number}: expected {} stones and scores",
                PositionId::COUNT
            );

            let mut board = Board::new();
            for (position, symbol) in PositionId::iter().zip(record.stones.bytes()) {
                match symbol {
                    b'X' => board.place_unchecked(position, Stone::Black),
                    b'O' => board.place_unchecked(position, Stone::White),
                    b'.' => {}
                    other => bail!(
                        "line {line_number}: unexpected stone {:?}",
                        char::from(other)
                    ),
                }
            }

            // Wins and losses are worth 1e8 points; the cap keeps them from
            // drowning the differences between quiet moves.
            let clamped = |score: f32| score.clamp(-cap, cap);
            let clamped_scores = || record.scores.iter().flatten().map(|&score| clamped(score));
            let best_score = clamped_scores().fold(f32::NEG_INFINITY, f32::max);
            let worst_score = clamped_scores().fold(f32::INFINITY, f32::min);
            // A board where every move ties says nothing about a strategy.
            if worst_score >= best_score {
                tied_boards += 1;
                continue;
            }
            let mut shortfall = PositionArray::new(None);
            for (position, score) in PositionId::iter().zip(&record.scores) {
                *shortfall.get_mut(position) =
                    score.map(|score| (best_score - clamped(score)) / cap);
            }
            boards.push(ScoredBoard { board, shortfall });
        }
        ensure!(!boards.is_empty(), "no scored boards with a move to prefer");
        info!(boards = boards.len(), tied_boards, "Read scored boards");

        Ok(Self {
            boards,
            boards_per_generation,
            rank_decay,
            cap,
        })
    }
}

impl EvaluateFitness for ScoredBoardFitness {
    #[instrument(name = "ScoredBoardFitness", skip_all)]
    fn evaluate<S: EvolvableStrategy>(
        &self,
        strategies: &[S],
        rng: &mut fastrand::Rng,
    ) -> Vec<FitnessScore> {
        // Every strategy sees the same boards; the next generation draws afresh.
        let sampled_boards = rng.choose_multiple(&self.boards, self.boards_per_generation);
        let sampled_board_count = f32::from(
            u16::try_from(sampled_boards.len()).expect("boards per generation fit in u16"),
        );

        // Counted as floats so the rates below need no conversions.
        let mut picks = 0.0_f32;
        let mut pick_shortfall_sum = 0.0_f32;
        let mut blunders = 0.0_f32;
        let mut best_picks = 0.0_f32;

        let scores = strategies
            .iter()
            .map(|strategy| {
                let mut error_sum = 0.0;
                for scored_board in &sampled_boards {
                    let ranked_positions = strategy.rank_positions(
                        scored_board.board.stone_to_move(),
                        &scored_board.board,
                        rng,
                    );

                    // Each rank weighs `rank_decay` times the one before it,
                    // so a decay of 0 counts only the pick.
                    let mut weight = 1.0;
                    let mut weighted_shortfall_sum = 0.0;
                    let mut weight_sum = 0.0;
                    for &position in &ranked_positions {
                        let shortfall = scored_board
                            .shortfall
                            .get(position)
                            .expect("ranked positions are empty");
                        weighted_shortfall_sum += weight * shortfall;
                        weight_sum += weight;
                        weight *= self.rank_decay;
                    }
                    error_sum += weighted_shortfall_sum / weight_sum;

                    let pick_shortfall = scored_board
                        .shortfall
                        .get(ranked_positions[0])
                        .expect("ranked positions are empty");
                    picks += 1.0;
                    pick_shortfall_sum += pick_shortfall;
                    if pick_shortfall >= BLUNDER_SHORTFALL {
                        blunders += 1.0;
                    }
                    if pick_shortfall == 0.0 {
                        best_picks += 1.0;
                    }
                }
                FitnessScore::new(-error_sum / sampled_board_count)
            })
            .collect();

        info!(
            mean_pick_shortfall_points = pick_shortfall_sum / picks * self.cap,
            blunder_rate = blunders / picks,
            best_pick_rate = best_picks / picks,
            "Scored boards"
        );
        scores
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::Strategy;
    use crate::test_utils::TestGenes;

    /// Ranks the listed positions first, in order, and every other empty
    /// position after them in index order, whatever the board.
    struct FixedRanking(Vec<usize>);

    impl Strategy for FixedRanking {
        fn choose_move(&self, _: Stone, _: &Board, _: &mut fastrand::Rng) -> PositionId {
            unreachable!("the scored-board evaluator ranks, it does not play")
        }

        fn label(&self) -> &'static str {
            "fixed"
        }
    }

    impl EvolvableStrategy for FixedRanking {
        type Genes = TestGenes;

        fn random(_: impl Into<String>, _: &mut fastrand::Rng) -> Self {
            unreachable!()
        }

        fn genes(&self) -> &Self::Genes {
            unreachable!()
        }

        fn from_genes(_: impl Into<String>, _: Self::Genes) -> Self {
            unreachable!()
        }

        fn rank_positions(
            &self,
            _: Stone,
            board: &Board,
            _: &mut fastrand::Rng,
        ) -> Vec<PositionId> {
            let mut ranked_positions: Vec<PositionId> = self
                .0
                .iter()
                .map(|&index| PositionId::from_index(index))
                .collect();
            ranked_positions.extend(
                board
                    .empty_position_ids()
                    .into_iter()
                    .filter(|position| !self.0.contains(&position.to_index())),
            );
            ranked_positions
        }
    }

    /// One line of the file: `stones` as (index, symbol), `default_score`
    /// for every other position, overridden by `scores`.
    fn line(stones: &[(usize, char)], default_score: i32, scores: &[(usize, i32)]) -> String {
        let mut symbols = vec!['.'; PositionId::COUNT];
        let mut points = vec![Some(default_score); PositionId::COUNT];
        for &(index, symbol) in stones {
            symbols[index] = symbol;
            points[index] = None;
        }
        for &(index, score) in scores {
            points[index] = Some(score);
        }
        let stones: String = symbols.into_iter().collect();
        format!(
            "{{\"stones\":{:?},\"scores\":{}}}",
            stones,
            serde_json::to_string(&points).unwrap()
        )
    }

    /// Black at 0, White at 1, Black to move. Position 2 is best, 3 is 200
    /// points behind, 4 is a loss, and everything else is 500 behind.
    fn one_board() -> String {
        line(
            &[(0, 'X'), (1, 'O')],
            0,
            &[(2, 500), (3, 300), (4, -2_000_000)],
        )
    }

    fn evaluator(text: &str, rank_decay: f32) -> ScoredBoardFitness {
        ScoredBoardFitness::read(text.as_bytes(), 1000.0, 100, rank_decay).unwrap()
    }

    fn fitness(evaluator: &ScoredBoardFitness, strategy: FixedRanking) -> f32 {
        let mut rng = fastrand::Rng::with_seed(42);
        evaluator.evaluate(&[strategy], &mut rng)[0].value()
    }

    #[test]
    fn reads_the_board_and_the_shortfalls_from_a_record() {
        let evaluator = evaluator(&one_board(), 0.1);

        let scored_board = &evaluator.boards[0];
        assert_eq!(
            scored_board.board.stone(PositionId::from_index(0)),
            Some(Stone::Black)
        );
        assert_eq!(
            scored_board.board.stone(PositionId::from_index(1)),
            Some(Stone::White)
        );
        assert_eq!(scored_board.board.stone_to_move(), Stone::Black);
        assert_eq!(*scored_board.shortfall.get(PositionId::from_index(0)), None);
        assert_eq!(
            *scored_board.shortfall.get(PositionId::from_index(2)),
            Some(0.0)
        );
        assert_eq!(
            *scored_board.shortfall.get(PositionId::from_index(3)),
            Some(0.2)
        );
        assert_eq!(
            *scored_board.shortfall.get(PositionId::from_index(4)),
            Some(1.5)
        );
        assert_eq!(
            *scored_board.shortfall.get(PositionId::from_index(5)),
            Some(0.5)
        );
    }

    #[test]
    fn drops_boards_where_every_move_ties() {
        let all_wins = line(&[(0, 'X')], 99_999_990, &[(5, 99_999_992)]);
        let text = format!("{}\n{all_wins}\n", one_board());

        let evaluator = evaluator(&text, 0.1);

        assert_eq!(evaluator.boards.len(), 1);
    }

    #[test]
    fn rejects_a_file_with_no_board_to_learn_from() {
        let all_tied = line(&[(0, 'X')], 7, &[]);

        let error = ScoredBoardFitness::read(all_tied.as_bytes(), 1000.0, 100, 0.1)
            .err()
            .unwrap();

        assert!(error.to_string().contains("no scored boards"), "{error}");
    }

    #[test]
    fn rejects_a_record_with_the_wrong_number_of_scores() {
        let one_score_short = one_board().replace(",0]}", "]}");

        let error = ScoredBoardFitness::read(one_score_short.as_bytes(), 1000.0, 100, 0.1)
            .err()
            .unwrap();

        assert!(error.to_string().contains("expected 225"), "{error}");
    }

    #[test]
    fn rejects_an_unknown_stone_symbol() {
        let odd_symbol = one_board().replacen('.', "?", 1);

        let error = ScoredBoardFitness::read(odd_symbol.as_bytes(), 1000.0, 100, 0.1)
            .err()
            .unwrap();

        assert!(error.to_string().contains("unexpected stone"), "{error}");
    }

    #[test]
    fn rejects_a_cap_that_is_not_positive() {
        let error = ScoredBoardFitness::read(one_board().as_bytes(), 0.0, 100, 0.1)
            .err()
            .unwrap();

        assert!(error.to_string().contains("cap"), "{error}");
    }

    #[test]
    fn the_ranking_minimax_prefers_has_the_least_error_at_any_decay() {
        // At a decay of 1 every rank weighs the same and every ranking ties.
        for rank_decay in [0.0, 0.1, 0.5, 0.99] {
            let evaluator = evaluator(&one_board(), rank_decay);

            let perfect = fitness(&evaluator, FixedRanking(vec![2, 3]));
            let swapped = fitness(&evaluator, FixedRanking(vec![3, 2]));
            let losing_first = fitness(&evaluator, FixedRanking(vec![4, 2, 3]));

            assert!(
                perfect >= swapped,
                "decay {rank_decay}: {perfect} < {swapped}"
            );
            assert!(
                swapped > losing_first,
                "decay {rank_decay}: {swapped} <= {losing_first}"
            );
        }
    }

    #[test]
    fn a_losing_pick_scores_worse_than_the_best_pick() {
        let evaluator = evaluator(&one_board(), 0.1);

        let best_first = fitness(&evaluator, FixedRanking(vec![2]));
        let losing_first = fitness(&evaluator, FixedRanking(vec![4]));

        assert!(best_first > losing_first, "{best_first} <= {losing_first}");
    }

    #[test]
    fn decay_zero_counts_only_the_pick() {
        let evaluator = evaluator(&one_board(), 0.0);

        for (first_ranked, expected_fitness) in [(2, 0.0), (3, -0.2), (4, -1.5)] {
            let fitness = fitness(&evaluator, FixedRanking(vec![first_ranked]));

            assert!(
                (fitness - expected_fitness).abs() < f32::EPSILON,
                "position {first_ranked} first: {fitness} != {expected_fitness}"
            );
        }
    }

    #[test]
    fn decay_weights_later_ranks_less() {
        let evaluator = evaluator(&one_board(), 0.1);

        // Position 3 first (0.2), then 2 (0.0), then 4 (1.5), then 0.5 forever.
        let error = -fitness(&evaluator, FixedRanking(vec![3, 2, 4]));

        // Weights 1, 0.1, 0.01, 0.001, ... sum to 1 / 0.9.
        let expected = (0.2 + 0.01 * 1.5 + 0.5 * 0.001 / 0.9) / (1.0 / 0.9);
        assert!((error - expected).abs() < 1e-5, "{error} != {expected}");
    }

    #[test]
    fn every_strategy_in_one_evaluation_sees_the_same_boards() {
        // Three boards on which picking position 3 costs 0.0, 0.2 and 1.0.
        let text = [
            line(&[(0, 'X'), (1, 'O')], 0, &[(3, 500)]),
            line(&[(0, 'X'), (1, 'O')], 0, &[(2, 500), (3, 300)]),
            line(&[(0, 'X'), (1, 'O')], 0, &[(2, 500), (3, -500)]),
        ]
        .join("\n");
        let evaluator = ScoredBoardFitness::read(text.as_bytes(), 1000.0, 1, 0.0).unwrap();
        let mut rng = fastrand::Rng::with_seed(42);

        let scores = evaluator.evaluate(&[FixedRanking(vec![3]), FixedRanking(vec![3])], &mut rng);

        assert_eq!(scores[0], scores[1]);
    }
}
