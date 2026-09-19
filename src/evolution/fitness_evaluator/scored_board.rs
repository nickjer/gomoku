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
    largest_estimate: f32,
}

/// A board with, for every empty position, how far its value falls behind
/// the board's best position, as a share of the file's largest estimate.
struct ScoredBoard {
    board: Board,
    shortfall: PositionArray<Option<f32>>,
}

/// One line of the scored-board file: every position as `.`, `X` or `O`,
/// every position's score for the side to move (`null` under a stone), and
/// the search depth the scores came from.
#[derive(Deserialize)]
struct ScoredBoardRecord {
    stones: String,
    scores: Vec<Option<CellScore>>,
    depth: u32,
}

/// One empty position's score: minimax's estimate in points, or the game
/// end it saw there and after how many more stones. Written as a bare
/// integer, `{"win": n}` or `{"loss": n}`.
#[derive(Deserialize, Clone, Copy)]
#[serde(untagged)]
enum CellScore {
    Points(f32),
    Win { win: u16 },
    Loss { loss: u16 },
}

/// An estimate this large can only be a raw win or loss score from an old file.
const LARGEST_ESTIMATE: f32 = 10_000_000.0;

/// A pick at least this far behind the best is counted as a blunder: a
/// losing move on a quiet board costs about the largest estimate.
const BLUNDER_SHORTFALL: f32 = 0.9;

impl ScoredBoardFitness {
    /// Reads scored boards from JSON Lines and gives every position a value:
    /// an estimate is its points, and a win or loss after `n` more stones is
    /// worth the largest estimate in the file times
    /// `1 + nearness_weight * 2 / n`, so every win ranks above every
    /// estimate and every loss below. Shortfalls are taken from the values,
    /// and boards where every move ties are dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if the nearness weight is negative, a line does not
    /// parse or does not describe a whole board, a score is a raw win or
    /// loss instead of a spelled-out game end, no estimate is nonzero, or no
    /// board is left.
    #[expect(
        clippy::too_many_lines,
        reason = "parsing the lines and then valuing them reads better than helpers with one caller"
    )]
    pub fn read(
        reader: impl BufRead,
        nearness_weight: f32,
        boards_per_generation: usize,
        rank_decay: f32,
    ) -> Result<Self> {
        ensure!(
            nearness_weight >= 0.0,
            "the nearness weight must not be negative"
        );

        // Every board is parsed before any is valued, since a win or loss is
        // worth the largest estimate in the whole file.
        let mut records = Vec::new();
        let mut depths = Vec::new();
        let mut largest_estimate = 0.0_f32;
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
            if !depths.contains(&record.depth) {
                depths.push(record.depth);
            }

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

            for cell in record.scores.iter().flatten() {
                if let CellScore::Points(points) = *cell {
                    largest_estimate = largest_estimate.max(points.abs());
                }
            }
            records.push((board, record.scores));
        }
        ensure!(
            largest_estimate > 0.0,
            "no estimate in the scored boards to scale wins and losses by"
        );

        let mut boards = Vec::new();
        let mut tied_boards = 0;
        for (line_index, (board, scores)) in records.into_iter().enumerate() {
            let line_number = line_index + 1;

            // A five after `stones` more stones weighs
            // `1 + nearness_weight * 2 / stones` largest estimates.
            let nearness = |stones: u16| -> Result<f32> {
                ensure!(
                    stones > 0,
                    "line {line_number}: a game end comes after at least one more stone"
                );
                Ok(1.0 + nearness_weight * 2.0 / f32::from(stones))
            };
            let mut values = Vec::with_capacity(PositionId::COUNT);
            for cell in &scores {
                let value = match *cell {
                    None => None,
                    Some(CellScore::Points(points)) => {
                        ensure!(
                            points.abs() < LARGEST_ESTIMATE,
                            "line {line_number}: {points} is a raw win or loss score, \
                             not an estimate; regenerate the file with game ends spelled out"
                        );
                        Some(points)
                    }
                    Some(CellScore::Win { win: stones }) => {
                        Some(largest_estimate * nearness(stones)?)
                    }
                    Some(CellScore::Loss { loss: stones }) => {
                        Some(-largest_estimate * nearness(stones)?)
                    }
                };
                values.push(value);
            }

            let best_value = values
                .iter()
                .flatten()
                .fold(f32::NEG_INFINITY, |best, &value| best.max(value));
            let worst_value = values
                .iter()
                .flatten()
                .fold(f32::INFINITY, |worst, &value| worst.min(value));
            // A board where every move ties says nothing about a strategy.
            if worst_value >= best_value {
                tied_boards += 1;
                continue;
            }
            let mut shortfall = PositionArray::new(None);
            for (position, value) in PositionId::iter().zip(&values) {
                *shortfall.get_mut(position) =
                    value.map(|value| (best_value - value) / largest_estimate);
            }
            boards.push(ScoredBoard { board, shortfall });
        }
        ensure!(!boards.is_empty(), "no scored boards with a move to prefer");
        info!(
            boards = boards.len(),
            tied_boards,
            depths = ?depths,
            largest_estimate,
            "Read scored boards"
        );

        Ok(Self {
            boards,
            boards_per_generation,
            rank_decay,
            largest_estimate,
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
            mean_pick_shortfall_points = pick_shortfall_sum / picks * self.largest_estimate,
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
    /// (a JSON value) for every other position, overridden by `scores`.
    fn line(stones: &[(usize, char)], default_score: &str, scores: &[(usize, &str)]) -> String {
        let mut symbols = vec!['.'; PositionId::COUNT];
        let mut cells = vec![default_score; PositionId::COUNT];
        for &(index, symbol) in stones {
            symbols[index] = symbol;
            cells[index] = "null";
        }
        for &(index, score) in scores {
            cells[index] = score;
        }
        let stones: String = symbols.into_iter().collect();
        format!(
            "{{\"stones\":{stones:?},\"scores\":[{}],\"depth\":4}}",
            cells.join(",")
        )
    }

    /// Black at 0, White at 1, Black to move. Position 2 is best with the
    /// largest estimate, 1000; 3 is 200 points behind; 4 is a loss after
    /// eight stones, worth -1500; and everything else is 500 behind.
    fn one_board() -> String {
        line(
            &[(0, 'X'), (1, 'O')],
            "500",
            &[(2, "1000"), (3, "800"), (4, "{\"loss\":8}")],
        )
    }

    /// Black at 0, Black to move. Wins after 1 and 3 stones at 2 and 3, an
    /// even estimate at 4, losses after 4 and 2 stones at 5 and 6, the
    /// largest estimate, 1000, at 7, and 0 elsewhere.
    fn game_end_board() -> String {
        line(
            &[(0, 'X')],
            "0",
            &[
                (2, "{\"win\":1}"),
                (3, "{\"win\":3}"),
                (4, "0"),
                (5, "{\"loss\":4}"),
                (6, "{\"loss\":2}"),
                (7, "1000"),
            ],
        )
    }

    fn evaluator(text: &str, rank_decay: f32) -> ScoredBoardFitness {
        ScoredBoardFitness::read(text.as_bytes(), 2.0, 100, rank_decay).unwrap()
    }

    fn shortfall_at(evaluator: &ScoredBoardFitness, index: usize) -> f32 {
        evaluator.boards[0]
            .shortfall
            .get(PositionId::from_index(index))
            .unwrap()
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
            Some(2.5)
        );
        assert_eq!(
            *scored_board.shortfall.get(PositionId::from_index(5)),
            Some(0.5)
        );
    }

    #[test]
    fn a_game_end_is_worth_more_the_sooner_it_comes() {
        let evaluator = evaluator(&game_end_board(), 0.1);

        // Values with a largest estimate of 1000 and a nearness weight of 2:
        // a win after one stone 5000, after three 2333.3, the estimates 0
        // and 1000, a loss after four stones -2000, after two -3000.
        for (index, expected) in [
            (2, 0.0),
            (3, 8.0 / 3.0),
            (4, 5.0),
            (5, 7.0),
            (6, 8.0),
            (7, 4.0),
        ] {
            let shortfall = shortfall_at(&evaluator, index);
            assert!(
                (shortfall - expected).abs() < 1e-4,
                "position {index}: {shortfall} != {expected}"
            );
        }
    }

    #[test]
    fn nearness_weight_zero_weighs_every_game_end_like_the_largest_estimate() {
        let evaluator =
            ScoredBoardFitness::read(game_end_board().as_bytes(), 0.0, 100, 0.1).unwrap();

        for (index, expected) in [(2, 0.0), (3, 0.0), (4, 1.0), (5, 2.0), (6, 2.0), (7, 0.0)] {
            let shortfall = shortfall_at(&evaluator, index);
            assert!(
                (shortfall - expected).abs() < 1e-6,
                "position {index}: {shortfall} != {expected}"
            );
        }
    }

    #[test]
    fn drops_boards_where_every_move_ties() {
        let all_wins = line(&[(0, 'X')], "{\"win\":1}", &[]);
        let text = format!("{}\n{all_wins}\n", one_board());

        let evaluator = evaluator(&text, 0.1);

        assert_eq!(evaluator.boards.len(), 1);
    }

    #[test]
    fn rejects_a_file_with_no_board_to_learn_from() {
        let all_tied = line(&[(0, 'X')], "7", &[]);

        let error = ScoredBoardFitness::read(all_tied.as_bytes(), 2.0, 100, 0.1)
            .err()
            .unwrap();

        assert!(error.to_string().contains("no scored boards"), "{error}");
    }

    #[test]
    fn rejects_a_raw_win_or_loss_score() {
        let raw_loss = line(&[(0, 'X')], "0", &[(5, "-99999990")]);

        let error = ScoredBoardFitness::read(raw_loss.as_bytes(), 2.0, 100, 0.1)
            .err()
            .unwrap();

        assert!(error.to_string().contains("raw win or loss"), "{error}");
    }

    #[test]
    fn rejects_a_game_end_after_no_stones() {
        let instant_win = line(&[(0, 'X')], "0", &[(4, "100"), (5, "{\"win\":0}")]);

        let error = ScoredBoardFitness::read(instant_win.as_bytes(), 2.0, 100, 0.1)
            .err()
            .unwrap();

        assert!(
            error.to_string().contains("at least one more stone"),
            "{error}"
        );
    }

    #[test]
    fn rejects_a_record_without_a_depth() {
        let no_depth = one_board().replace(",\"depth\":4", "");

        let error = ScoredBoardFitness::read(no_depth.as_bytes(), 2.0, 100, 0.1)
            .err()
            .unwrap();

        assert!(error.to_string().contains("not a scored board"), "{error}");
    }

    #[test]
    fn rejects_a_record_with_the_wrong_number_of_scores() {
        let one_score_short = one_board().replacen(",500]", "]", 1);

        let error = ScoredBoardFitness::read(one_score_short.as_bytes(), 2.0, 100, 0.1)
            .err()
            .unwrap();

        assert!(error.to_string().contains("expected 225"), "{error}");
    }

    #[test]
    fn rejects_an_unknown_stone_symbol() {
        let odd_symbol = one_board().replacen('.', "?", 1);

        let error = ScoredBoardFitness::read(odd_symbol.as_bytes(), 2.0, 100, 0.1)
            .err()
            .unwrap();

        assert!(error.to_string().contains("unexpected stone"), "{error}");
    }

    #[test]
    fn rejects_a_file_without_an_estimate_to_scale_game_ends_by() {
        let only_game_ends = line(&[(0, 'X')], "0", &[(5, "{\"win\":1}")]);

        let error = ScoredBoardFitness::read(only_game_ends.as_bytes(), 2.0, 100, 0.1)
            .err()
            .unwrap();

        assert!(error.to_string().contains("no estimate"), "{error}");
    }

    #[test]
    fn rejects_a_negative_nearness_weight() {
        let error = ScoredBoardFitness::read(one_board().as_bytes(), -1.0, 100, 0.1)
            .err()
            .unwrap();

        assert!(error.to_string().contains("nearness weight"), "{error}");
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

        for (first_ranked, expected_fitness) in [(2, 0.0), (3, -0.2), (4, -2.5)] {
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

        // Position 3 first (0.2), then 2 (0.0), then 4 (2.5), then 0.5 forever.
        let error = -fitness(&evaluator, FixedRanking(vec![3, 2, 4]));

        // Weights 1, 0.1, 0.01, 0.001, ... sum to 1 / 0.9.
        let expected = (0.2 + 0.01 * 2.5 + 0.5 * 0.001 / 0.9) / (1.0 / 0.9);
        assert!((error - expected).abs() < 1e-5, "{error} != {expected}");
    }

    #[test]
    fn every_strategy_in_one_evaluation_sees_the_same_boards() {
        // Three boards on which picking position 3 costs 0.0, 0.4 and 2.0
        // of the largest estimate, 500.
        let text = [
            line(&[(0, 'X'), (1, 'O')], "0", &[(3, "500")]),
            line(&[(0, 'X'), (1, 'O')], "0", &[(2, "500"), (3, "300")]),
            line(&[(0, 'X'), (1, 'O')], "0", &[(2, "500"), (3, "-500")]),
        ]
        .join("\n");
        let evaluator = ScoredBoardFitness::read(text.as_bytes(), 2.0, 1, 0.0).unwrap();
        let mut rng = fastrand::Rng::with_seed(42);

        let scores = evaluator.evaluate(&[FixedRanking(vec![3]), FixedRanking(vec![3])], &mut rng);

        assert_eq!(scores[0], scores[1]);
    }
}
