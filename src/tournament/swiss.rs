use std::collections::HashSet;

use crate::game::Play;
use crate::outcome::Outcome;
use crate::strategy::Strategy;

use super::{RunTournament, Standing};

/// Returns the number of rounds for a given number of strategies.
#[must_use]
pub fn total_rounds(count: usize) -> u32 {
    if count < 2 {
        return 0;
    }
    (count - 1).ilog2() + 1
}

/// A Swiss-system tournament runner.
#[derive(Debug, Clone, Copy, Default)]
pub struct Swiss;

impl Swiss {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    fn play_round<S: Strategy, R: Play>(
        state: &mut State,
        strategies: &[S],
        game: &R,
        rng: &mut fastrand::Rng,
    ) {
        let pairings = Self::generate_pairings(state);

        for (black_idx, white_idx) in pairings {
            if let Some(white_idx) = white_idx {
                let result = game.play(&strategies[black_idx], &strategies[white_idx], rng);
                Self::update_state(state, black_idx, white_idx, result.outcome());
            } else {
                state.award_bye(black_idx);
            }
        }
    }

    fn generate_pairings(state: &State) -> Vec<(usize, Option<usize>)> {
        let sorted_indices = state.rankings();
        let mut paired = HashSet::new();
        let mut pairings = Vec::new();

        for &idx in &sorted_indices {
            if paired.contains(&idx) {
                continue;
            }

            let opponent_idx = sorted_indices.iter().copied().find(|&candidate| {
                candidate != idx
                    && !paired.contains(&candidate)
                    && !state.played_against(idx, candidate)
            });

            if let Some(opp) = opponent_idx {
                pairings.push((idx, Some(opp)));
                paired.insert(idx);
                paired.insert(opp);
            } else {
                pairings.push((idx, None));
                paired.insert(idx);
            }
        }

        pairings
    }

    fn update_state(state: &mut State, black_idx: usize, white_idx: usize, outcome: Outcome) {
        match outcome {
            Outcome::BlackWins => state.award_win(black_idx, white_idx),
            Outcome::WhiteWins => state.award_win(white_idx, black_idx),
            Outcome::Draw => state.award_draw(black_idx, white_idx),
        }
    }
}

impl RunTournament for Swiss {
    /// Runs a Swiss tournament with the given strategies.
    ///
    /// Returns standings sorted by ranking (best first).
    fn run<S: Strategy, R: Play>(
        &self,
        strategies: &[S],
        game: &R,
        rng: &mut fastrand::Rng,
    ) -> Vec<Standing> {
        let mut state = State::new(strategies.len());

        for _ in 0..total_rounds(strategies.len()) {
            Self::play_round(&mut state, strategies, game, rng);
        }

        state.build_standings()
    }
}

/// Internal state for tracking tournament progress.
struct State {
    size: usize,
    points: Vec<u32>,
    wins: Vec<u32>,
    losses: Vec<u32>,
    draws: Vec<u32>,
    byes: Vec<u32>,
    opponents_played: Vec<HashSet<usize>>,
}

impl State {
    fn new(size: usize) -> Self {
        Self {
            size,
            points: vec![0; size],
            wins: vec![0; size],
            losses: vec![0; size],
            draws: vec![0; size],
            byes: vec![0; size],
            opponents_played: (0..size).map(|_| HashSet::new()).collect(),
        }
    }

    fn award_win(&mut self, winner_idx: usize, loser_idx: usize) {
        self.points[winner_idx] += 2;
        self.wins[winner_idx] += 1;
        self.losses[loser_idx] += 1;
        self.record_opponents(winner_idx, loser_idx);
    }

    fn award_draw(&mut self, black_idx: usize, white_idx: usize) {
        self.points[black_idx] += 1;
        self.points[white_idx] += 1;
        self.draws[black_idx] += 1;
        self.draws[white_idx] += 1;
        self.record_opponents(black_idx, white_idx);
    }

    fn award_bye(&mut self, idx: usize) {
        self.points[idx] += 2;
        self.byes[idx] += 1;
    }

    fn played_against(&self, idx: usize, opponent_idx: usize) -> bool {
        self.opponents_played[idx].contains(&opponent_idx)
    }

    fn rankings(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.size).collect();
        indices.sort_by(|&a, &b| {
            let points_cmp = self.points[b].cmp(&self.points[a]);
            if points_cmp != std::cmp::Ordering::Equal {
                return points_cmp;
            }
            self.buchholz_score(b).cmp(&self.buchholz_score(a))
        });
        indices
    }

    fn build_standings(&self) -> Vec<Standing> {
        self.rankings()
            .into_iter()
            .map(|idx| {
                Standing::new(
                    idx,
                    self.wins[idx],
                    self.losses[idx],
                    self.draws[idx],
                    self.byes[idx],
                )
            })
            .collect()
    }

    fn record_opponents(&mut self, idx1: usize, idx2: usize) {
        self.opponents_played[idx1].insert(idx2);
        self.opponents_played[idx2].insert(idx1);
    }

    fn buchholz_score(&self, idx: usize) -> u32 {
        self.opponents_played[idx]
            .iter()
            .map(|&opp_idx| self.points[opp_idx])
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{ScriptedGame, StubStrategy, Winner};

    fn make_strategies(labels: &[&str]) -> Vec<StubStrategy> {
        labels
            .iter()
            .map(|&label| StubStrategy::new(label))
            .collect()
    }

    fn labels<'a>(standings: &[Standing], strategies: &'a [StubStrategy]) -> Vec<&'a str> {
        standings
            .iter()
            .map(|s| strategies[s.strategy_index()].label())
            .collect()
    }

    #[test]
    fn total_rounds_for_two_strategies() {
        assert_eq!(total_rounds(2), 1);
    }

    #[test]
    fn total_rounds_for_four_strategies() {
        assert_eq!(total_rounds(4), 2);
    }

    #[test]
    fn total_rounds_for_eight_strategies() {
        assert_eq!(total_rounds(8), 3);
    }

    #[test]
    fn total_rounds_for_single_strategy() {
        assert_eq!(total_rounds(1), 0);
    }

    #[test]
    fn total_rounds_for_zero_strategies() {
        assert_eq!(total_rounds(0), 0);
    }

    #[test]
    fn run_returns_standings() {
        let game = ScriptedGame::new().add("a", "b", Winner::Label("a"));
        let tournament = Swiss::new();
        let strategies = make_strategies(&["a", "b"]);
        let mut rng = fastrand::Rng::new();

        let standings = tournament.run(&strategies, &game, &mut rng);

        assert_eq!(standings.len(), 2);
    }

    #[test]
    fn winner_ranks_first() {
        let game = ScriptedGame::new().add("a", "b", Winner::Label("a"));
        let tournament = Swiss::new();
        let strategies = make_strategies(&["a", "b"]);
        let mut rng = fastrand::Rng::new();

        let standings = tournament.run(&strategies, &game, &mut rng);

        assert_eq!(labels(&standings, &strategies), vec!["a", "b"]);
    }

    #[test]
    fn loser_ranks_last() {
        let game = ScriptedGame::new().add("a", "b", Winner::Label("b"));
        let tournament = Swiss::new();
        let strategies = make_strategies(&["a", "b"]);
        let mut rng = fastrand::Rng::new();

        let standings = tournament.run(&strategies, &game, &mut rng);

        assert_eq!(labels(&standings, &strategies), vec!["b", "a"]);
    }

    #[test]
    fn draw_keeps_original_order() {
        let game = ScriptedGame::new().add("a", "b", Winner::Draw);
        let tournament = Swiss::new();
        let strategies = make_strategies(&["a", "b"]);
        let mut rng = fastrand::Rng::new();

        let standings = tournament.run(&strategies, &game, &mut rng);

        assert_eq!(labels(&standings, &strategies), vec!["a", "b"]);
    }

    #[test]
    fn bye_awarded_with_odd_strategies() {
        // 3 strategies, 2 rounds
        // Round 1: a vs b (a wins), c gets bye (2 points)
        // Round 2: a vs c (a wins), b gets bye (2 points)
        // Final: a=4, b=2, c=2
        let game =
            ScriptedGame::new()
                .add("a", "b", Winner::Label("a"))
                .add("a", "c", Winner::Label("a"));
        let tournament = Swiss::new();
        let strategies = make_strategies(&["a", "b", "c"]);
        let mut rng = fastrand::Rng::new();

        let standings = tournament.run(&strategies, &game, &mut rng);

        assert_eq!(labels(&standings, &strategies), vec!["a", "b", "c"]);
    }

    #[test]
    fn four_player_tournament_rankings() {
        // Round 1: a vs b (a wins), c vs d (d wins)
        // Points after R1: a=2, d=2, b=0, c=0
        // Round 2: a vs d (draw), b vs c (b wins)
        // Final points: a=3, d=3, b=2, c=0
        // Buchholz: a played b(2)+d(3)=5, d played c(0)+a(3)=3
        let game = ScriptedGame::new()
            .add("a", "b", Winner::Label("a"))
            .add("c", "d", Winner::Label("d"))
            .add("a", "d", Winner::Draw)
            .add("b", "c", Winner::Label("b"));
        let tournament = Swiss::new();
        let strategies = make_strategies(&["a", "b", "c", "d"]);
        let mut rng = fastrand::Rng::new();

        let standings = tournament.run(&strategies, &game, &mut rng);

        assert_eq!(labels(&standings, &strategies), vec!["a", "d", "b", "c"]);
    }

    #[test]
    fn buchholz_tiebreaker() {
        // Round 1: a vs b (a wins), c vs d (c wins)
        // Points after R1: a=2, c=2, b=0, d=0
        // Round 2: a vs c (c wins), b vs d (b wins)
        // Final points: c=4, a=2, b=2, d=0
        // Buchholz for a: b(2) + c(4) = 6
        // Buchholz for b: a(2) + d(0) = 2
        let game = ScriptedGame::new()
            .add("a", "b", Winner::Label("a"))
            .add("c", "d", Winner::Label("c"))
            .add("a", "c", Winner::Label("c"))
            .add("b", "d", Winner::Label("b"));
        let tournament = Swiss::new();
        let strategies = make_strategies(&["a", "b", "c", "d"]);
        let mut rng = fastrand::Rng::new();

        let standings = tournament.run(&strategies, &game, &mut rng);

        assert_eq!(labels(&standings, &strategies), vec!["c", "a", "b", "d"]);
    }
}
