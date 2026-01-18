//! Shared test utilities for unit tests.

use std::cell::Cell;
use std::collections::HashMap;

use crate::board::Board;
use crate::cache_id::CacheId;
use crate::cache_repository::CacheRepository;
use crate::evolution::selection::HasFitness;
use crate::match_result::MatchResult;
use crate::match_runner::RunMatch;
use crate::outcome::Outcome;
use crate::position::Position;
use crate::position_id::PositionId;
use crate::stone::Stone;
use crate::strategy::{EvolvableStrategy, Strategy};
use crate::tournament::{RunTournament, Standing};

/// A test individual with a fitness value for selection tests.
pub struct TestIndividual {
    pub fitness: u32,
}

impl HasFitness for TestIndividual {
    fn fitness(&self) -> u32 {
        self.fitness
    }
}

/// A test strategy that plays predetermined moves.
pub struct ScriptedStrategy {
    label: String,
    moves: Vec<PositionId>,
    index: Cell<usize>,
}

impl ScriptedStrategy {
    pub fn new(label: impl Into<String>, moves: Vec<PositionId>) -> Self {
        Self {
            label: label.into(),
            moves,
            index: Cell::new(0),
        }
    }

    pub fn with_positions(label: impl Into<String>, positions: &[(u8, u8)]) -> Self {
        let moves = positions
            .iter()
            .map(|&(row, col)| PositionId::from_position(Position::new(row, col)))
            .collect();
        Self::new(label, moves)
    }

    /// Creates strategies where black wins (5 in a row horizontally on row 0).
    pub fn black_wins() -> (Self, Self) {
        // Black plays (0,0), (0,1), (0,2), (0,3), (0,4) - horizontal win
        let black = Self::with_positions("black", &[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]);
        // White plays (1,0), (1,1), (1,2), (1,3) - no win
        let white = Self::with_positions("white", &[(1, 0), (1, 1), (1, 2), (1, 3)]);
        (black, white)
    }

    /// Creates strategies where white wins (5 in a row horizontally on row 1).
    pub fn white_wins() -> (Self, Self) {
        // Black plays scattered positions - no win
        let black = Self::with_positions("black", &[(0, 0), (0, 2), (0, 4), (0, 6), (0, 8)]);
        // White plays (1,0), (1,1), (1,2), (1,3), (1,4) - horizontal win
        let white = Self::with_positions("white", &[(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]);
        (black, white)
    }

    /// Creates strategies that result in a draw.
    pub fn draw() -> (Self, Self) {
        let moves = draw_moves();
        let black_positions: Vec<_> = moves
            .iter()
            .filter(|(_, s)| *s == Stone::Black)
            .map(|(p, _)| *p)
            .collect();
        let white_positions: Vec<_> = moves
            .iter()
            .filter(|(_, s)| *s == Stone::White)
            .map(|(p, _)| *p)
            .collect();
        let black = Self::new("black", black_positions);
        let white = Self::new("white", white_positions);
        (black, white)
    }
}

impl Strategy for ScriptedStrategy {
    fn cache_dependencies(&self) -> &[CacheId] {
        &[]
    }

    fn choose_move(
        &self,
        _current_stone: Stone,
        _board: &Board,
        _cache_repo: &CacheRepository,
        _rng: &mut fastrand::Rng,
    ) -> PositionId {
        let idx = self.index.get();
        let position = self.moves[idx];
        self.index.set(idx + 1);
        position
    }

    fn label(&self) -> &str {
        &self.label
    }
}

/// Generates moves for a draw game using a modified checkerboard pattern.
///
/// Returns positions with their stone colors in play order (black first, alternating).
/// The pattern flips every 4 rows to break diagonal 5-in-a-row sequences.
pub fn draw_moves() -> Vec<(PositionId, Stone)> {
    let mut black = Vec::new();
    let mut white = Vec::new();

    for position_id in PositionId::iter() {
        let pos = position_id.position();
        let row = pos.row();
        let col = pos.col();
        let checkerboard = (row + col) % 2 == 0;
        let flip_band = (row / 4) % 2 == 1;

        if checkerboard ^ flip_band {
            white.push(position_id);
        } else {
            black.push(position_id);
        }
    }

    let mut moves = Vec::with_capacity(225);
    let mut black_iter = black.into_iter();
    let mut white_iter = white.into_iter();

    while let Some(b) = black_iter.next() {
        moves.push((b, Stone::Black));
        if let Some(w) = white_iter.next() {
            moves.push((w, Stone::White));
        }
    }

    moves
}

/// Specifies the winner of a scripted match.
#[derive(Clone)]
pub enum Winner {
    /// The strategy with this label wins.
    Label(&'static str),
    /// The match ends in a draw.
    Draw,
}

/// A stub strategy for tournament testing (doesn't play actual moves).
pub struct StubStrategy {
    label: String,
}

impl StubStrategy {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
        }
    }
}

impl Strategy for StubStrategy {
    fn cache_dependencies(&self) -> &[CacheId] {
        &[]
    }

    fn choose_move(
        &self,
        _current_stone: Stone,
        _board: &Board,
        _cache_repo: &CacheRepository,
        _rng: &mut fastrand::Rng,
    ) -> PositionId {
        panic!("StubStrategy::choose_move should not be called in tournament tests")
    }

    fn label(&self) -> &str {
        &self.label
    }
}

/// A test match runner that returns predetermined outcomes.
pub struct ScriptedMatchRunner {
    outcomes: HashMap<(String, String), Winner>,
}

impl ScriptedMatchRunner {
    pub fn new() -> Self {
        Self {
            outcomes: HashMap::new(),
        }
    }

    /// Adds a match outcome. The labels are sorted internally for lookup.
    pub fn add(mut self, label1: &'static str, label2: &'static str, winner: Winner) -> Self {
        let key = Self::make_key(label1, label2);
        self.outcomes.insert(key, winner);
        self
    }

    fn make_key(label1: &str, label2: &str) -> (String, String) {
        let mut labels = [label1.to_string(), label2.to_string()];
        labels.sort();
        (labels[0].clone(), labels[1].clone())
    }

    fn determine_outcome(winner: &Winner, black_label: &str, white_label: &str) -> Outcome {
        match winner {
            Winner::Draw => Outcome::Draw,
            Winner::Label(label) if *label == black_label => Outcome::BlackWins,
            Winner::Label(label) if *label == white_label => Outcome::WhiteWins,
            Winner::Label(label) => panic!("Invalid winner label: {label}"),
        }
    }
}

impl Default for ScriptedMatchRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl RunMatch for ScriptedMatchRunner {
    fn run_match(
        &self,
        black_strategy: &dyn Strategy,
        white_strategy: &dyn Strategy,
        _rng: &mut fastrand::Rng,
    ) -> MatchResult {
        let black_label = black_strategy.label();
        let white_label = white_strategy.label();
        let key = Self::make_key(black_label, white_label);

        let winner = self
            .outcomes
            .get(&key)
            .unwrap_or_else(|| panic!("No outcome defined for {key:?}"));

        let outcome = Self::determine_outcome(winner, black_label, white_label);

        MatchResult::new(
            outcome,
            black_label.to_string(),
            white_label.to_string(),
            0,
            String::new(),
        )
    }
}

/// A test tournament that returns standings in a predetermined order.
pub struct ScriptedTournament {
    ranking: Vec<&'static str>,
}

impl ScriptedTournament {
    pub fn new(ranking: Vec<&'static str>) -> Self {
        Self { ranking }
    }
}

impl RunTournament for ScriptedTournament {
    fn run<S: Strategy>(&self, strategies: &[S], _rng: &mut fastrand::Rng) -> Vec<Standing> {
        self.ranking
            .iter()
            .map(|&label| {
                let index = strategies
                    .iter()
                    .position(|s| s.label() == label)
                    .unwrap_or_else(|| panic!("Strategy with label '{label}' not found"));
                Standing::new(index, 0, 0, 0, 0)
            })
            .collect()
    }
}

/// A test tournament that returns standings in input order (first strategy ranks first).
pub struct InputOrderTournament;

impl RunTournament for InputOrderTournament {
    fn run<S: Strategy>(&self, strategies: &[S], _rng: &mut fastrand::Rng) -> Vec<Standing> {
        (0..strategies.len())
            .map(|i| Standing::new(i, 0, 0, 0, 0))
            .collect()
    }
}

/// A fake evolvable strategy for evolution tests.
pub struct FakeEvolvableStrategy {
    label: String,
    genes: Vec<u8>,
}

impl FakeEvolvableStrategy {
    pub fn new(label: impl Into<String>, genes: Vec<u8>) -> Self {
        Self {
            label: label.into(),
            genes,
        }
    }
}

impl Strategy for FakeEvolvableStrategy {
    fn cache_dependencies(&self) -> &[CacheId] {
        &[]
    }

    fn choose_move(
        &self,
        _current_stone: Stone,
        _board: &Board,
        _cache_repo: &CacheRepository,
        _rng: &mut fastrand::Rng,
    ) -> PositionId {
        panic!("FakeEvolvableStrategy::choose_move should not be called")
    }

    fn label(&self) -> &str {
        &self.label
    }
}

impl EvolvableStrategy for FakeEvolvableStrategy {
    type Gene = u8;

    fn random_genes(rng: &mut fastrand::Rng) -> Vec<Self::Gene> {
        let mut genes = vec![1, 2, 3, 4, 5];
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
