//! Shared test utilities for unit tests.

use std::cell::Cell;

use crate::board::Board;
use crate::conv::{ConvTiny, ConvWeights};
use crate::evolution::fitness_score::FitnessScore;
use crate::evolution::selection::HasFitness;
use crate::position::Position;
use crate::position_id::PositionId;
use crate::stone::Stone;
use crate::strategy::{EvolvableStrategy, Strategy};

/// A test individual with a fitness value for selection tests.
pub struct TestIndividual {
    pub fitness: FitnessScore,
}

impl HasFitness for TestIndividual {
    fn fitness(&self) -> FitnessScore {
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

    pub fn with_positions(label: impl Into<String>, positions: &[(usize, usize)]) -> Self {
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
    fn choose_move(
        &self,
        _current_stone: Stone,
        _board: &Board,
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
    fn choose_move(
        &self,
        _current_stone: Stone,
        _board: &Board,
        _rng: &mut fastrand::Rng,
    ) -> PositionId {
        panic!("StubStrategy::choose_move should not be called in tournament tests")
    }

    fn label(&self) -> &str {
        &self.label
    }
}

/// A fake evolvable strategy for evolution tests.
///
/// Uses `ConvTiny` weights as its gene type for lightweight testing.
pub struct FakeEvolvableStrategy {
    label: String,
    weights: <ConvTiny as EvolvableStrategy>::Genes,
}

impl Strategy for FakeEvolvableStrategy {
    fn choose_move(
        &self,
        _current_stone: Stone,
        _board: &Board,
        _rng: &mut fastrand::Rng,
    ) -> PositionId {
        panic!("FakeEvolvableStrategy::choose_move should not be called")
    }

    fn label(&self) -> &str {
        &self.label
    }
}

impl EvolvableStrategy for FakeEvolvableStrategy {
    type Genes = ConvWeights<3, 32, 2, 0>;

    fn random(label: impl Into<String>, rng: &mut fastrand::Rng) -> Self {
        Self {
            label: label.into(),
            weights: ConvWeights::random(rng),
        }
    }

    fn genes(&self) -> &Self::Genes {
        &self.weights
    }

    fn from_genes(label: impl Into<String>, genes: Self::Genes) -> Self {
        Self {
            label: label.into(),
            weights: genes,
        }
    }
}
