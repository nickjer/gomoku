//! Threat detection and defensive capability testing.
//!
//! This module provides tools to test whether strategies can recognize and block
//! immediate threats:
//! - 4 in a row with at least one open end (must block or lose)
//! - Open three: 3 in a row with both ends open (must block or opponent gets open four)

use crate::board::Board;
use crate::offset::Offset;
use crate::position::Position;
use crate::position_id::PositionId;
use crate::stone::Stone;
use crate::strategy::Strategy;

/// Maximum number of stones in a threat scenario (4 opponent + 4 player filler).
const MAX_PLACED_STONES: usize = 8;

/// The four directions to check for lines.
const DIRECTIONS: [Offset; 4] = [
    Offset::new(0, 1),  // horizontal
    Offset::new(1, 0),  // vertical
    Offset::new(1, 1),  // diagonal down-right
    Offset::new(1, -1), // diagonal down-left
];

/// A threat scenario: a board position where the opponent has a dangerous
/// formation that must be blocked to prevent losing.
#[derive(Debug, Clone)]
pub struct ThreatScenario {
    placed_stones: Vec<(PositionId, Stone)>,
    threat_positions: Vec<PositionId>,
    player_to_move: Stone,
}

impl ThreatScenario {
    /// Creates a threat scenario with the given placed stones and blocking positions.
    fn new(
        placed_stones: Vec<(PositionId, Stone)>,
        threat_positions: Vec<PositionId>,
        player_to_move: Stone,
    ) -> Self {
        Self {
            placed_stones,
            threat_positions,
            player_to_move,
        }
    }

    /// Returns true if the given move blocks the threat.
    #[must_use]
    pub fn is_blocking_move(&self, position: PositionId) -> bool {
        self.threat_positions.contains(&position)
    }
}

/// Generates a set of threat scenarios for testing defensive capability.
pub fn generate_threat_scenarios(rng: &mut fastrand::Rng) -> Vec<ThreatScenario> {
    let mut scenarios = Vec::new();

    // Generate 4-in-a-row threats
    for _ in 0..10 {
        if let Some(scenario) = generate_four_in_a_row_threat(rng, Stone::Black) {
            scenarios.push(scenario);
        }
        if let Some(scenario) = generate_four_in_a_row_threat(rng, Stone::White) {
            scenarios.push(scenario);
        }
    }

    // Generate open three threats (3-in-a-row with both ends open)
    for _ in 0..10 {
        if let Some(scenario) = generate_open_three_threat(rng, Stone::Black) {
            scenarios.push(scenario);
        }
        if let Some(scenario) = generate_open_three_threat(rng, Stone::White) {
            scenarios.push(scenario);
        }
    }

    scenarios
}

/// Generates a threat scenario where opponent has 4 in a row.
fn generate_four_in_a_row_threat(
    rng: &mut fastrand::Rng,
    player_to_move: Stone,
) -> Option<ThreatScenario> {
    let opponent = player_to_move.opponent();
    let direction = DIRECTIONS[rng.usize(..DIRECTIONS.len())];

    // Pick a random starting position that allows 4 stones + blocking positions
    let start_row = rng.usize(2..12);
    let start_col = rng.usize(2..12);
    let start = PositionId::from_position(Position::new(start_row, start_col));

    // Build the line of 4 opponent stones
    let mut placed_stones = Vec::with_capacity(MAX_PLACED_STONES);
    let mut current = start;

    placed_stones.push((start, opponent));
    for _ in 0..3 {
        current = current.from_offset(direction)?;
        placed_stones.push((current, opponent));
    }

    // Find blocking positions (open ends of the line)
    let mut threat_positions = Vec::new();

    if let Some(before) = start.from_offset(-direction) {
        threat_positions.push(before);
    }
    if let Some(after) = current.from_offset(direction) {
        threat_positions.push(after);
    }

    if threat_positions.is_empty() {
        return None;
    }

    // Add some random player stones to make the board more realistic
    add_filler_stones(rng, &mut placed_stones, &threat_positions, player_to_move);

    Some(ThreatScenario::new(
        placed_stones,
        threat_positions,
        player_to_move,
    ))
}

/// Generates an open three threat: 3 in a row with both ends open.
/// This is dangerous because the opponent will get an open four next turn.
fn generate_open_three_threat(
    rng: &mut fastrand::Rng,
    player_to_move: Stone,
) -> Option<ThreatScenario> {
    let opponent = player_to_move.opponent();
    let direction = DIRECTIONS[rng.usize(..DIRECTIONS.len())];

    // Pick a starting position with room for stones + both open ends
    let start_row = rng.usize(2..12);
    let start_col = rng.usize(2..12);
    let start = PositionId::from_position(Position::new(start_row, start_col));

    // Build the line of 3 opponent stones
    let mut placed_stones = Vec::with_capacity(MAX_PLACED_STONES);
    let mut current = start;

    placed_stones.push((start, opponent));
    for _ in 0..2 {
        current = current.from_offset(direction)?;
        placed_stones.push((current, opponent));
    }

    // Both ends must be valid positions (on the board) for an "open three"
    let before = start.from_offset(-direction)?;
    let after = current.from_offset(direction)?;

    let threat_positions = vec![before, after];

    // Add some random player stones to make the board more realistic
    add_filler_stones(rng, &mut placed_stones, &threat_positions, player_to_move);

    Some(ThreatScenario::new(
        placed_stones,
        threat_positions,
        player_to_move,
    ))
}

/// Adds random filler stones for the player to make the scenario more realistic.
fn add_filler_stones(
    rng: &mut fastrand::Rng,
    placed_stones: &mut Vec<(PositionId, Stone)>,
    threat_positions: &[PositionId],
    stone: Stone,
) {
    let num_filler = rng.usize(2..=4);
    let mut added = 0;
    while added < num_filler {
        let pos = PositionId::from_position(Position::new(
            rng.usize(..PositionId::WIDTH),
            rng.usize(..PositionId::WIDTH),
        ));
        let already_placed = placed_stones.iter().any(|&(p, _)| p == pos);
        if !already_placed && !threat_positions.contains(&pos) {
            placed_stones.push((pos, stone));
            added += 1;
        }
    }
}

/// Tests a strategy's defensive capability by checking if it blocks threats.
/// Returns the number of threats successfully blocked.
///
/// # Panics
///
/// Panics if a scenario placement is invalid or the scenario count exceeds `u16::MAX`.
pub fn test_defense<S: Strategy>(
    strategy: &S,
    scenarios: &[ThreatScenario],
    rng: &mut fastrand::Rng,
) -> f32 {
    let blocked_count = scenarios
        .iter()
        .filter(|scenario| {
            let mut board = Board::new();
            for &(pos, stone) in &scenario.placed_stones {
                board.place(pos, stone).expect("valid scenario placement");
            }
            let chosen = strategy.choose_move(scenario.player_to_move, &board, rng);
            scenario.is_blocking_move(chosen)
        })
        .count();
    f32::from(u16::try_from(blocked_count).expect("scenario count fits in u16"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::ScriptedStrategy;

    #[test]
    fn generate_threat_scenarios_produces_scenarios() {
        let mut rng = fastrand::Rng::with_seed(42);
        let scenarios = generate_threat_scenarios(&mut rng);

        assert!(!scenarios.is_empty());
    }

    #[test]
    fn threat_scenario_identifies_blocking_move() {
        let pos = |r, c| PositionId::from_position(Position::new(r, c));

        // 4 white stones in a row: (7,5), (7,6), (7,7), (7,8)
        let placements = vec![
            (pos(7, 5), Stone::White),
            (pos(7, 6), Stone::White),
            (pos(7, 7), Stone::White),
            (pos(7, 8), Stone::White),
        ];

        // Blocking positions are (7,4) and (7,9)
        let scenario = ThreatScenario::new(placements, vec![pos(7, 4), pos(7, 9)], Stone::Black);

        assert!(scenario.is_blocking_move(pos(7, 4)));
        assert!(scenario.is_blocking_move(pos(7, 9)));
        assert!(!scenario.is_blocking_move(pos(7, 7))); // occupied
        assert!(!scenario.is_blocking_move(pos(0, 0))); // not blocking
    }

    #[test]
    fn test_defense_counts_blocked_threats() {
        let pos = |r, c| PositionId::from_position(Position::new(r, c));

        // Create a threat at row 7
        let placements = vec![
            (pos(7, 5), Stone::White),
            (pos(7, 6), Stone::White),
            (pos(7, 7), Stone::White),
            (pos(7, 8), Stone::White),
        ];

        let scenario = ThreatScenario::new(placements, vec![pos(7, 4), pos(7, 9)], Stone::Black);

        // Strategy that plays at the blocking position
        let blocking_strategy = ScriptedStrategy::with_positions("blocker", &[(7, 4)]);
        let mut rng = fastrand::Rng::with_seed(42);

        let blocks = test_defense(&blocking_strategy, &[scenario.clone()], &mut rng);
        assert_eq!(blocks, 1.0);

        // Strategy that plays elsewhere
        let non_blocking = ScriptedStrategy::with_positions("non_blocker", &[(0, 0)]);
        let blocks = test_defense(&non_blocking, &[scenario], &mut rng);
        assert_eq!(blocks, 0.0);
    }

    #[test]
    fn open_three_scenario_identifies_blocking_moves() {
        let pos = |r, c| PositionId::from_position(Position::new(r, c));

        // 3 white stones in a row: (7,5), (7,6), (7,7)
        // with both ends open at (7,4) and (7,8)
        let placements = vec![
            (pos(7, 5), Stone::White),
            (pos(7, 6), Stone::White),
            (pos(7, 7), Stone::White),
        ];

        let scenario = ThreatScenario::new(placements, vec![pos(7, 4), pos(7, 8)], Stone::Black);

        assert!(scenario.is_blocking_move(pos(7, 4)));
        assert!(scenario.is_blocking_move(pos(7, 8)));
        assert!(!scenario.is_blocking_move(pos(7, 6))); // occupied
        assert!(!scenario.is_blocking_move(pos(0, 0))); // not blocking
    }
}
