use crate::position_id::PositionId;

/// Selects the best position based on fingerprint priorities.
///
/// # Arguments
/// * `empty_positions` - The available positions to choose from
/// * `fingerprint_positions` - Lookup table: `fingerprint_positions[fp_index]` = rank
/// * `get_fp_index` - Closure to get fingerprint index for a position
/// * `rng` - Random number generator for tiebreaking
///
/// # Panics
///
/// Panics if `empty_positions` is empty.
#[must_use]
pub fn select_best_position(
    empty_positions: &[PositionId],
    fingerprint_positions: &[usize],
    get_fp_index: impl Fn(PositionId) -> usize,
    rng: &mut fastrand::Rng,
) -> PositionId {
    assert!(!empty_positions.is_empty(), "No empty positions on board");

    let mut best_rank = usize::MAX;
    let mut best_positions = Vec::new();

    for &pos in empty_positions {
        let fp_index = get_fp_index(pos);
        let rank = fingerprint_positions[fp_index];

        match rank.cmp(&best_rank) {
            std::cmp::Ordering::Less => {
                best_rank = rank;
                best_positions.clear();
                best_positions.push(pos);
            }
            std::cmp::Ordering::Equal => {
                best_positions.push(pos);
            }
            std::cmp::Ordering::Greater => {}
        }
    }

    best_positions[rng.usize(..best_positions.len())]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;

    fn pos(row: usize, col: usize) -> PositionId {
        PositionId::from_position(Position::new(row, col))
    }

    #[test]
    fn selects_position_with_lowest_rank() {
        // fingerprint_positions[fp_index] = rank
        // Lower rank = higher priority
        let fingerprint_positions = vec![2usize, 0, 1]; // fp 1 has best rank (0)
        let empty_positions = vec![pos(0, 0), pos(0, 1), pos(0, 2)];
        let mut rng = fastrand::Rng::new();

        // Position 0 -> fp index 0 (rank 2)
        // Position 1 -> fp index 1 (rank 0, best)
        // Position 2 -> fp index 2 (rank 1)
        let get_fp_index = |p: PositionId| match (p.row(), p.col()) {
            (0, 0) => 0usize,
            (0, 1) => 1,
            (0, 2) => 2,
            _ => panic!("Unexpected position"),
        };

        let chosen = select_best_position(
            &empty_positions,
            &fingerprint_positions,
            get_fp_index,
            &mut rng,
        );

        assert_eq!(chosen, pos(0, 1));
    }

    #[test]
    fn selects_from_tied_positions() {
        let fingerprint_positions = vec![0usize, 0]; // Both have same rank
        let empty_positions = vec![pos(0, 0), pos(0, 1)];
        let mut rng = fastrand::Rng::new();

        let get_fp_index = |p: PositionId| if p.col() == 0 { 0usize } else { 1 };

        let chosen = select_best_position(
            &empty_positions,
            &fingerprint_positions,
            get_fp_index,
            &mut rng,
        );

        assert!(chosen == pos(0, 0) || chosen == pos(0, 1));
    }
}
