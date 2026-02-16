use crate::position_id::PositionId;
use crate::position_map::PositionMap;

/// Selects the position with the highest policy value.
///
/// When multiple positions are tied for the highest value, one is chosen
/// uniformly at random using reservoir sampling.
///
/// # Panics
///
/// Panics if `empty_positions` is empty.
pub fn select_best_position(
    empty_positions: &[PositionId],
    policy: &PositionMap<f32>,
    rng: &mut fastrand::Rng,
) -> PositionId {
    let (&first, rest) = empty_positions
        .split_first()
        .expect("no empty positions to select from");

    let mut best_pos = first;
    let mut best_value = policy.get(first)[0];
    let mut tie_count = 1;

    for &pos in rest {
        let value = policy.get(pos)[0];

        if value > best_value {
            best_value = value;
            best_pos = pos;
            tie_count = 1;
        } else if (value - best_value).abs() < f32::EPSILON {
            tie_count += 1;
            // Reservoir sampling: replace with probability 1/tie_count
            if rng.usize(..tie_count) == 0 {
                best_pos = pos;
            }
        }
    }

    best_pos
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;
    use crate::position_map::PositionMap;

    fn pos(row: usize, col: usize) -> PositionId {
        PositionId::from_position(Position::new(row, col))
    }

    fn policy_with_values(values: &[(PositionId, f32)]) -> PositionMap<f32> {
        let mut map = PositionMap::new(f32::NEG_INFINITY, 1);
        for &(pos, value) in values {
            map.get_mut(pos)[0] = value;
        }
        map
    }

    #[test]
    fn selects_highest_value() {
        let positions = vec![pos(0, 0), pos(0, 1), pos(0, 2)];
        let policy = policy_with_values(&[(pos(0, 0), 1.0), (pos(0, 1), 5.0), (pos(0, 2), 3.0)]);
        let mut rng = fastrand::Rng::with_seed(42);

        let selected = select_best_position(&positions, &policy, &mut rng);

        assert_eq!(selected, pos(0, 1));
    }

    #[test]
    fn handles_negative_values() {
        let positions = vec![pos(0, 0), pos(0, 1), pos(0, 2)];
        let policy = policy_with_values(&[(pos(0, 0), -5.0), (pos(0, 1), -1.0), (pos(0, 2), -3.0)]);
        let mut rng = fastrand::Rng::with_seed(42);

        let selected = select_best_position(&positions, &policy, &mut rng);

        assert_eq!(selected, pos(0, 1));
    }

    #[test]
    fn tiebreaking_selects_from_tied_positions() {
        let positions = vec![pos(0, 0), pos(0, 1), pos(0, 2)];
        let policy = policy_with_values(&[(pos(0, 0), 5.0), (pos(0, 1), 5.0), (pos(0, 2), 1.0)]);
        let mut rng = fastrand::Rng::with_seed(42);

        let selected = select_best_position(&positions, &policy, &mut rng);

        assert!(selected == pos(0, 0) || selected == pos(0, 1));
    }

    #[test]
    fn tiebreaking_is_uniform() {
        let positions = vec![pos(0, 0), pos(0, 1)];
        let policy = policy_with_values(&[(pos(0, 0), 5.0), (pos(0, 1), 5.0)]);

        let mut counts = [0, 0];
        for seed in 0..1000 {
            let mut rng = fastrand::Rng::with_seed(seed);
            let selected = select_best_position(&positions, &policy, &mut rng);

            if selected == pos(0, 0) {
                counts[0] += 1;
            } else {
                counts[1] += 1;
            }
        }

        // With 1000 trials, each should be ~500. Allow 40% to 60% range.
        assert!(
            counts[0] > 400 && counts[0] < 600,
            "distribution not uniform: {counts:?}"
        );
    }

    #[test]
    fn deterministic_with_same_seed() {
        let positions = vec![pos(0, 0), pos(0, 1), pos(0, 2)];
        let policy = policy_with_values(&[(pos(0, 0), 5.0), (pos(0, 1), 5.0), (pos(0, 2), 5.0)]);

        let results: Vec<_> = (0..5)
            .map(|_| {
                let mut rng = fastrand::Rng::with_seed(12345);
                select_best_position(&positions, &policy, &mut rng)
            })
            .collect();

        assert!(results.windows(2).all(|w| w[0] == w[1]));
    }

    #[test]
    fn only_considers_provided_positions() {
        // pos(0,0) has highest value but isn't in the list
        let positions = vec![pos(0, 1), pos(0, 2)];
        let policy = policy_with_values(&[(pos(0, 0), 100.0), (pos(0, 1), 5.0), (pos(0, 2), 3.0)]);
        let mut rng = fastrand::Rng::with_seed(42);

        let selected = select_best_position(&positions, &policy, &mut rng);

        assert_eq!(selected, pos(0, 1));
    }

    #[test]
    fn single_position_returns_that_position() {
        let positions = vec![pos(7, 7)];
        let policy = policy_with_values(&[(pos(7, 7), 0.0)]);
        let mut rng = fastrand::Rng::with_seed(42);

        let selected = select_best_position(&positions, &policy, &mut rng);

        assert_eq!(selected, pos(7, 7));
    }

    #[test]
    #[should_panic(expected = "no empty positions")]
    fn panics_on_empty_positions() {
        let positions: Vec<PositionId> = vec![];
        let policy = PositionMap::new(0.0f32, 1);
        let mut rng = fastrand::Rng::with_seed(42);

        select_best_position(&positions, &policy, &mut rng);
    }
}
