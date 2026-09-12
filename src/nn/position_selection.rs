use crate::position_id::PositionId;
use crate::position_map::PositionMap;

/// Selects the position with the highest scores value.
///
/// When multiple positions are tied for the highest value, one is chosen
/// uniformly at random using reservoir sampling.
///
/// # Panics
///
/// Panics if `empty_positions` is empty.
pub fn highest_scored_empty_position(
    empty_positions: &[PositionId],
    scores: &PositionMap<f32, 1>,
    rng: &mut fastrand::Rng,
) -> PositionId {
    let (&first, rest) = empty_positions
        .split_first()
        .expect("no empty positions to select from");

    let mut best_pos = first;
    let [mut best_value] = *scores.get(first);
    let mut tie_count = 1;

    for &pos in rest {
        let [value] = *scores.get(pos);

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

    fn scores_with_values(values: &[(PositionId, f32)]) -> PositionMap<f32, 1> {
        let mut map = PositionMap::new(f32::NEG_INFINITY);
        for &(pos, value) in values {
            *map.get_mut(pos) = [value];
        }
        map
    }

    #[test]
    fn selects_highest_value() {
        let positions = vec![pos(0, 0), pos(0, 1), pos(0, 2)];
        let scores = scores_with_values(&[(pos(0, 0), 1.0), (pos(0, 1), 5.0), (pos(0, 2), 3.0)]);
        let mut rng = fastrand::Rng::with_seed(42);

        let selected = highest_scored_empty_position(&positions, &scores, &mut rng);

        assert_eq!(selected, pos(0, 1));
    }

    #[test]
    fn handles_negative_values() {
        let positions = vec![pos(0, 0), pos(0, 1), pos(0, 2)];
        let scores = scores_with_values(&[(pos(0, 0), -5.0), (pos(0, 1), -1.0), (pos(0, 2), -3.0)]);
        let mut rng = fastrand::Rng::with_seed(42);

        let selected = highest_scored_empty_position(&positions, &scores, &mut rng);

        assert_eq!(selected, pos(0, 1));
    }

    #[test]
    fn tiebreaking_selects_from_tied_positions() {
        let positions = vec![pos(0, 0), pos(0, 1), pos(0, 2)];
        let scores = scores_with_values(&[(pos(0, 0), 5.0), (pos(0, 1), 5.0), (pos(0, 2), 1.0)]);
        let mut rng = fastrand::Rng::with_seed(42);

        let selected = highest_scored_empty_position(&positions, &scores, &mut rng);

        assert!([pos(0, 0), pos(0, 1)].contains(&selected), "{selected:?}");
    }

    #[test]
    fn tiebreaking_is_uniform() {
        let positions = vec![pos(0, 0), pos(0, 1)];
        let scores = scores_with_values(&[(pos(0, 0), 5.0), (pos(0, 1), 5.0)]);

        let mut counts = [0, 0];
        for seed in 0..1000 {
            let mut rng = fastrand::Rng::with_seed(seed);
            let selected = highest_scored_empty_position(&positions, &scores, &mut rng);

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
        let scores = scores_with_values(&[(pos(0, 0), 5.0), (pos(0, 1), 5.0), (pos(0, 2), 5.0)]);

        let results: Vec<_> = (0..5)
            .map(|_| {
                let mut rng = fastrand::Rng::with_seed(12345);
                highest_scored_empty_position(&positions, &scores, &mut rng)
            })
            .collect();

        assert!(results.windows(2).all(|w| w[0] == w[1]));
    }

    #[test]
    fn only_considers_provided_positions() {
        // pos(0,0) has highest value but isn't in the list
        let positions = vec![pos(0, 1), pos(0, 2)];
        let scores = scores_with_values(&[(pos(0, 0), 100.0), (pos(0, 1), 5.0), (pos(0, 2), 3.0)]);
        let mut rng = fastrand::Rng::with_seed(42);

        let selected = highest_scored_empty_position(&positions, &scores, &mut rng);

        assert_eq!(selected, pos(0, 1));
    }

    #[test]
    fn single_position_returns_that_position() {
        let positions = vec![pos(7, 7)];
        let scores = scores_with_values(&[(pos(7, 7), 0.0)]);
        let mut rng = fastrand::Rng::with_seed(42);

        let selected = highest_scored_empty_position(&positions, &scores, &mut rng);

        assert_eq!(selected, pos(7, 7));
    }

    #[test]
    #[should_panic(expected = "no empty positions")]
    fn panics_on_empty_positions() {
        let positions: Vec<PositionId> = vec![];
        let scores = PositionMap::<f32, 1>::new(0.0);
        let mut rng = fastrand::Rng::with_seed(42);

        highest_scored_empty_position(&positions, &scores, &mut rng);
    }
}
