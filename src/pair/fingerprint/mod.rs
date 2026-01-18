pub mod nn1;
pub mod nn2;
pub mod nn3;
pub mod nn4;

pub use nn1::FingerprintNN1;
pub use nn2::FingerprintNN2;
pub use nn3::FingerprintNN3;
pub use nn4::FingerprintNN4;

use crate::pair::NeighborCounts;

/// Generates all possible `NeighborCounts` combinations for a given total neighbor count.
#[must_use]
pub fn combinations_for_total(total: u8) -> Vec<NeighborCounts> {
    let mut result = Vec::new();
    for self_count in 0..=total {
        for opponent_count in 0..=(total - self_count) {
            let empty_count = total - self_count - opponent_count;
            result.push(NeighborCounts::new(self_count, opponent_count, empty_count));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combinations_for_total_zero() {
        let combos = combinations_for_total(0);

        assert_eq!(combos.len(), 1);
        assert_eq!(combos[0], NeighborCounts::new(0, 0, 0));
    }

    #[test]
    fn combinations_for_total_two() {
        let combos = combinations_for_total(2);

        assert_eq!(combos.len(), 6);
        assert_eq!(combos[0], NeighborCounts::new(0, 0, 2));
        assert_eq!(combos[1], NeighborCounts::new(0, 1, 1));
        assert_eq!(combos[2], NeighborCounts::new(0, 2, 0));
        assert_eq!(combos[3], NeighborCounts::new(1, 0, 1));
        assert_eq!(combos[4], NeighborCounts::new(1, 1, 0));
        assert_eq!(combos[5], NeighborCounts::new(2, 0, 0));
    }

    #[test]
    fn combinations_for_total_four() {
        let combos = combinations_for_total(4);

        // For total n, there are (n+1)(n+2)/2 combinations
        // For n=4: 5*6/2 = 15
        assert_eq!(combos.len(), 15);
    }
}
