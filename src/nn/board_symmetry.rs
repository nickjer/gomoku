use crate::board::Board;
use crate::position_id::PositionId;

/// One of the eight ways to turn a square board over onto itself: four
/// rotations and four reflections.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoardSymmetry {
    Identity,
    Rotate90,
    Rotate180,
    Rotate270,
    FlipHorizontal,
    FlipVertical,
    FlipDiagonal,
    FlipAntiDiagonal,
}

impl BoardSymmetry {
    /// Every symmetry, identity first.
    pub const ALL: [Self; 8] = [
        Self::Identity,
        Self::Rotate90,
        Self::Rotate180,
        Self::Rotate270,
        Self::FlipHorizontal,
        Self::FlipVertical,
        Self::FlipDiagonal,
        Self::FlipAntiDiagonal,
    ];

    /// A random symmetry.
    #[must_use]
    pub fn random(rng: &mut fastrand::Rng) -> Self {
        Self::ALL[rng.usize(0..8)]
    }

    /// Where this symmetry sends a position.
    #[must_use]
    pub fn apply(self, pos: PositionId) -> PositionId {
        match self {
            Self::Identity => pos,
            Self::Rotate90 => pos.transpose().flip_horizontal(),
            Self::Rotate180 => pos.invert(),
            Self::Rotate270 => pos.transpose().flip_vertical(),
            Self::FlipHorizontal => pos.flip_horizontal(),
            Self::FlipVertical => pos.flip_vertical(),
            Self::FlipDiagonal => pos.transpose(),
            Self::FlipAntiDiagonal => pos.transpose().invert(),
        }
    }

    /// Where a position came from: undoes [`Self::apply`].
    #[must_use]
    pub fn apply_inverse(self, pos: PositionId) -> PositionId {
        match self {
            Self::Identity => pos,
            Self::Rotate90 => pos.transpose().flip_vertical(),
            Self::Rotate180 => pos.invert(),
            Self::Rotate270 => pos.transpose().flip_horizontal(),
            Self::FlipHorizontal => pos.flip_horizontal(),
            Self::FlipVertical => pos.flip_vertical(),
            Self::FlipDiagonal => pos.transpose(),
            Self::FlipAntiDiagonal => pos.transpose().invert(),
        }
    }

    /// Turns a whole board: every stone moves to where this symmetry sends
    /// its position.
    #[must_use]
    pub fn apply_to_board(self, board: &Board) -> Board {
        let mut turned_board = Board::new();
        for position in PositionId::iter() {
            if let Some(stone) = board.stone(position) {
                turned_board.place_unchecked(self.apply(position), stone);
            }
        }
        turned_board
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;
    use crate::stone::Stone;

    #[test]
    fn rotate180_moves_a_stone_to_the_opposite_corner() {
        let mut board = Board::new();
        board.place(pos(0, 0), Stone::Black).unwrap();
        board.place(pos(0, 1), Stone::White).unwrap();

        let turned_board = BoardSymmetry::Rotate180.apply_to_board(&board);

        assert_eq!(turned_board.stone(pos(14, 14)), Some(Stone::Black));
        assert_eq!(turned_board.stone(pos(14, 13)), Some(Stone::White));
        assert_eq!(turned_board.stone(pos(0, 0)), None);
        assert_eq!(turned_board.move_count(), 2);
    }

    #[test]
    fn apply_to_board_sends_every_stone_where_apply_says() {
        let mut board = Board::new();
        for black_position in [pos(0, 0), pos(3, 7), pos(14, 2)] {
            board.place(black_position, Stone::Black).unwrap();
        }
        board.place(pos(9, 9), Stone::White).unwrap();

        for symmetry in BoardSymmetry::ALL {
            let turned_board = symmetry.apply_to_board(&board);
            for position in PositionId::iter() {
                assert_eq!(
                    turned_board.stone(symmetry.apply(position)),
                    board.stone(position),
                    "{symmetry:?} at {position:?}"
                );
            }
        }
    }

    fn pos(row: usize, col: usize) -> PositionId {
        PositionId::from_position(Position::new(row, col))
    }

    #[test]
    fn identity_preserves_position() {
        for p in PositionId::iter() {
            assert_eq!(BoardSymmetry::Identity.apply(p), p);
        }
    }

    #[test]
    fn rotate90_transforms_corners() {
        // (0,0) -> transpose -> (0,0) -> flip_h -> (0,14)
        assert_eq!(BoardSymmetry::Rotate90.apply(pos(0, 0)), pos(0, 14));
        // (0,14) -> transpose -> (14,0) -> flip_h -> (14,14)
        assert_eq!(BoardSymmetry::Rotate90.apply(pos(0, 14)), pos(14, 14));
        // (14,14) -> transpose -> (14,14) -> flip_h -> (14,0)
        assert_eq!(BoardSymmetry::Rotate90.apply(pos(14, 14)), pos(14, 0));
        // (14,0) -> transpose -> (0,14) -> flip_h -> (0,0)
        assert_eq!(BoardSymmetry::Rotate90.apply(pos(14, 0)), pos(0, 0));
    }

    #[test]
    fn rotate90_four_times_is_identity() {
        for p in PositionId::iter() {
            let r1 = BoardSymmetry::Rotate90.apply(p);
            let r2 = BoardSymmetry::Rotate90.apply(r1);
            let r3 = BoardSymmetry::Rotate90.apply(r2);
            let r4 = BoardSymmetry::Rotate90.apply(r3);
            assert_eq!(r4, p);
        }
    }

    #[test]
    fn rotate180_is_invert() {
        for p in PositionId::iter() {
            assert_eq!(BoardSymmetry::Rotate180.apply(p), p.invert());
        }
    }

    #[test]
    fn rotate270_is_rotate90_inverse() {
        for p in PositionId::iter() {
            let r90 = BoardSymmetry::Rotate90.apply(p);
            let back = BoardSymmetry::Rotate270.apply(r90);
            assert_eq!(back, p);
        }
    }

    #[test]
    fn all_transforms_are_invertible() {
        for t in BoardSymmetry::ALL {
            for p in PositionId::iter() {
                let transformed = t.apply(p);
                let back = t.apply_inverse(transformed);
                assert_eq!(back, p, "{:?} is not properly invertible", t);
            }
        }
    }

    #[test]
    fn center_is_fixed_by_all_transforms() {
        let center = PositionId::center();
        for t in BoardSymmetry::ALL {
            assert_eq!(t.apply(center), center, "{:?} should fix center", t);
        }
    }

    #[test]
    fn flip_h_mirrors_horizontally() {
        assert_eq!(BoardSymmetry::FlipHorizontal.apply(pos(0, 0)), pos(0, 14));
        assert_eq!(BoardSymmetry::FlipHorizontal.apply(pos(5, 3)), pos(5, 11));
    }

    #[test]
    fn flip_v_mirrors_vertically() {
        assert_eq!(BoardSymmetry::FlipVertical.apply(pos(0, 0)), pos(14, 0));
        assert_eq!(BoardSymmetry::FlipVertical.apply(pos(3, 5)), pos(11, 5));
    }

    #[test]
    fn flip_diagonal_transposes() {
        assert_eq!(BoardSymmetry::FlipDiagonal.apply(pos(0, 5)), pos(5, 0));
        assert_eq!(BoardSymmetry::FlipDiagonal.apply(pos(2, 3)), pos(3, 2));
    }

    #[test]
    fn flip_anti_diagonal_combines_transpose_and_invert() {
        for p in PositionId::iter() {
            assert_eq!(
                BoardSymmetry::FlipAntiDiagonal.apply(p),
                p.transpose().invert()
            );
        }
    }

    #[test]
    fn random_returns_valid_transform() {
        let mut rng = fastrand::Rng::with_seed(42);
        for _ in 0..100 {
            let t = BoardSymmetry::random(&mut rng);
            assert!(BoardSymmetry::ALL.contains(&t));
        }
    }

    #[test]
    fn random_produces_variety() {
        let mut rng = fastrand::Rng::with_seed(42);
        let mut seen = [false; 8];
        for _ in 0..1000 {
            let t = BoardSymmetry::random(&mut rng);
            let idx = BoardSymmetry::ALL.iter().position(|&x| x == t).unwrap();
            seen[idx] = true;
        }
        assert!(seen.iter().all(|&x| x), "Should see all transforms");
    }
}
