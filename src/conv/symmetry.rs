use crate::position_id::PositionId;

/// D8 symmetry group transformations for the board.
///
/// The D8 group has 8 elements: 4 rotations and 4 reflections.
/// These are used to average CNN outputs for equivariant move selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum D8Transform {
    Identity,
    Rotate90,
    Rotate180,
    Rotate270,
    FlipH,
    FlipV,
    FlipDiagonal,
    FlipAntiDiagonal,
}

impl D8Transform {
    /// All 8 D8 transformations.
    pub const ALL: [Self; 8] = [
        Self::Identity,
        Self::Rotate90,
        Self::Rotate180,
        Self::Rotate270,
        Self::FlipH,
        Self::FlipV,
        Self::FlipDiagonal,
        Self::FlipAntiDiagonal,
    ];

    /// Applies this transformation to a position.
    #[must_use]
    pub fn apply(self, pos: PositionId) -> PositionId {
        match self {
            Self::Identity => pos,
            Self::Rotate90 => pos.transpose().flip_horizontal(),
            Self::Rotate180 => pos.invert(),
            Self::Rotate270 => pos.transpose().flip_vertical(),
            Self::FlipH => pos.flip_horizontal(),
            Self::FlipV => pos.flip_vertical(),
            Self::FlipDiagonal => pos.transpose(),
            Self::FlipAntiDiagonal => pos.transpose().invert(),
        }
    }

    /// Returns the inverse transformation.
    #[must_use]
    pub const fn inverse(self) -> Self {
        match self {
            Self::Identity => Self::Identity,
            Self::Rotate90 => Self::Rotate270,
            Self::Rotate180 => Self::Rotate180,
            Self::Rotate270 => Self::Rotate90,
            Self::FlipH => Self::FlipH,
            Self::FlipV => Self::FlipV,
            Self::FlipDiagonal => Self::FlipDiagonal,
            Self::FlipAntiDiagonal => Self::FlipAntiDiagonal,
        }
    }

    /// Applies the inverse transformation to a position.
    #[must_use]
    pub fn apply_inverse(self, pos: PositionId) -> PositionId {
        self.inverse().apply(pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;

    fn pos(row: usize, col: usize) -> PositionId {
        PositionId::from_position(Position::new(row, col))
    }

    #[test]
    fn identity_preserves_position() {
        for p in PositionId::iter() {
            assert_eq!(D8Transform::Identity.apply(p), p);
        }
    }

    #[test]
    fn rotate90_transforms_corners() {
        // (0,0) -> transpose -> (0,0) -> flip_h -> (0,14)
        assert_eq!(D8Transform::Rotate90.apply(pos(0, 0)), pos(0, 14));
        // (0,14) -> transpose -> (14,0) -> flip_h -> (14,14)
        assert_eq!(D8Transform::Rotate90.apply(pos(0, 14)), pos(14, 14));
        // (14,14) -> transpose -> (14,14) -> flip_h -> (14,0)
        assert_eq!(D8Transform::Rotate90.apply(pos(14, 14)), pos(14, 0));
        // (14,0) -> transpose -> (0,14) -> flip_h -> (0,0)
        assert_eq!(D8Transform::Rotate90.apply(pos(14, 0)), pos(0, 0));
    }

    #[test]
    fn rotate90_four_times_is_identity() {
        for p in PositionId::iter() {
            let r1 = D8Transform::Rotate90.apply(p);
            let r2 = D8Transform::Rotate90.apply(r1);
            let r3 = D8Transform::Rotate90.apply(r2);
            let r4 = D8Transform::Rotate90.apply(r3);
            assert_eq!(r4, p);
        }
    }

    #[test]
    fn rotate180_is_invert() {
        for p in PositionId::iter() {
            assert_eq!(D8Transform::Rotate180.apply(p), p.invert());
        }
    }

    #[test]
    fn rotate270_is_rotate90_inverse() {
        for p in PositionId::iter() {
            let r90 = D8Transform::Rotate90.apply(p);
            let back = D8Transform::Rotate270.apply(r90);
            assert_eq!(back, p);
        }
    }

    #[test]
    fn all_transforms_are_invertible() {
        for t in D8Transform::ALL {
            for p in PositionId::iter() {
                let transformed = t.apply(p);
                let back = t.apply_inverse(transformed);
                assert_eq!(back, p, "{:?} is not properly invertible", t);
            }
        }
    }

    #[test]
    fn flips_are_self_inverse() {
        let self_inverse = [
            D8Transform::Identity,
            D8Transform::Rotate180,
            D8Transform::FlipH,
            D8Transform::FlipV,
            D8Transform::FlipDiagonal,
            D8Transform::FlipAntiDiagonal,
        ];
        for t in self_inverse {
            assert_eq!(t.inverse(), t, "{:?} should be self-inverse", t);
        }
    }

    #[test]
    fn center_is_fixed_by_all_transforms() {
        let center = PositionId::center();
        for t in D8Transform::ALL {
            assert_eq!(t.apply(center), center, "{:?} should fix center", t);
        }
    }

    #[test]
    fn flip_h_mirrors_horizontally() {
        assert_eq!(D8Transform::FlipH.apply(pos(0, 0)), pos(0, 14));
        assert_eq!(D8Transform::FlipH.apply(pos(5, 3)), pos(5, 11));
    }

    #[test]
    fn flip_v_mirrors_vertically() {
        assert_eq!(D8Transform::FlipV.apply(pos(0, 0)), pos(14, 0));
        assert_eq!(D8Transform::FlipV.apply(pos(3, 5)), pos(11, 5));
    }

    #[test]
    fn flip_diagonal_transposes() {
        assert_eq!(D8Transform::FlipDiagonal.apply(pos(0, 5)), pos(5, 0));
        assert_eq!(D8Transform::FlipDiagonal.apply(pos(2, 3)), pos(3, 2));
    }

    #[test]
    fn flip_anti_diagonal_combines_transpose_and_invert() {
        for p in PositionId::iter() {
            assert_eq!(
                D8Transform::FlipAntiDiagonal.apply(p),
                p.transpose().invert()
            );
        }
    }
}
