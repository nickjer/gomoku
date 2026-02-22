use crate::position_id::PositionId;

// ── PositionArray ────────────────────────────────────────────────────

/// A stack-allocated, `PositionId`-indexed array.
#[derive(Debug, Clone)]
pub struct PositionArray<T> {
    data: [T; PositionId::COUNT],
}

impl<T: Copy> PositionArray<T> {
    /// Creates a new array with all positions set to `value`.
    #[must_use]
    pub const fn new(value: T) -> Self {
        Self {
            data: [value; PositionId::COUNT],
        }
    }
}

impl<T> PositionArray<T> {
    /// Returns a reference to the value at the given position.
    #[must_use]
    pub const fn get(&self, position: PositionId) -> &T {
        &self.data[position.to_index()]
    }

    /// Returns a mutable reference to the value at the given position.
    pub const fn get_mut(&mut self, position: PositionId) -> &mut T {
        &mut self.data[position.to_index()]
    }
}

// ── PositionMap ───────────────────────────────────────────────────────

/// An owned map from board positions to values with a runtime stride.
///
/// Each position stores `stride` contiguous elements.
/// Use `get`/`get_mut` to access `&[T]`/`&mut [T]` per position.
#[derive(Debug, Clone)]
pub struct PositionMap<T> {
    data: Vec<T>,
    stride: usize,
}

impl<T: Clone> PositionMap<T> {
    /// Creates a new map with `stride` elements per position, all set to `value`.
    ///
    /// # Panics
    ///
    /// Panics if `PositionId::COUNT * stride` overflows.
    pub fn new(value: T, stride: usize) -> Self {
        let len = PositionId::COUNT
            .checked_mul(stride)
            .expect("PositionMap::new: stride overflow");
        Self {
            data: vec![value; len],
            stride,
        }
    }

    /// Creates a map pre-filled with `fill`, then calls `f` for each position
    /// with a mutable slice of `stride` elements for in-place modification.
    ///
    /// # Panics
    ///
    /// Panics if `PositionId::COUNT * stride` overflows.
    #[must_use]
    pub fn from_fn(fill: T, stride: usize, mut f: impl FnMut(PositionId, &mut [T])) -> Self {
        let mut map = Self::new(fill, stride);
        for pos in PositionId::iter() {
            f(pos, map.get_mut(pos));
        }
        map
    }
}

impl<T> PositionMap<T> {
    /// Returns the stride (number of elements per position).
    #[must_use]
    pub const fn stride(&self) -> usize {
        self.stride
    }

    /// Returns a reference to the elements at the given position.
    ///
    /// # Panics
    ///
    /// Panics if `position_id` is out of bounds or on offset overflow.
    #[must_use]
    pub fn get(&self, position_id: PositionId) -> &[T] {
        let start = usize::from(position_id)
            .checked_mul(self.stride)
            .expect("PositionMap::get: offset overflow");
        &self.data[start..start + self.stride]
    }

    /// Returns a mutable reference to the elements at the given position.
    ///
    /// # Panics
    ///
    /// Panics if `position_id` is out of bounds or on offset overflow.
    pub fn get_mut(&mut self, position_id: PositionId) -> &mut [T] {
        let start = usize::from(position_id)
            .checked_mul(self.stride)
            .expect("PositionMap::get_mut: offset overflow");
        &mut self.data[start..start + self.stride]
    }

    /// Returns the underlying data as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;

    fn pos(row: usize, col: usize) -> PositionId {
        PositionId::from_position(Position::new(row, col))
    }

    // ── PositionArray tests ────────────────────────────────────────

    #[test]
    fn position_array_initializes_to_value() {
        let array = PositionArray::new(false);

        assert!(!array.get(pos(0, 0)));
        assert!(!array.get(pos(7, 7)));
        assert!(!array.get(pos(14, 14)));
    }

    #[test]
    fn position_array_get_mut_modifies_value() {
        let mut array = PositionArray::new(0i32);
        *array.get_mut(pos(3, 4)) = 42;

        assert_eq!(*array.get(pos(3, 4)), 42);
        assert_eq!(*array.get(pos(0, 0)), 0);
    }

    // ── PositionMap tests ──────────────────────────────────────────

    #[test]
    fn stride_map_get_returns_correct_pair() {
        let mut map = PositionMap::new(0.0f32, 2);
        map.get_mut(pos(3, 4)).copy_from_slice(&[1.0, 2.0]);

        assert_eq!(map.get(pos(3, 4)), &[1.0, 2.0]);
        assert_eq!(map.get(pos(0, 0)), &[0.0, 0.0]);
    }

    #[test]
    fn stride_map_get_mut_modifies_correctly() {
        let mut map = PositionMap::new(0i32, 2);
        map.get_mut(pos(7, 7)).copy_from_slice(&[10, 20]);

        assert_eq!(map.get(pos(7, 7)), &[10, 20]);
        assert_eq!(map.get(pos(7, 8)), &[0, 0]);
    }

    #[test]
    fn stride_map_stride_returns_stride() {
        let map = PositionMap::new(0i32, 3);

        assert_eq!(map.stride(), 3);
    }

    // ── PositionMap::from_fn tests ─────────────────────────────────────

    #[test]
    fn from_fn_produces_correct_stride() {
        let map = PositionMap::from_fn(0i32, 3, |_, _| {});

        assert_eq!(map.stride(), 3);
    }

    #[test]
    fn from_fn_unmodified_positions_retain_fill_value() {
        let map = PositionMap::from_fn(42i32, 2, |_, _| {});

        assert_eq!(map.get(pos(0, 0)), &[42, 42]);
        assert_eq!(map.get(pos(14, 14)), &[42, 42]);
    }

    #[test]
    fn from_fn_closure_can_overwrite_values() {
        let map = PositionMap::from_fn(0.0f32, 2, |position, slice| {
            let idx = usize::from(position);
            slice[0] = idx as f32;
            slice[1] = idx as f32 * 10.0;
        });

        assert_eq!(map.get(pos(0, 0)), &[0.0, 0.0]);
        assert_eq!(map.get(pos(0, 1)), &[1.0, 10.0]);
        assert_eq!(map.get(pos(1, 0)), &[15.0, 150.0]);
    }

    #[test]
    fn from_fn_closure_receives_correct_position() {
        let map = PositionMap::from_fn(0usize, 1, |position, slice| {
            slice[0] = usize::from(position);
        });

        for position in PositionId::iter() {
            assert_eq!(map.get(position), &[usize::from(position)]);
        }
    }

    #[test]
    fn from_fn_matches_manual_construction() {
        let mut expected = PositionMap::new(0.0f32, 2);
        expected.get_mut(pos(3, 4)).copy_from_slice(&[1.0, 2.0]);
        expected.get_mut(pos(7, 7)).copy_from_slice(&[3.0, 4.0]);

        let actual = PositionMap::from_fn(0.0f32, 2, |position, slice| {
            if position == pos(3, 4) {
                slice.copy_from_slice(&[1.0, 2.0]);
            } else if position == pos(7, 7) {
                slice.copy_from_slice(&[3.0, 4.0]);
            }
        });

        for position in PositionId::iter() {
            assert_eq!(actual.get(position), expected.get(position));
        }
    }
}
