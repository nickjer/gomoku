use crate::position_id::PositionId;

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
}

impl<T: Clone> PositionMap<T> {
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

    /// Returns a borrowed view of this map.
    #[must_use]
    pub fn as_view(&self) -> PositionMapView<'_, T> {
        PositionMapView::new(&self.data, self.stride)
    }
}

impl<'a, T> From<&'a PositionMap<T>> for PositionMapView<'a, T> {
    fn from(map: &'a PositionMap<T>) -> Self {
        map.as_view()
    }
}

// ── PositionMapView ───────────────────────────────────────────────────

/// A borrowed view of position-indexed data with a runtime stride.
///
/// Each position holds `stride` contiguous elements. Use `get` to access
/// `&[T]` per position and `stride()` to query the stride.
#[derive(Debug, Clone, Copy)]
pub struct PositionMapView<'a, T> {
    data: &'a [T],
    stride: usize,
}

impl<'a, T> PositionMapView<'a, T> {
    /// Wraps a slice with the given stride.
    ///
    /// Takes the first `PositionId::COUNT * stride` elements from the slice.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() < PositionId::COUNT * stride` or on overflow.
    #[must_use]
    pub fn new(data: &'a [T], stride: usize) -> Self {
        let required = PositionId::COUNT
            .checked_mul(stride)
            .expect("PositionMapView::new: stride overflow");
        assert!(
            data.len() >= required,
            "PositionMapView requires at least {required} elements (stride={stride}), got {}",
            data.len(),
        );
        Self {
            data: &data[..required],
            stride,
        }
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
            .expect("PositionMapView::get: offset overflow");
        &self.data[start..start + self.stride]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;

    fn pos(row: usize, col: usize) -> PositionId {
        PositionId::from_position(Position::new(row, col))
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

    #[test]
    fn stride_map_as_view_returns_view() {
        let mut map = PositionMap::new(0.0f32, 2);
        map.get_mut(pos(3, 4)).copy_from_slice(&[1.0, 2.0]);

        let view = map.as_view();

        assert_eq!(view.get(pos(3, 4)), &[1.0, 2.0]);
        assert_eq!(view.get(pos(0, 0)), &[0.0, 0.0]);
    }

    #[test]
    fn stride_map_from_converts_to_stride_slice() {
        let map = PositionMap::new(0.0f32, 2);
        let view: PositionMapView<'_, f32> = (&map).into();

        assert_eq!(view.get(pos(0, 0)), &[0.0, 0.0]);
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

    // ── PositionMapView stride=1 tests ─────────────────────────────────

    #[test]
    fn stride_slice_get_returns_correct_value() {
        let data: Vec<i32> = (0..PositionId::COUNT as i32).collect();
        let slice = PositionMapView::new(&data, 1);

        assert_eq!(slice.get(pos(0, 0)), &[0]);
        assert_eq!(slice.get(pos(0, 1)), &[1]);
        assert_eq!(slice.get(pos(1, 0)), &[15]);
    }

    #[test]
    fn stride_slice_get_returns_single_element() {
        let data = vec![0i32; PositionId::COUNT];
        let slice = PositionMapView::new(&data, 1);

        assert_eq!(slice.get(pos(0, 0)).len(), 1);
    }

    #[test]
    #[should_panic(expected = "PositionMapView requires at least")]
    fn stride_slice_panics_on_too_small_data() {
        let data = vec![0i32; 100];
        let _ = PositionMapView::new(&data, 1);
    }

    #[test]
    fn stride_slice_accepts_oversized_data() {
        let data = vec![0i32; PositionId::COUNT + 100];
        let slice = PositionMapView::new(&data, 1);

        assert_eq!(slice.get(pos(0, 0)), &[0]);
    }

    // ── PositionMapView stride=2 tests ─────────────────────────────────

    #[test]
    fn stride_slice_stride2_wraps_flat_data() {
        let data: Vec<f32> = (0..PositionId::COUNT * 2).map(|idx| idx as f32).collect();
        let slice = PositionMapView::new(&data, 2);

        assert_eq!(slice.get(pos(0, 0)), &[0.0, 1.0]);
        assert_eq!(slice.get(pos(0, 1)), &[2.0, 3.0]);
        assert_eq!(slice.get(pos(1, 0)), &[30.0, 31.0]);
    }

    #[test]
    #[should_panic(expected = "PositionMapView requires at least")]
    fn stride_slice_stride2_panics_on_too_small_data() {
        let data = vec![0.0f32; 100];
        let _ = PositionMapView::new(&data, 2);
    }
}
