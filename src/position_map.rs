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

    /// Returns the underlying data as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Returns the underlying data as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Returns a borrowed view of this map.
    #[must_use]
    pub fn as_view(&self) -> PositionMapView<'_, T> {
        PositionMapView {
            data: &self.data,
            stride: self.stride,
        }
    }

    /// Returns a mutable borrowed view of this map.
    pub fn as_view_mut(&mut self) -> PositionMapViewMut<'_, T> {
        PositionMapViewMut {
            data: &mut self.data,
            stride: self.stride,
        }
    }

    /// Consumes the map and returns the underlying data as a `Vec`.
    #[must_use]
    pub fn into_vec(self) -> Vec<T> {
        self.data
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
            .expect("PositionMapView::get: offset overflow");
        &self.data[start..start + self.stride]
    }
}

// ── PositionMapViewMut ────────────────────────────────────────────────

/// A mutable borrowed view of position-indexed data with a runtime stride.
///
/// Each position holds `stride` contiguous elements. Use `get`/`get_mut` to access
/// `&[T]`/`&mut [T]` per position and `stride()` to query the stride.
#[derive(Debug)]
pub struct PositionMapViewMut<'a, T> {
    data: &'a mut [T],
    stride: usize,
}

impl<'a, T> PositionMapViewMut<'a, T> {
    /// Wraps a mutable slice with the given stride.
    ///
    /// Takes the first `PositionId::COUNT * stride` elements from the slice.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() < PositionId::COUNT * stride` or on overflow.
    #[must_use]
    pub fn new(data: &'a mut [T], stride: usize) -> Self {
        let required = PositionId::COUNT
            .checked_mul(stride)
            .expect("PositionMapViewMut::new: stride overflow");
        assert!(
            data.len() >= required,
            "PositionMapViewMut requires at least {required} elements (stride={stride}), got {}",
            data.len(),
        );
        Self {
            data: &mut data[..required],
            stride,
        }
    }

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
            .expect("PositionMapViewMut::get: offset overflow");
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
            .expect("PositionMapViewMut::get_mut: offset overflow");
        &mut self.data[start..start + self.stride]
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
    fn stride_map_as_slice_returns_full_data() {
        let map = PositionMap::new(1.0f32, 2);

        assert_eq!(map.as_slice().len(), PositionId::COUNT * 2);
        assert!(map.as_slice().iter().all(|&val| val == 1.0));
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

        let slice = map.as_view();

        assert_eq!(slice.stride(), 2);
        assert_eq!(slice.get(pos(3, 4)), &[1.0, 2.0]);
    }

    #[test]
    fn stride_map_from_converts_to_stride_slice() {
        let map = PositionMap::new(0.0f32, 2);
        let slice: PositionMapView<'_, f32> = (&map).into();

        assert_eq!(slice.stride(), 2);
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
    fn stride_slice_stride_returns_stride() {
        let data = vec![0i32; PositionId::COUNT];
        let slice = PositionMapView::new(&data, 1);

        assert_eq!(slice.stride(), 1);
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

    // ── PositionMapViewMut stride=1 tests ───────────────────────────────

    #[test]
    fn stride_slice_mut_allows_mutation() {
        let mut data = vec![0i32; PositionId::COUNT];
        {
            let mut slice = PositionMapViewMut::new(&mut data, 1);
            slice.get_mut(pos(5, 5))[0] = 42;
        }
        assert_eq!(data[5 * 15 + 5], 42);
    }

    #[test]
    #[should_panic(expected = "PositionMapViewMut requires at least")]
    fn stride_slice_mut_panics_on_too_small_data() {
        let mut data = vec![0i32; 100];
        let _ = PositionMapViewMut::new(&mut data, 1);
    }

    // ── PositionMapView stride=2 tests ─────────────────────────────────

    #[test]
    fn stride_slice_stride2_wraps_flat_data() {
        let data: Vec<f32> = (0..PositionId::COUNT * 2).map(|idx| idx as f32).collect();
        let slice = PositionMapView::new(&data, 2);

        assert_eq!(slice.stride(), 2);
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

    // ── PositionMapViewMut stride=2 tests ───────────────────────────────

    #[test]
    fn stride_slice_mut_stride2_allows_mutation() {
        let mut data = vec![0.0f32; PositionId::COUNT * 2];
        {
            let mut slice = PositionMapViewMut::new(&mut data, 2);
            let channels = slice.get_mut(pos(5, 5));
            channels[0] = 3.0;
            channels[1] = 4.0;
        }
        let idx = (5 * 15 + 5) * 2;
        assert_eq!(data[idx], 3.0);
        assert_eq!(data[idx + 1], 4.0);
    }

    #[test]
    #[should_panic(expected = "PositionMapViewMut requires at least")]
    fn stride_slice_mut_stride2_panics_on_too_small_data() {
        let mut data = vec![0.0f32; 100];
        let _ = PositionMapViewMut::new(&mut data, 2);
    }
}
