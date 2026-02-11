use crate::position_id::PositionId;

/// A map from board positions to values, indexed by `PositionId`.
///
/// Each position stores `N` contiguous elements.
/// For `N = 1` (default), this behaves as a simple position-to-value map with `Index` support.
/// For `N > 1`, use `get`/`get_mut` to access `&[T; N]` per position.
#[derive(Debug, Clone)]
pub struct PositionMap<T, const N: usize = 1> {
    data: Vec<T>,
}

impl<T: Clone, const N: usize> PositionMap<T, N> {
    pub fn new(value: T) -> Self {
        Self {
            data: vec![value; PositionId::COUNT * N],
        }
    }
}

impl<T, const N: usize> PositionMap<T, N> {
    /// Returns a reference to the `N` elements at the given position.
    ///
    /// # Panics
    ///
    /// Panics if `position_id` is out of bounds.
    #[must_use]
    pub fn get(&self, position_id: PositionId) -> &[T; N] {
        let start = usize::from(position_id) * N;
        self.data[start..start + N].try_into().unwrap()
    }

    /// Returns a mutable reference to the `N` elements at the given position.
    ///
    /// # Panics
    ///
    /// Panics if `position_id` is out of bounds.
    pub fn get_mut(&mut self, position_id: PositionId) -> &mut [T; N] {
        let start = usize::from(position_id) * N;
        (&mut self.data[start..start + N]).try_into().unwrap()
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

    /// Consumes the map and returns the underlying data as a `Vec`.
    #[must_use]
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
}

impl<T> std::ops::Index<PositionId> for PositionMap<T, 1> {
    type Output = T;

    fn index(&self, position_id: PositionId) -> &Self::Output {
        &self.data[usize::from(position_id)]
    }
}

impl<T> std::ops::IndexMut<PositionId> for PositionMap<T, 1> {
    fn index_mut(&mut self, position_id: PositionId) -> &mut Self::Output {
        &mut self.data[usize::from(position_id)]
    }
}

/// A borrowed view of position-indexed data with a runtime stride.
///
/// Each position holds `stride` contiguous elements. Use `get` to access
/// `&[T]` per position and `stride()` to query the stride.
#[derive(Debug, Clone, Copy)]
pub struct PositionSlice<'a, T> {
    data: &'a [T],
    stride: usize,
}

impl<'a, T> PositionSlice<'a, T> {
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
            .expect("PositionSlice: stride overflow");
        assert!(
            data.len() >= required,
            "PositionSlice requires at least {required} elements (stride={stride}), got {}",
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
    /// Panics if `position_id` is out of bounds.
    #[must_use]
    pub fn get(&self, position_id: PositionId) -> &[T] {
        let start = usize::from(position_id)
            .checked_mul(self.stride)
            .expect("PositionSlice::get: offset overflow");
        &self.data[start..start + self.stride]
    }
}

/// A mutable borrowed view of position-indexed data with a runtime stride.
///
/// Each position holds `stride` contiguous elements. Use `get`/`get_mut` to access
/// `&[T]`/`&mut [T]` per position and `stride()` to query the stride.
#[derive(Debug)]
pub struct PositionSliceMut<'a, T> {
    data: &'a mut [T],
    stride: usize,
}

impl<'a, T> PositionSliceMut<'a, T> {
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
            .expect("PositionSliceMut: stride overflow");
        assert!(
            data.len() >= required,
            "PositionSliceMut requires at least {required} elements (stride={stride}), got {}",
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
    /// Panics if `position_id` is out of bounds.
    #[must_use]
    pub fn get(&self, position_id: PositionId) -> &[T] {
        let start = usize::from(position_id)
            .checked_mul(self.stride)
            .expect("PositionSliceMut::get: offset overflow");
        &self.data[start..start + self.stride]
    }

    /// Returns a mutable reference to the elements at the given position.
    ///
    /// # Panics
    ///
    /// Panics if `position_id` is out of bounds.
    pub fn get_mut(&mut self, position_id: PositionId) -> &mut [T] {
        let start = usize::from(position_id)
            .checked_mul(self.stride)
            .expect("PositionSliceMut::get_mut: offset overflow");
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

    // ── PositionMap N=1 tests ──────────────────────────────────────────

    #[test]
    fn position_map_indexes_correctly() {
        let mut map = PositionMap::new(0i32);
        map[pos(0, 0)] = 1;
        map[pos(0, 1)] = 2;

        assert_eq!(map[pos(0, 0)], 1);
        assert_eq!(map[pos(0, 1)], 2);
        assert_eq!(map[pos(1, 0)], 0);
    }

    // ── PositionMap N=2 tests ──────────────────────────────────────────

    #[test]
    fn position_map_n2_get_returns_correct_pair() {
        let mut map = PositionMap::<f32, 2>::new(0.0);
        *map.get_mut(pos(3, 4)) = [1.0, 2.0];

        assert_eq!(map.get(pos(3, 4)), &[1.0, 2.0]);
        assert_eq!(map.get(pos(0, 0)), &[0.0, 0.0]);
    }

    #[test]
    fn position_map_n2_get_mut_modifies_correctly() {
        let mut map = PositionMap::<i32, 2>::new(0);
        *map.get_mut(pos(7, 7)) = [10, 20];

        assert_eq!(map.get(pos(7, 7)), &[10, 20]);
        assert_eq!(map.get(pos(7, 8)), &[0, 0]);
    }

    // ── PositionMap accessor tests ─────────────────────────────────────

    #[test]
    fn position_map_as_slice_returns_full_data() {
        let map = PositionMap::<f32, 2>::new(1.0);

        assert_eq!(map.as_slice().len(), PositionId::COUNT * 2);
        assert!(map.as_slice().iter().all(|&val| val == 1.0));
    }

    #[test]
    fn position_map_as_mut_slice_allows_modification() {
        let mut map = PositionMap::<i32>::new(0);
        map.as_mut_slice()[0] = 42;

        assert_eq!(map[pos(0, 0)], 42);
    }

    #[test]
    fn position_map_into_vec_returns_owned_data() {
        let mut map = PositionMap::<i32>::new(0);
        map[pos(0, 0)] = 7;

        let data = map.into_vec();

        assert_eq!(data.len(), PositionId::COUNT);
        assert_eq!(data[0], 7);
    }

    // ── PositionSlice stride=1 tests ───────────────────────────────────

    #[test]
    fn position_slice_get_returns_correct_value() {
        let data: Vec<i32> = (0..PositionId::COUNT as i32).collect();
        let slice = PositionSlice::new(&data, 1);

        assert_eq!(slice.get(pos(0, 0)), &[0]);
        assert_eq!(slice.get(pos(0, 1)), &[1]);
        assert_eq!(slice.get(pos(1, 0)), &[15]);
    }

    #[test]
    fn position_slice_stride_returns_stride() {
        let data = vec![0i32; PositionId::COUNT];
        let slice = PositionSlice::new(&data, 1);

        assert_eq!(slice.stride(), 1);
    }

    #[test]
    #[should_panic(expected = "PositionSlice requires at least")]
    fn position_slice_panics_on_too_small_data() {
        let data = vec![0i32; 100];
        let _ = PositionSlice::new(&data, 1);
    }

    #[test]
    fn position_slice_accepts_oversized_data() {
        let data = vec![0i32; PositionId::COUNT + 100];
        let slice = PositionSlice::new(&data, 1);

        assert_eq!(slice.get(pos(0, 0)), &[0]);
    }

    // ── PositionSliceMut stride=1 tests ────────────────────────────────

    #[test]
    fn position_slice_mut_allows_mutation() {
        let mut data = vec![0i32; PositionId::COUNT];
        {
            let mut slice = PositionSliceMut::new(&mut data, 1);
            slice.get_mut(pos(5, 5))[0] = 42;
        }
        assert_eq!(data[5 * 15 + 5], 42);
    }

    #[test]
    #[should_panic(expected = "PositionSliceMut requires at least")]
    fn position_slice_mut_panics_on_too_small_data() {
        let mut data = vec![0i32; 100];
        let _ = PositionSliceMut::new(&mut data, 1);
    }

    // ── PositionSlice stride=2 tests ───────────────────────────────────

    #[test]
    fn position_slice_stride2_wraps_flat_data() {
        let data: Vec<f32> = (0..PositionId::COUNT * 2).map(|idx| idx as f32).collect();
        let slice = PositionSlice::new(&data, 2);

        assert_eq!(slice.stride(), 2);
        assert_eq!(slice.get(pos(0, 0)), &[0.0, 1.0]);
        assert_eq!(slice.get(pos(0, 1)), &[2.0, 3.0]);
        assert_eq!(slice.get(pos(1, 0)), &[30.0, 31.0]);
    }

    #[test]
    #[should_panic(expected = "PositionSlice requires at least")]
    fn position_slice_stride2_panics_on_too_small_data() {
        let data = vec![0.0f32; 100];
        let _ = PositionSlice::new(&data, 2);
    }

    // ── PositionSliceMut stride=2 tests ────────────────────────────────

    #[test]
    fn position_slice_mut_stride2_allows_mutation() {
        let mut data = vec![0.0f32; PositionId::COUNT * 2];
        {
            let mut slice = PositionSliceMut::new(&mut data, 2);
            let channels = slice.get_mut(pos(5, 5));
            channels[0] = 3.0;
            channels[1] = 4.0;
        }
        let idx = (5 * 15 + 5) * 2;
        assert_eq!(data[idx], 3.0);
        assert_eq!(data[idx + 1], 4.0);
    }

    #[test]
    #[should_panic(expected = "PositionSliceMut requires at least")]
    fn position_slice_mut_stride2_panics_on_too_small_data() {
        let mut data = vec![0.0f32; 100];
        let _ = PositionSliceMut::new(&mut data, 2);
    }
}
