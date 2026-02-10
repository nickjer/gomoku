use crate::position_id::PositionId;

/// A map from board positions to values, indexed by `PositionId`.
///
/// Each position stores `N` contiguous elements in Array of Structures layout.
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

/// A borrowed view of position-indexed data.
///
/// Each position holds `N` contiguous elements. For `N = 1` (default), `Index` returns `&T`.
/// For `N > 1`, use `get` to access `&[T; N]` per position.
#[derive(Debug, Clone, Copy)]
pub struct PositionSlice<'a, T, const N: usize = 1> {
    data: &'a [T],
}

impl<'a, T, const N: usize> PositionSlice<'a, T, N> {
    /// Wraps a slice, asserting it has exactly `PositionId::COUNT * N` elements.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != PositionId::COUNT * N`.
    #[must_use]
    pub fn new(data: &'a [T]) -> Self {
        let expected = PositionId::COUNT * N;
        assert_eq!(
            data.len(),
            expected,
            "PositionSlice<_, {N}> requires exactly {expected} elements",
        );
        Self { data }
    }

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
}

impl<T> std::ops::Index<PositionId> for PositionSlice<'_, T, 1> {
    type Output = T;

    fn index(&self, position_id: PositionId) -> &Self::Output {
        &self.data[usize::from(position_id)]
    }
}

/// A mutable borrowed view of position-indexed data.
///
/// Each position holds `N` contiguous elements. For `N = 1` (default), `Index`/`IndexMut`
/// return `&T`/`&mut T`. For `N > 1`, use `get`/`get_mut` to access `&[T; N]` per position.
#[derive(Debug)]
pub struct PositionSliceMut<'a, T, const N: usize = 1> {
    data: &'a mut [T],
}

impl<'a, T, const N: usize> PositionSliceMut<'a, T, N> {
    /// Wraps a mutable slice, asserting it has exactly `PositionId::COUNT * N` elements.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != PositionId::COUNT * N`.
    #[must_use]
    pub fn new(data: &'a mut [T]) -> Self {
        let expected = PositionId::COUNT * N;
        assert_eq!(
            data.len(),
            expected,
            "PositionSliceMut<_, {N}> requires exactly {expected} elements",
        );
        Self { data }
    }

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
}

impl<T> std::ops::Index<PositionId> for PositionSliceMut<'_, T, 1> {
    type Output = T;

    fn index(&self, position_id: PositionId) -> &Self::Output {
        &self.data[usize::from(position_id)]
    }
}

impl<T> std::ops::IndexMut<PositionId> for PositionSliceMut<'_, T, 1> {
    fn index_mut(&mut self, position_id: PositionId) -> &mut Self::Output {
        &mut self.data[usize::from(position_id)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;

    fn pos(row: usize, col: usize) -> PositionId {
        PositionId::from_position(Position::new(row, col))
    }

    // ── N=1 tests ───────────────────────────────────────────────────────

    #[test]
    fn position_map_indexes_correctly() {
        let mut map = PositionMap::new(0i32);
        map[pos(0, 0)] = 1;
        map[pos(0, 1)] = 2;

        assert_eq!(map[pos(0, 0)], 1);
        assert_eq!(map[pos(0, 1)], 2);
        assert_eq!(map[pos(1, 0)], 0);
    }

    #[test]
    fn position_slice_indexes_correctly() {
        let data: Vec<i32> = (0..PositionId::COUNT as i32).collect();
        let slice = PositionSlice::new(&data);

        assert_eq!(slice[pos(0, 0)], 0);
        assert_eq!(slice[pos(0, 1)], 1);
        assert_eq!(slice[pos(1, 0)], 15);
    }

    #[test]
    #[should_panic(expected = "PositionSlice<_, 1> requires exactly")]
    fn position_slice_panics_on_wrong_length() {
        let data = vec![0i32; 100];
        let _ = PositionSlice::<i32>::new(&data);
    }

    #[test]
    fn position_slice_mut_allows_mutation() {
        let mut data = vec![0i32; PositionId::COUNT];
        {
            let mut slice = PositionSliceMut::new(&mut data);
            slice[pos(5, 5)] = 42;
        }
        assert_eq!(data[5 * 15 + 5], 42);
    }

    #[test]
    #[should_panic(expected = "PositionSliceMut<_, 1> requires exactly")]
    fn position_slice_mut_panics_on_wrong_length() {
        let mut data = vec![0i32; 100];
        let _ = PositionSliceMut::<i32>::new(&mut data);
    }

    // ── N=2 tests ───────────────────────────────────────────────────────

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

    #[test]
    fn position_slice_n2_wraps_flat_data() {
        // AoS layout: [pos0_ch0, pos0_ch1, pos1_ch0, pos1_ch1, ...]
        let data: Vec<f32> = (0..PositionId::COUNT * 2)
            .map(|idx| idx as f32)
            .collect();
        let slice = PositionSlice::<f32, 2>::new(&data);

        // Position (0,0) is index 0, so elements at [0, 1]
        assert_eq!(slice.get(pos(0, 0)), &[0.0, 1.0]);
        // Position (0,1) is index 1, so elements at [2, 3]
        assert_eq!(slice.get(pos(0, 1)), &[2.0, 3.0]);
        // Position (1,0) is index 15, so elements at [30, 31]
        assert_eq!(slice.get(pos(1, 0)), &[30.0, 31.0]);
    }

    #[test]
    fn position_slice_mut_n2_allows_mutation() {
        let mut data = vec![0.0f32; PositionId::COUNT * 2];
        {
            let mut slice = PositionSliceMut::<f32, 2>::new(&mut data);
            *slice.get_mut(pos(5, 5)) = [3.0, 4.0];
        }
        let idx = (5 * 15 + 5) * 2;
        assert_eq!(data[idx], 3.0);
        assert_eq!(data[idx + 1], 4.0);
    }

    #[test]
    #[should_panic(expected = "PositionSlice<_, 2> requires exactly")]
    fn position_slice_n2_panics_on_wrong_length() {
        let data = vec![0.0f32; 100];
        let _ = PositionSlice::<f32, 2>::new(&data);
    }

    #[test]
    #[should_panic(expected = "PositionSliceMut<_, 2> requires exactly")]
    fn position_slice_mut_n2_panics_on_wrong_length() {
        let mut data = vec![0.0f32; 100];
        let _ = PositionSliceMut::<f32, 2>::new(&mut data);
    }
}
