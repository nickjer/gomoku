use crate::position_id::PositionId;

/// A map from board positions to values, indexed by `PositionId`.
#[derive(Debug, Clone)]
pub struct PositionMap<T> {
    data: Vec<T>,
}

impl<T: Clone> PositionMap<T> {
    pub fn new(value: T) -> Self {
        Self {
            data: vec![value; PositionId::COUNT],
        }
    }
}

impl<T> PositionMap<T> {
    pub fn from_fn<F>(mut f: F) -> Self
    where
        F: FnMut(PositionId) -> T,
    {
        Self {
            data: PositionId::iter().map(&mut f).collect(),
        }
    }
}

impl<T> std::ops::Index<PositionId> for PositionMap<T> {
    type Output = T;

    fn index(&self, position_id: PositionId) -> &Self::Output {
        &self.data[usize::from(position_id)]
    }
}

impl<T> std::ops::IndexMut<PositionId> for PositionMap<T> {
    fn index_mut(&mut self, position_id: PositionId) -> &mut Self::Output {
        &mut self.data[usize::from(position_id)]
    }
}

impl<T: Copy + Default> PositionMap<T> {
    /// Returns a new map with positions transformed.
    ///
    /// For each position `p`, the new map has `result[f(p)] = self[p]`.
    #[must_use]
    pub fn transformed(&self, f: impl Fn(PositionId) -> PositionId) -> Self {
        PositionSlice::new(&self.data).transformed(f)
    }
}

impl<T: Copy + std::ops::AddAssign> PositionMap<T> {
    /// Accumulates values from `other` with positions transformed.
    ///
    /// For each position `p`, does `self[f(p)] += other[p]`.
    pub fn accumulate_transformed(&mut self, other: &Self, f: impl Fn(PositionId) -> PositionId) {
        for pos in PositionId::iter() {
            self[f(pos)] += other[pos];
        }
    }
}

/// A borrowed view of position-indexed data.
#[derive(Debug)]
pub struct PositionSlice<'a, T> {
    data: &'a [T],
}

impl<'a, T> PositionSlice<'a, T> {
    /// Wraps a slice, asserting it has exactly `PositionId::COUNT` elements.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != PositionId::COUNT`.
    #[must_use]
    pub fn new(data: &'a [T]) -> Self {
        assert_eq!(
            data.len(),
            PositionId::COUNT,
            "PositionSlice requires exactly {} elements",
            PositionId::COUNT
        );
        Self { data }
    }
}

impl<T: Copy + Default> PositionSlice<'_, T> {
    /// Returns a new map with positions transformed.
    ///
    /// For each position `p`, the new map has `result[f(p)] = self[p]`.
    #[must_use]
    pub fn transformed(&self, f: impl Fn(PositionId) -> PositionId) -> PositionMap<T> {
        let mut result = PositionMap::new(T::default());
        for pos in PositionId::iter() {
            result[f(pos)] = self[pos];
        }
        result
    }
}

impl<T> std::ops::Index<PositionId> for PositionSlice<'_, T> {
    type Output = T;

    fn index(&self, position_id: PositionId) -> &Self::Output {
        &self.data[usize::from(position_id)]
    }
}

/// A mutable borrowed view of position-indexed data.
#[derive(Debug)]
pub struct PositionSliceMut<'a, T> {
    data: &'a mut [T],
}

impl<'a, T> PositionSliceMut<'a, T> {
    /// Wraps a mutable slice, asserting it has exactly `PositionId::COUNT` elements.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != PositionId::COUNT`.
    #[must_use]
    pub fn new(data: &'a mut [T]) -> Self {
        assert_eq!(
            data.len(),
            PositionId::COUNT,
            "PositionSliceMut requires exactly {} elements",
            PositionId::COUNT
        );
        Self { data }
    }
}

impl<T> std::ops::Index<PositionId> for PositionSliceMut<'_, T> {
    type Output = T;

    fn index(&self, position_id: PositionId) -> &Self::Output {
        &self.data[usize::from(position_id)]
    }
}

impl<T> std::ops::IndexMut<PositionId> for PositionSliceMut<'_, T> {
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

    #[test]
    fn transformed_applies_function_to_positions() {
        let mut map = PositionMap::new(0i32);
        map[pos(0, 0)] = 1;
        map[pos(0, 1)] = 2;

        // flip_horizontal: (0,0) -> (0,14), (0,1) -> (0,13)
        let transformed = map.transformed(PositionId::flip_horizontal);

        assert_eq!(transformed[pos(0, 14)], 1);
        assert_eq!(transformed[pos(0, 13)], 2);
        assert_eq!(transformed[pos(0, 0)], 0);
    }

    #[test]
    fn transformed_with_identity_preserves_values() {
        let map = PositionMap::from_fn(|p| usize::from(p));

        let transformed = map.transformed(|p| p);

        for p in PositionId::iter() {
            assert_eq!(map[p], transformed[p]);
        }
    }

    #[test]
    fn accumulate_transformed_adds_values() {
        let mut acc = PositionMap::new(0i32);
        let mut other = PositionMap::new(0i32);
        other[pos(0, 0)] = 5;

        // flip_horizontal: (0,0) -> (0,14)
        acc.accumulate_transformed(&other, PositionId::flip_horizontal);

        assert_eq!(acc[pos(0, 14)], 5);
        assert_eq!(acc[pos(0, 0)], 0);
    }

    #[test]
    fn accumulate_transformed_accumulates_multiple_times() {
        let mut acc = PositionMap::new(0i32);
        let other = PositionMap::new(1i32);

        acc.accumulate_transformed(&other, |p| p);
        acc.accumulate_transformed(&other, |p| p);

        assert_eq!(acc[pos(0, 0)], 2);
        assert_eq!(acc[pos(7, 7)], 2);
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
    fn position_slice_transformed_returns_position_map() {
        let data: Vec<i32> = (0..PositionId::COUNT as i32).collect();
        let slice = PositionSlice::new(&data);

        let map = slice.transformed(PositionId::invert);

        // invert: (0,0) -> (14,14), so map[(14,14)] = slice[(0,0)] = 0
        assert_eq!(map[pos(14, 14)], 0);
        // invert: (14,14) -> (0,0), so map[(0,0)] = slice[(14,14)] = 224
        assert_eq!(map[pos(0, 0)], 224);
    }

    #[test]
    #[should_panic(expected = "PositionSlice requires exactly")]
    fn position_slice_panics_on_wrong_length() {
        let data = vec![0i32; 100];
        PositionSlice::new(&data);
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
    #[should_panic(expected = "PositionSliceMut requires exactly")]
    fn position_slice_mut_panics_on_wrong_length() {
        let mut data = vec![0i32; 100];
        PositionSliceMut::new(&mut data);
    }
}
