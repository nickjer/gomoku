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

/// A heap-allocated map from board positions to `C` channel values each.
#[derive(Debug, Clone)]
pub struct PositionMap<T, const C: usize> {
    data: Box<[[T; C]; PositionId::COUNT]>,
}

impl<T: Copy, const C: usize> PositionMap<T, C> {
    /// Creates a new map with every channel of every position set to `fill`.
    #[must_use]
    pub fn new(fill: T) -> Self {
        // `Box::new([[fill; C]; COUNT])` would build the whole table on the stack first.
        let boxed: Box<[[T; C]]> = vec![[fill; C]; PositionId::COUNT].into_boxed_slice();
        let data = boxed
            .try_into()
            // Drop the `Err` payload (the box) so `expect` needs no `T: Debug`.
            .map_err(drop)
            .expect("boxed slice has exactly PositionId::COUNT elements");
        Self { data }
    }
}

impl<T, const C: usize> PositionMap<T, C> {
    /// Builds a map by calling `f` once per position, in [`PositionId::iter`] order.
    #[must_use]
    pub fn from_fn(f: impl FnMut(PositionId) -> [T; C]) -> Self {
        let boxed: Box<[[T; C]]> = PositionId::iter().map(f).collect();
        let data = boxed
            .try_into()
            .map_err(drop)
            .expect("PositionId::iter yields exactly PositionId::COUNT positions");
        Self { data }
    }

    /// Returns the channel values at the given position.
    #[must_use]
    pub const fn get(&self, position: PositionId) -> &[T; C] {
        &self.data[position.to_index()]
    }

    /// Returns the channel values at the given position, mutably.
    pub const fn get_mut(&mut self, position: PositionId) -> &mut [T; C] {
        &mut self.data[position.to_index()]
    }

    /// Iterates over every position's channel values in [`PositionId::iter`] order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &[T; C]> {
        self.data.iter()
    }

    /// Iterates mutably over every position's channel values in
    /// [`PositionId::iter`] order.
    pub fn iter_mut(&mut self) -> impl ExactSizeIterator<Item = &mut [T; C]> {
        self.data.iter_mut()
    }

    /// Returns all values as one flat mutable slice of length
    /// `PositionId::COUNT * C`, position-major.
    pub const fn as_flattened_mut(&mut self) -> &mut [T] {
        self.data.as_flattened_mut()
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
    fn new_fills_every_channel_of_every_position() {
        let map = PositionMap::<i32, 3>::new(42);

        for position in PositionId::iter() {
            assert_eq!(map.get(position), &[42, 42, 42]);
        }
    }

    #[test]
    fn get_mut_modifies_only_the_target_position() {
        let mut map = PositionMap::<f32, 2>::new(0.0);
        *map.get_mut(pos(3, 4)) = [1.0, 2.0];

        assert_eq!(map.get(pos(3, 4)), &[1.0, 2.0]);
        assert_eq!(map.get(pos(0, 0)), &[0.0, 0.0]);
        assert_eq!(map.get(pos(3, 5)), &[0.0, 0.0]);
    }

    #[test]
    fn iter_mut_visits_positions_in_id_order() {
        let mut map = PositionMap::<usize, 1>::new(0);
        for (position, channels) in PositionId::iter().zip(map.iter_mut()) {
            *channels = [position.to_index()];
        }

        for position in PositionId::iter() {
            assert_eq!(map.get(position), &[position.to_index()]);
        }
    }

    #[test]
    fn iter_visits_positions_in_id_order() {
        let mut map = PositionMap::<usize, 1>::new(0);
        for position in PositionId::iter() {
            *map.get_mut(position) = [position.to_index()];
        }

        for (position, channels) in PositionId::iter().zip(map.iter()) {
            assert_eq!(channels, &[position.to_index()]);
        }
        assert_eq!(map.iter().len(), PositionId::COUNT);
    }

    #[test]
    fn iter_mut_yields_one_item_per_position() {
        let mut map = PositionMap::<u8, 4>::new(0);

        assert_eq!(map.iter_mut().count(), PositionId::COUNT);
    }

    #[test]
    fn as_flattened_mut_is_position_major() {
        let mut map = PositionMap::<f32, 2>::new(0.0);
        let flat = map.as_flattened_mut();

        assert_eq!(flat.len(), PositionId::COUNT * 2);

        // Position 1 occupies flat indices 2 and 3.
        flat[2] = 5.0;
        flat[3] = 6.0;

        assert_eq!(map.get(pos(0, 1)), &[5.0, 6.0]);
        assert_eq!(map.get(pos(0, 0)), &[0.0, 0.0]);
    }

    #[test]
    fn from_fn_calls_once_per_position_in_id_order() {
        let map = PositionMap::<usize, 1>::from_fn(|position| [position.to_index()]);

        for position in PositionId::iter() {
            assert_eq!(map.get(position), &[position.to_index()]);
        }
    }

    #[test]
    fn clone_is_independent_of_original() {
        let mut original = PositionMap::<i32, 1>::new(1);
        let copy = original.clone();
        *original.get_mut(pos(7, 7)) = [99];

        assert_eq!(copy.get(pos(7, 7)), &[1]);
        assert_eq!(original.get(pos(7, 7)), &[99]);
    }
}
