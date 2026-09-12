use crate::offset::Offset;
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

    /// Every position's neighborhood as it is: for each channel, the values at
    /// the offsets in offset order, with `empty` for offsets that leave the board.
    #[must_use]
    pub fn neighborhoods<const N: usize>(
        &self,
        offsets: &[Offset; N],
        empty: T,
    ) -> PositionMap<[T; N], C> {
        let mut neighborhoods = PositionMap::new([empty; N]);
        for (position, rows) in PositionId::iter().zip(neighborhoods.iter_mut()) {
            for (slot, &offset) in offsets.iter().enumerate() {
                if let Some(neighbor) = position.offset(offset) {
                    for (row, &value) in rows.iter_mut().zip(self.get(neighbor)) {
                        row[slot] = value;
                    }
                }
            }
        }
        neighborhoods
    }

    /// For every position and channel, reads the values at the offsets (in
    /// offset order, with `empty` for offsets that leave the board) and turns
    /// that row into one new value.
    ///
    /// Use [`Self::neighborhoods`] to keep the rows as they are.
    #[must_use]
    pub fn map_neighborhoods<const N: usize, U: Copy>(
        &self,
        offsets: &[Offset; N],
        empty: T,
        f: impl Fn(&[T; N]) -> U,
    ) -> PositionMap<U, C> {
        // Every value is overwritten below; an all-empty row just gives a
        // valid starting value without asking anything more of `U`.
        let mut mapped = PositionMap::new(f(&[empty; N]));
        for (position, outputs) in PositionId::iter().zip(mapped.iter_mut()) {
            // One row of channel values per offset; off-board offsets stay empty.
            let mut neighbors = [[empty; C]; N];
            for (values, &offset) in neighbors.iter_mut().zip(offsets) {
                if let Some(neighbor) = position.offset(offset) {
                    *values = *self.get(neighbor);
                }
            }
            for (channel, output) in outputs.iter_mut().enumerate() {
                let row: [T; N] = std::array::from_fn(|slot| neighbors[slot][channel]);
                *output = f(&row);
            }
        }
        mapped
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
    fn neighborhoods_reads_offsets_in_table_order() {
        let mut map = PositionMap::<f32, 1>::new(0.0);
        *map.get_mut(pos(7, 7)) = [1.0];
        *map.get_mut(pos(6, 7)) = [2.0];
        *map.get_mut(pos(7, 8)) = [3.0];
        let offsets = [Offset::new(0, 0), Offset::new(-1, 0), Offset::new(0, 1)];

        let rows = map.neighborhoods(&offsets, 0.0);

        assert_eq!(rows.get(pos(7, 7)), &[[1.0, 2.0, 3.0]]);
    }

    #[test]
    fn neighborhoods_reads_off_board_offsets_as_empty() {
        let map = PositionMap::<f32, 1>::new(1.0);
        let offsets = [Offset::new(-1, 0), Offset::new(0, 0), Offset::new(0, -1)];

        let rows = map.neighborhoods(&offsets, -1.0);

        assert_eq!(rows.get(pos(0, 0)), &[[-1.0, 1.0, -1.0]]);
    }

    #[test]
    fn neighborhoods_keeps_channels_separate() {
        let mut map = PositionMap::<f32, 2>::new(0.0);
        *map.get_mut(pos(7, 7)) = [1.0, 2.0];
        *map.get_mut(pos(8, 7)) = [3.0, 4.0];
        let offsets = [Offset::new(0, 0), Offset::new(1, 0)];

        let rows = map.neighborhoods(&offsets, 0.0);

        assert_eq!(rows.get(pos(7, 7)), &[[1.0, 3.0], [2.0, 4.0]]);
    }

    #[test]
    fn map_neighborhoods_matches_neighborhoods_when_f_keeps_the_row() {
        let mut map = PositionMap::<f32, 2>::new(0.0);
        *map.get_mut(pos(7, 7)) = [1.0, 2.0];
        *map.get_mut(pos(8, 7)) = [3.0, 4.0];
        let offsets = [Offset::new(0, 0), Offset::new(1, 0), Offset::new(-1, 0)];

        let mapped = map.map_neighborhoods(&offsets, -1.0, |row| *row);
        let plain = map.neighborhoods(&offsets, -1.0);

        for position in PositionId::iter() {
            assert_eq!(mapped.get(position), plain.get(position));
        }
    }

    #[test]
    fn map_neighborhoods_applies_f_to_every_position_and_channel() {
        let mut map = PositionMap::<f32, 2>::new(0.0);
        *map.get_mut(pos(7, 7)) = [1.0, 10.0];
        *map.get_mut(pos(7, 8)) = [2.0, 20.0];
        let offsets = [Offset::new(0, 0), Offset::new(0, 1)];

        let sums = map.map_neighborhoods(&offsets, 0.0, |row| row.iter().sum::<f32>());

        assert_eq!(sums.get(pos(7, 7)), &[3.0, 30.0]);
        assert_eq!(sums.get(pos(7, 8)), &[2.0, 20.0]);
        assert_eq!(sums.get(pos(0, 0)), &[0.0, 0.0]);
    }

    #[test]
    fn map_neighborhoods_can_change_the_value_type() {
        let map = PositionMap::<f32, 1>::new(1.0);
        let offsets = [Offset::new(0, 0), Offset::new(1, 0)];

        let rows = map.map_neighborhoods(&offsets, 0.0, |row| *row);

        assert_eq!(rows.get(pos(7, 7)), &[[1.0, 1.0]]);
        assert_eq!(rows.get(pos(14, 7)), &[[1.0, 0.0]]);
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
