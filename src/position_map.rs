use crate::position_id::PositionId;

const BOARD_SIZE: usize = 225;

/// A map from board positions to values, indexed by `PositionId`.
#[derive(Debug, Clone)]
pub struct PositionMap<T> {
    data: Vec<T>,
}

impl<T: Clone> PositionMap<T> {
    pub fn new(value: T) -> Self {
        Self {
            data: vec![value; BOARD_SIZE],
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
        &self.data[usize::from(position_id.index())]
    }
}

impl<T> std::ops::IndexMut<PositionId> for PositionMap<T> {
    fn index_mut(&mut self, position_id: PositionId) -> &mut Self::Output {
        &mut self.data[usize::from(position_id.index())]
    }
}
