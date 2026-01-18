/// Identifier for caches that can be lazily activated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheId {
    NeighborNN1,
    NeighborNN2,
    NeighborNN3,
    NeighborNN4,
}
