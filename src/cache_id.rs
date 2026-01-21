/// Identifier for caches that can be lazily activated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheId {
    FingerprintNN1Index,
    FingerprintNN2Index,
    FingerprintNN3Index,
    FingerprintNN4Index,
}
