mod nn1;
mod nn2;
mod nn3;
mod nn4;

pub use nn1::FingerprintNN1IndexCache;
pub use nn2::FingerprintNN2IndexCache;
pub use nn3::FingerprintNN3IndexCache;
pub use nn4::FingerprintNN4IndexCache;

use crate::position_id::PositionId;
use crate::stone::Stone;

/// Trait for fingerprint index caches that provide O(1) fingerprint lookups.
pub trait FingerprintIndexCache {
    /// Returns the fingerprint index for the given position and stone color.
    fn get(&self, position_id: PositionId, current_stone: Stone) -> usize;
}
