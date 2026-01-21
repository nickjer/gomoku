use serde::{Deserialize, Serialize};

/// A gene representing a fingerprint index in a strategy's priority ordering.
///
/// Genes are indices into a fingerprint lookup table. Lower positions in a
/// strategy's gene list indicate higher priority for that fingerprint pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Gene(usize);

impl Gene {
    /// Creates a new gene with the given index value.
    #[must_use]
    pub const fn new(value: usize) -> Self {
        Self(value)
    }

    /// Returns the underlying index value.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}
