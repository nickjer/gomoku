use std::cmp::Ordering;

/// Newtype for fitness scores, wrapping `f32`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FitnessScore(f32);

impl FitnessScore {
    #[must_use]
    pub fn new(value: f32) -> Self {
        Self(value)
    }

    #[must_use]
    pub fn value(self) -> f32 {
        self.0
    }
}

impl Eq for FitnessScore {}

impl PartialOrd for FitnessScore {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FitnessScore {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}
