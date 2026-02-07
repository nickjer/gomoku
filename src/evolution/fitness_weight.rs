/// Newtype for evaluator weights, wrapping `f32`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FitnessWeight(f32);

impl FitnessWeight {
    #[must_use]
    pub fn new(value: f32) -> Self {
        Self(value)
    }

    #[must_use]
    pub fn value(self) -> f32 {
        self.0
    }
}
