use crate::outcome::Outcome;

/// The result of a match between two strategies.
#[derive(Debug, Clone)]
pub struct MatchResult {
    outcome: Outcome,
    black_label: String,
    white_label: String,
    turn_count: usize,
    board_state: String,
}

impl MatchResult {
    #[must_use]
    pub fn new(
        outcome: Outcome,
        black_label: String,
        white_label: String,
        turn_count: usize,
        board_state: String,
    ) -> Self {
        Self {
            outcome,
            black_label,
            white_label,
            turn_count,
            board_state,
        }
    }

    #[must_use]
    pub fn outcome(&self) -> Outcome {
        self.outcome
    }

    #[must_use]
    pub fn black_label(&self) -> &str {
        &self.black_label
    }

    #[must_use]
    pub fn white_label(&self) -> &str {
        &self.white_label
    }

    #[must_use]
    pub fn turn_count(&self) -> usize {
        self.turn_count
    }

    #[must_use]
    pub fn board_state(&self) -> &str {
        &self.board_state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accessors_return_correct_values() {
        let result = MatchResult::new(
            Outcome::BlackWins,
            "black".to_string(),
            "white".to_string(),
            9,
            "board".to_string(),
        );

        assert_eq!(result.outcome(), Outcome::BlackWins);
        assert_eq!(result.black_label(), "black");
        assert_eq!(result.white_label(), "white");
        assert_eq!(result.turn_count(), 9);
        assert_eq!(result.board_state(), "board");
    }
}
