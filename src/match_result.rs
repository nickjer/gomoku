use crate::outcome::Outcome;

/// The result of a match between two strategies.
#[derive(Debug, Clone)]
pub struct MatchResult {
    outcome: Outcome,
    black_label: String,
    white_label: String,
    turn_count: u32,
    board_state: String,
}

impl MatchResult {
    #[must_use]
    pub fn new(
        outcome: Outcome,
        black_label: String,
        white_label: String,
        turn_count: u32,
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
    pub fn turn_count(&self) -> u32 {
        self.turn_count
    }

    #[must_use]
    pub fn board_state(&self) -> &str {
        &self.board_state
    }

    #[must_use]
    pub fn winner_label(&self) -> Option<&str> {
        match self.outcome {
            Outcome::BlackWins => Some(&self.black_label),
            Outcome::WhiteWins => Some(&self.white_label),
            Outcome::Draw => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn winner_label_returns_black_label_when_black_wins() {
        let result = MatchResult::new(
            Outcome::BlackWins,
            "black_strategy".to_string(),
            "white_strategy".to_string(),
            9,
            String::new(),
        );

        assert_eq!(result.winner_label(), Some("black_strategy"));
    }

    #[test]
    fn winner_label_returns_white_label_when_white_wins() {
        let result = MatchResult::new(
            Outcome::WhiteWins,
            "black_strategy".to_string(),
            "white_strategy".to_string(),
            10,
            String::new(),
        );

        assert_eq!(result.winner_label(), Some("white_strategy"));
    }

    #[test]
    fn winner_label_returns_none_on_draw() {
        let result = MatchResult::new(
            Outcome::Draw,
            "black".to_string(),
            "white".to_string(),
            225,
            String::new(),
        );

        assert_eq!(result.winner_label(), None);
    }

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
