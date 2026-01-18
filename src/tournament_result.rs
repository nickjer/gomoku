/// The result for a single strategy in a tournament.
#[derive(Debug, Clone)]
pub struct TournamentResult {
    strategy_index: usize,
    wins: u32,
    losses: u32,
    draws: u32,
    byes: u32,
}

impl TournamentResult {
    #[must_use]
    pub fn new(strategy_index: usize, wins: u32, losses: u32, draws: u32, byes: u32) -> Self {
        Self {
            strategy_index,
            wins,
            losses,
            draws,
            byes,
        }
    }

    #[must_use]
    pub fn strategy_index(&self) -> usize {
        self.strategy_index
    }

    #[must_use]
    pub fn wins(&self) -> u32 {
        self.wins
    }

    #[must_use]
    pub fn losses(&self) -> u32 {
        self.losses
    }

    #[must_use]
    pub fn draws(&self) -> u32 {
        self.draws
    }

    #[must_use]
    pub fn byes(&self) -> u32 {
        self.byes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accessors_return_correct_values() {
        let result = TournamentResult::new(5, 3, 1, 2, 1);

        assert_eq!(result.strategy_index(), 5);
        assert_eq!(result.wins(), 3);
        assert_eq!(result.losses(), 1);
        assert_eq!(result.draws(), 2);
        assert_eq!(result.byes(), 1);
    }
}
