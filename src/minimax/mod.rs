mod lines;
mod score;
mod search;
mod search_state;
mod strategy;

pub use search::{DEFAULT_DEPTH, board_score, score_move};
pub use strategy::MinimaxStrategy;
