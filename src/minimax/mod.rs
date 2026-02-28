mod lines;
mod score;
mod search;
mod search_state;
mod strategy;

pub use search::{DEFAULT_DEPTH, score_move};
pub use strategy::MinimaxStrategy;
