#![warn(clippy::as_conversions)]

pub mod board;
pub mod cli;
pub mod conv;
pub mod evolution;
pub mod game;
pub mod interactive_strategy;
pub mod match_result;
pub mod offset;
pub mod outcome;
pub mod position;
pub mod position_id;
pub mod position_map;
pub mod simple;
pub mod stone;
pub mod strategy;
pub mod threat;
pub mod tournament;

#[cfg(test)]
mod test_utils;
