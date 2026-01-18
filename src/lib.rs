#![warn(clippy::as_conversions)]

pub mod board;
pub mod cache_id;
pub mod cache_repository;
pub mod match_result;
pub mod match_runner;
pub mod neighbor_cache;
pub mod neighbor_counts_cache;
pub mod offset;
pub mod outcome;
pub mod pair;
pub mod position;
pub mod position_id;
pub mod position_map;
pub mod simple;
pub mod stone;
pub mod strategy;
pub mod swiss_tournament;
pub mod tournament_result;

#[cfg(test)]
mod test_utils;
