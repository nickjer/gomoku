# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Gomoku is a Rust-based implementation of the classic Gomoku (Five in a Row) board game.

## Build and Development Commands

```bash
# Build
cargo build              # Debug build
cargo build --release    # Optimized release build

# Run
cargo run                # Build and run debug version
cargo run --release      # Build and run release version

# Test
cargo test               # Run all tests
cargo test <test_name>   # Run specific test

# Code quality
cargo clippy             # Run linter
cargo fmt                # Format code
cargo check              # Type-check without building
```

## Technology Stack

- **Language:** Rust (Edition 2024)
- **Package Manager:** Cargo
