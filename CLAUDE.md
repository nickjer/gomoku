# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Gomoku is a genetic algorithm tool that evolves strategies to play Gomoku (five in a row) by learning optimal priority orderings of local board patterns.

## Build and Development Commands

```bash
cargo build              # Debug build
cargo build --release    # Optimized release build
cargo test               # Run all tests
cargo test <test_name>   # Run specific test
cargo clippy             # Run linter
cargo fmt                # Format code
```

## Architecture

### Core Game
- **Board**: 15×15 grid, stones are `Black`/`White`/`Empty`
- **Win condition**: 5 consecutive stones (horizontal, vertical, diagonal)
- **PositionId**: Encapsulates board positions (0-224), supports neighbor navigation
- **Offset**: Direction vectors for neighbor calculations

### Fingerprint System
Fingerprints encode local board patterns as `NeighborCounts` tuples of (player, opponent, empty) for neighbor rings:
- **NN1**: 4 orthogonal neighbors at distance 1 (~70 fingerprints)
- **NN2**: NN1 + 4 diagonal neighbors (~1,100 fingerprints)
- **NN3**: NN2 + 4 orthogonal neighbors at distance 2 (~10,000 fingerprints)
- **NN4**: NN3 + 8 knight-move neighbors (~100,000 fingerprints)

### Strategy
Strategies implement the `Strategy` trait. Evolvable strategies additionally implement `EvolvableStrategy` with gene manipulation methods.

Move selection for fingerprint-based strategies (NN1-NN4):
1. Calculate fingerprint for each empty position
2. Find fingerprint's index in the gene list (priority)
3. Select position with highest priority (lowest index)
4. Break ties randomly

### Evolution
The `Evolver` orchestrates the genetic algorithm with configurable parameters:

- **Fitness**: Swiss tournament ranking (`population_size - rank`)
- **Selection** enum: `TournamentWithReplacement`, `TournamentWithoutReplacement`
- **Crossover** enum: `Order` (OX), `Pmx` (Partially Mapped Crossover)
- **Mutation** enum: `Swap`, `Insert`, `Inversion`
- **Elitism**: Preserve top N performers unchanged each generation

### Caching
`CacheRepository` provides lazy-loaded `NeighborCountsCache` instances for each NN level, updated incrementally as stones are placed.

### CLI
The CLI is a work in progress.

## Code Style

Prefer static dispatch and zero-cost abstractions over dynamic dispatch. Use enum variants (e.g., `Crossover`, `Mutation`, `Selection`) rather than trait objects.

- `#[must_use]` on constructors and getters returning owned/computed values
- `const fn` where possible

### Implementation Order

Struct implementations must follow this order:

1. Struct definition
2. Main `impl` block (constructors, public methods, private methods)
3. Standard library traits (`Default`, `Clone`, `Display`, etc.)
4. Custom traits (`RunCrossover`, `RunMutation`, `RunSelection`, etc.)

### Linting
- Clippy `all` and `pedantic` warnings enabled
- `unsafe_code = "forbid"`

### Testing
- Behavior-based test names describing what is verified
- Deterministic RNG seeding for reproducible tests
- Use `cargo llvm-cov` for test coverage
