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
- **Board**: 15×15 grid (but can be easily changed in the code), stones are `Black`/`White`/`Empty`
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

- **Fingerprint strategies (NN1-NN4)**: Select moves based on priority ordering of local patterns
- **Conv strategies (ConvTiny, ConvSmall)**: CNN-based policy networks with 3×3 kernels
- **InteractiveStrategy**: TUI-based human input, generic over `Backend` for testability

Move selection for fingerprint-based strategies (NN1-NN4):
1. Calculate fingerprint for each empty position
2. Find fingerprint's index in the gene list (priority)
3. Select position with highest priority (lowest index)
4. Break ties randomly

Move selection for conv strategies:
1. Encode board as 2-channel tensor (own stones, opponent stones)
2. Apply random D8 transform for data augmentation
3. Forward pass through CNN layers
4. Select position with highest policy output (reservoir sampling for ties)
5. Apply inverse transform to get original coordinates

### Game
The `Game` enum represents Gomoku rule variants using `enum_dispatch`:
- **Freestyle**: 5+ in a row wins, no restrictions (current implementation)
- Future variants: Standard, Renju, Caro (commented out)
- Test-only variants: `Stub` (panics if called), `Scripted` (predetermined outcomes)

### Tournament
The `Tournament` enum manages competition formats using `enum_dispatch`:
- **Swiss**: Swiss-system tournament with Buchholz tiebreaker
- Test-only variants: `InputOrder`, `Scripted`

### Evolution
The `Evolver` orchestrates the genetic algorithm. Call `evolve(strategies, rng)` with initial strategies.

- **Fitness**: Swiss tournament ranking (`population_size - rank`)
- **Selection** enum: `TournamentWithReplacement`, `TournamentWithoutReplacement`
- **Crossover** enum: `Order` (OX), `Pmx` (Partially Mapped Crossover)
- **Mutation** enum: `Swap`, `Insert`, `Inversion`
- **Elitism**: Preserve top N performers unchanged each generation
- **Game**: Game variant to use for matches (defaults to `Freestyle`)

### Caching
`CacheRepository` provides lazy-loaded `NeighborCountsCache` instances for each NN level, updated incrementally as stones are placed.

### CLI

**Subcommands:** `evolve`, `play`, `interactive`

The `evolve` command has strategy-type subcommands: `nn1`, `nn2`, `nn3`, `nn4`, `conv-tiny`, `conv-small`.

```bash
# Generate and evolve 16 random NN4 strategies for 10 generations
cargo run --release -- evolve nn4 -p 16 -o tmp/output -g 10

# Evolve ConvSmall CNN strategies
cargo run --release -- evolve conv-small -p 8 -o tmp/output -g 20 --seed 42

# Continue evolving from saved strategies
cargo run --release -- evolve nn4 -i tmp/output -o tmp/output2 -g 50

# Play a game between two strategies
cargo run --release -- play tmp/output/1_*.bin tmp/output/2_*.bin

# Play interactively against a strategy (TUI)
cargo run --release -- interactive tmp/output/1_*.bin
cargo run --release -- interactive tmp/output/1_*.bin --play-as white
```

**Interactive controls:** Arrow keys/hjkl to move cursor, Enter/Space to place stone, q/Esc to quit.

**Output format:** Each strategy is saved as an individual binary file `{rank}_{label}.bin` using postcard serialization.

**Evolve options:**
- `-p/--population` (required without `-i`): Number of random strategies to generate
- `-o/--output` (required): Output directory for evolved strategies
- `-i/--input`: Load strategies from directory (skips random generation)
- `-g/--generations` [10]: Number of generations to evolve
- `-e/--elitism` [2]: Number of top performers preserved each generation
- `-c/--crossover` (order/pmx) [order]: Crossover operator (fingerprint strategies only)
- `--crossover-rate` [0.8]: Crossover probability
- `-m/--mutation` (swap/insert/inversion) [swap]: Mutation operator (fingerprint strategies only)
- `--mutation-rate` [0.1]: Mutation probability
- `--sigma` [0.01]: Gaussian mutation sigma (conv strategies only)
- `--seed`: RNG seed for reproducibility
- `-l/--log-level`: Log level (error/warn/info/debug/trace)

## Code Style

Prefer static dispatch and zero-cost abstractions over dynamic dispatch. Use `enum_dispatch` for polymorphism (e.g., `Game`, `Tournament`, `Crossover`, `Mutation`, `Selection`) rather than trait objects.

- `#[must_use]` on constructors and getters returning owned/computed values
- `const fn` where possible
- Avoid single-letter variable names unless they are incrementing loop indexes (e.g., `i`, `j`)

### Test-Only Enum Variants
Use `#[cfg(test)]` for test-only enum variants. The enum's `Default` impl can return different variants based on `#[cfg(test)]` vs `#[cfg(not(test))]`:
```rust
impl Default for Game {
    fn default() -> Self {
        #[cfg(not(test))]
        { Freestyle.into() }
        #[cfg(test)]
        { Stub.into() }
    }
}
```

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

## Profiling

Use `perf` for CPU profiling on Linux. The release profile includes debug symbols (`debug = true`).

```bash
# Record profile data (use tmp/ directory for output)
perf record -o tmp/perf.data -F 1000 -- ./target/release/gomoku evolve conv-small -p 4 -o tmp/profile -g 1 --seed 42

# View function-level breakdown
perf report -i tmp/perf.data --stdio -g none --percent-limit=0.5

# View instruction-level hotspots
perf annotate -i tmp/perf.data --stdio -s 'function_name' | grep -E '^\s+[0-9]+\.[0-9]+' | sort -rn | head -25

# Check for SIMD instructions (look for vaddps, vmulps, vfma*, ymm/zmm registers)
perf annotate -i tmp/perf.data --stdio | grep -E 'vaddps|vmulps|vfma|ymm|zmm'
```

For flamegraphs, install and use `cargo-flamegraph`:
```bash
cargo install flamegraph
cargo flamegraph --release -o tmp/flamegraph.svg -- evolve conv-small -p 4 -o tmp/profile -g 1
```

**Key optimization insight:** The conv layers use im2col with rank-1 updates for cache efficiency. Patches are gathered into `[patch_size][225]` layout, then accumulated via outer products with loop order `k -> out_ch -> pos` to keep each patch slice in cache across all output channels. Weights are stored in transposed `[patch_size][OUT_C]` layout to avoid runtime transposition.
