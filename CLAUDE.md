# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Path Style

Always use relative paths from the project root. Never use absolute paths in commands, tool calls, or file references.

## Project Overview

Gomoku is a genetic algorithm tool that evolves neural network strategies to play Gomoku (five in a row).

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

### Shared Neural Network Utilities (`src/nn/`)
- **`encode_board()`**: Encodes board as a `PositionMap<f32, INPUT_CHANNELS>` (own stones, opponent stones)
- **`select_best_position()`**: Argmax over empty positions with reservoir sampling for ties
- **`he_std()`**: He initialization standard deviation
- **`relu_inplace()`**: ReLU activation in-place

### Strategy
Strategies implement the `Strategy` trait. Evolvable strategies additionally implement `EvolvableStrategy` with gene manipulation methods.

- **Conv strategies (`ConvTiny`, `ConvSmall`)**: CNN-based policy networks with 3×3 kernels in `src/conv/`
- **Cluster strategies (`ClusterTiny`, `ClusterSmall`)**: D8-equivariant polynomial feature networks in `src/cluster/`
- **InteractiveStrategy**: TUI-based human input, generic over `Backend` for testability

Move selection for conv strategies:
1. Encode board as 2-channel tensor (own stones, opponent stones)
2. Apply random D8 transform for data augmentation
3. Forward pass through CNN layers
4. Select position with highest policy output (reservoir sampling for ties)
5. Apply inverse transform to get original coordinates

Move selection for cluster strategies:
1. Encode board as 2-channel tensor (own stones, opponent stones)
2. Forward pass through cluster layers (no D8 augmentation — features are inherently D8-equivariant)
3. Select position with highest policy output (reservoir sampling for ties)

### Cluster Architecture
Cluster layers replace linear 3×3 convolution with **D8-equivariant polynomial features** — sums of products of neighbor values grouped by geometric equivalence classes. This detects topological shapes (bridges, wedges, T-shapes) that linear kernels cannot express in a single layer.

Each position's 8 neighbors are indexed clockwise (N, NE, E, SE, S, SW, W, NW). The 9 features (orders 0-2) are:
- **Order 0**: Center value
- **Order 1**: Ortho sum, Diag sum
- **Order 2**: Wedge-45, Ortho-90, Wedge-135, Ortho-180, Diag-90, Diag-180

**F-truncation**: Features are ordered by polynomial order. `ClusterParams<IN_C, OUT_C, F>` stores only the first F features per channel. Spatial layers use F=9 (all features), the last layer uses F=1 (pointwise — only the center value). The last layer is applied with `pointwise2d`, an inherent method that exists only on `ClusterParams<_, _, 1>`; it feeds the input's flat channel data straight into the dot product with no gathering.

### Game
The `Game` enum represents Gomoku rule variants using `enum_dispatch`:
- **Freestyle**: 5+ in a row wins, no restrictions (current implementation)
- **RandomOpening**: pre-places N random stones before handing off to Freestyle rules
- Future variants: Standard, Renju, Caro (commented out)
- Test-only variants: `Stub` (panics if called), `Scripted` (predetermined outcomes)

Games are played via `Play::play_from(&mut Board, observer, ...)` which accepts a mutable board
(callers can pre-populate it) and a `GameObserver` for per-move hooks. `Play::play()` is a
provided default using an empty board and a no-op observer.

### Tournament
The `Tournament` enum manages competition formats using `enum_dispatch`:
- **Swiss**: Swiss-system tournament with Buchholz tiebreaker
- Test-only variants: `InputOrder`, `Scripted`

### Evolution
The `Evolver` orchestrates the genetic algorithm. Call `evolve(strategies, rng, on_generation)` with initial strategies and a per-generation callback.

- **Fitness**: Weighted combination of tournament ranking, threat defense evaluation, and minimax challenge
- **Selection** enum: `Tournament` (configured via `TournamentMode::WithReplacement` or `WithoutReplacement`)
- **Crossover** enum: `Uniform`
- **Mutation** enum: `Gaussian { sigma }`
- **Elitism**: Preserve top N performers unchanged each generation
- **Game**: Game variant to use for matches (defaults to `Freestyle`)

### CLI

**Subcommands:** `evolve`, `inspect`, `play`, `interactive`

The `evolve` command has strategy-type subcommands: `conv-tiny`, `conv-small`, `cluster-tiny`, `cluster-small`.

```bash
# Evolve ConvTiny CNN strategies
cargo run --release -- evolve conv-tiny -p 16 -o tmp/output -g 10 --seed 42

# Evolve ConvSmall CNN strategies
cargo run --release -- evolve conv-small -p 8 -o tmp/output -g 20 --seed 42

# Evolve ClusterTiny strategies
cargo run --release -- evolve cluster-tiny -p 16 -o tmp/output -g 10 --seed 42

# Evolve ClusterSmall strategies
cargo run --release -- evolve cluster-small -p 8 -o tmp/output -g 20 --seed 42

# Evolve with checkpoints every 5 generations
cargo run --release -- evolve conv-small -p 8 -o tmp/output -g 20 --checkpoint-every 5 --seed 42

# Evolve with random opening positions (4 pre-placed stones per game)
cargo run --release -- evolve conv-small -p 8 -o tmp/output -g 20 --opening-moves 4 --seed 42

# Continue evolving from a checkpoint
cargo run --release -- evolve conv-small -i tmp/output/gen_05 -o tmp/output2 -g 50

# Inspect a strategy's summary statistics
cargo run --release -- inspect tmp/output/gen_20/1_*.bin

# Play a game between two strategies (file path or "minimax" / "minimax:DEPTH")
cargo run --release -- play tmp/output/gen_20/1_*.bin tmp/output/gen_20/2_*.bin
cargo run --release -- play tmp/output/gen_20/1_*.bin minimax:6
cargo run --release -- play tmp/output/gen_20/1_*.bin minimax:6 --opening-moves 4

# Play interactively against a strategy (TUI)
cargo run --release -- interactive tmp/output/gen_20/1_*.bin
cargo run --release -- interactive tmp/output/gen_20/1_*.bin --play-as white
cargo run --release -- interactive minimax:4
cargo run --release -- interactive minimax:4 --opening-moves 6
```

**Interactive controls:** Arrow keys/hjkl to move cursor, Enter/Space to place stone, q/Esc to quit.

**Output format:** The output directory contains `gen_NNN/` sub-directories (zero-padded). Each sub-directory holds individual binary files `{rank}_{label}.bin` using postcard serialization. The final generation is always saved; intermediate checkpoints are controlled by `--checkpoint-every`.

**Evolve options:**
- `-p/--population` (required without `-i`): Number of random strategies to generate
- `-o/--output` (required): Output directory for evolved strategies
- `-i/--input`: Load strategies from directory (skips random generation)
- `-g/--generations` [10]: Number of generations to evolve
- `-e/--elitism` [2]: Number of top performers preserved each generation
- `--crossover-rate` [0.8]: Crossover probability
- `--mutation-rate` [0.1]: Mutation probability
- `--sigma` [0.01]: Gaussian mutation sigma
- `--checkpoint-every` [0]: Save a checkpoint every N generations (0 to disable)
- `--tournament-weight` [1.0]: Tournament evaluator weight (0 to disable)
- `--defense-weight` [0.0]: Threat defense evaluator weight (0 to disable)
- `--minimax-weight` [0.0]: Minimax evaluator weight (0 to disable)
- `--minimax-depth` [4]: Minimax search depths (space-separated; evaluated smallest to largest with early cutoff)
- `--minimax-scoring-depth`: Depth for per-move minimax scoring (enables move scoring mode when set)
- `--opening-moves` [0]: Pre-place N random stones before each game (0 = start from empty board; applies to tournament and minimax evaluators)
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

Be sure to copy the target release builds under the local `tmp/` directory so that you can perform profiling without having to rebuild when comparing two or more builds. You do not need to create the `tmp/` directory as it already exists.

For precise wall-time benchmarking of two binaries, use `hyperfine`. Always pass `--seed` to fix the game so variance comes from the CPU, not game length:
```bash
hyperfine --warmup 1 \
  "tmp/gomoku_baseline play minimax:5 minimax:7 --seed 42" \
  "tmp/gomoku_new      play minimax:5 minimax:7 --seed 42"
```

**Key optimization insights:**

Layer inputs and outputs are `PositionMap<f32, C>`: the channel count is a const generic and the storage is `Box<[[f32; C]; PositionId::COUNT]>`. Stacking layers with mismatched channel counts is a compile error, and every per-position channel loop has an exact, compile-time length. `PositionArray<T>` is the stack/const-friendly sibling used for lookup tables (a `Box` cannot live in a `const`).

Both conv and cluster layers use a two-phase workspace pattern: gather input data into a per-position workspace, then compute dot products against transposed weights (`[STRIDE][OUT_C]` layout). The stride is `IN_C * K * K` for conv and `IN_C * F` for cluster. That product cannot be written as an array length on stable Rust (`generic_const_exprs` is unstable, verified on 1.98), so the workspace is typed with nested arrays instead: `PositionMap<[[f32; K]; K], IN_C>` for conv and `PositionMap<[f32; F], IN_C>` for cluster. Each position is flattened with `as_flattened()` into a `&[f32]` whose length LLVM constant-folds after inlining. The dot-product functions take an iterator of these per-position slices, which lets the `pointwise2d` methods (K=1 / F=1) feed the input's own `[f32; IN_C]` rows in directly with no gathering. Weight rows are typed `&[[f32; OUT_C]]` via `as_chunks::<OUT_C>()`.

**LLVM alias analysis and `&self`:** Hot compute functions must NOT take `&self`. LLVM treats pointers loaded from a struct (e.g., `self.weights.ptr`) as "MayAlias" with fresh heap allocations (like the output buffer), which blocks auto-vectorization. The fix is to extract the weight slice (`&[f32]`) and bias array (`&[f32; OUT_C]`) in the caller and pass them as separate function parameters — LLVM's alias analysis can prove that function-parameter pointers don't alias with in-function allocations. See `ConvParams::conv2d_from_workspace` and `ClusterParams::cluster2d_from_workspace` (associated functions, no `&self`).

**Exact lengths in accessors:** `ConvParams::weights()` / `ClusterParams::weights()` assert the Vec length equals the compile-time constant (`assert_eq!(self.weights.len(), Self::EXPECTED_WEIGHTS)`), which tells LLVM the exact slice length so it can eliminate bounds checks and fully unroll/vectorize loops over these slices. `bias()` goes one step further and returns `&[f32; OUT_C]` (via `try_into`), so the length is in the type.
