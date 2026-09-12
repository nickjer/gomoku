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

### Neural Networks (`src/nn/`)
One network, written once. The way a layer looks at the board around each position is chosen as a type.

- **`NeighborhoodEncoder`** (`neighborhood_encoder.rs`): the trait every encoder implements. `type Neighborhood` is what one channel reads at one position, `encode` reads it for every position, `board_symmetry` says whether the board must be turned randomly before scoring (default: leave it), and `fmt_weight_stats` formats a layer's weights for `inspect`. `FlatValues` lets a layer read any neighborhood as one flat list of `f32`.
- **Encoders**: `Square3x3` (`square3x3.rs`) reads a position and its eight neighbors as they are and asks for a random symmetry. `ClusterExpansion<CLUSTERS>` (`cluster_expansion.rs`) sums the neighbors by shape and needs no symmetry. `NoNeighbors` (`no_neighbors.rs`) reads the position alone and is used by the scoring layer.
- **`Layer<Encoder, IN_CHANNELS, OUT_CHANNELS>`** (`layer.rs`): weights and biases for one weighted-sum step. `apply` consumes its input so `NoNeighbors` can pass it through without copying.
- **`NeuralNetwork<Encoder, CHANNELS, LAYERS>`** (`neural_network.rs`): `board_layer`, `middle_layers`, `scoring_layer`; `score_positions` gives every position a score, zeroing negatives between layers. Implements `EvolvableGenes`.
- **`NeuralNetworkStrategy<Encoder, CHANNELS, LAYERS>`** (`neural_network_strategy.rs`): plays the highest-scored empty position. `ConvTiny`/`ConvSmall` are aliases over `Square3x3`; `ClusterTiny`/`ClusterSmall` over `ClusterExpansion<9>`.
- **`BoardSymmetry`** (`board_symmetry.rs`): the eight rotations and reflections of the board, with `apply`, `apply_inverse`, and `apply_to_map`.
- **`board_to_stone_channels`**, `STONE_CHANNELS` (`stone_channels.rs`): the board as two channels per position (own stones, opponent stones).

The one piece of board geometry the encoders share lives with the board types, not in `nn`: `Offset::CENTER_AND_NEIGHBORS`, a position and its eight neighbors clockwise from north. Each encoder owns its own loop over the board; `PositionMap` is only a container.

### Strategy
Strategies implement the `Strategy` trait. Evolvable strategies additionally implement `EvolvableStrategy` with gene manipulation methods.

- **Neural network strategies (`ConvTiny`, `ConvSmall`, `ClusterTiny`, `ClusterSmall`)**: one `NeuralNetworkStrategy` in `src/nn/`, differing only in encoder and size
- **InteractiveStrategy**: TUI-based human input, generic over `Backend` for testability
- **`SavedStrategy`** (`src/cli/saved_strategy.rs`): the one list of strategy kinds the CLI knows, and the `.bin` file format (postcard, label included). One variant per kind; `derive_more::From`/`TryInto` wrap and unwrap it, `strum::EnumDiscriminants` derives `StrategyKind` for the `evolve <KIND>` argument, and the variant doc lines are the `--help` text. Adding a kind touches three places: the alias in `nn`, one variant here, and one arm in `run_evolve` (the compiler flags the missing arm).

Move selection for neural network strategies:
1. Ask the encoder for a board symmetry (random for `Square3x3`, none for `ClusterExpansion`)
2. Turn the board's stone channels by that symmetry
3. Score every position with the network
4. Pick the highest-scored empty position (reservoir sampling for ties)
5. Map the chosen position back through the inverse symmetry

### Cluster Architecture
Cluster layers replace linear 3×3 convolution with a **cluster expansion** — sums of products of neighbor values over clusters (single neighbors, neighbor pairs) grouped into D8-equivalent orbits. This detects topological shapes (bridges, wedges, T-shapes) that linear kernels cannot express in a single layer.

Each position's 8 neighbors are indexed clockwise (N, NE, E, SE, S, SW, W, NW). The 9 clusters (orders 0-2) are:
- **Order 0**: Center value
- **Order 1**: Ortho sum, Diag sum
- **Order 2**: Wedge-45, Ortho-90, Wedge-135, Ortho-180, Diag-90, Diag-180

**Truncating the expansion**: Clusters are ordered by size. `ClusterExpansion<CLUSTERS>` keeps only the first `CLUSTERS` cluster sums per channel; a `CLUSTERS` outside 1..=9 fails to compile. The spatial layers use all 9. The scoring layer uses `NoNeighbors` instead, which reads only the position and gathers nothing.

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
- **Crossover** enum: `Uniform`; `apply(parent1, parent2, rng)` builds the child
- **Mutation** enum: `Gaussian { sigma }`; `apply(&mut genes, rng)` changes them in place
- **EvolvableGenes** (`evolution/genes.rs`): what the operators act on. A gene type exposes its numbers as ordered groups via `gene_groups` / `gene_groups_mut`; operators overwrite inside the groups and can never change the shape. Each operator's `apply` holds the only `match` on its enum, so a new operator is one variant and one arm, and gene types never mention operator names. Operators cut within a group, never across groups, so one-point or two-point crossover would blend every layer rather than move whole layers between parents.
- **Elitism**: Preserve top N performers unchanged each generation
- **Game**: Game variant to use for matches (defaults to `Freestyle`)

### CLI

**Subcommands:** `evolve`, `inspect`, `play`, `interactive`

The `evolve` command takes the strategy kind as its first argument: `conv-tiny`, `conv-small`, `cluster-tiny`, `cluster-small`. With `-i` the kind is read from the files and may be omitted; giving it anyway checks that the files hold that kind.

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

# Continue evolving from a checkpoint (kind comes from the files)
cargo run --release -- evolve -i tmp/output/gen_05 -o tmp/output2 -g 50

# Tournament selection that never draws the same individual twice
cargo run --release -- evolve conv-tiny -p 16 -o tmp/output --selection without-replacement

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

**Output format:** The output directory contains `gen_NNN/` sub-directories (zero-padded). Each sub-directory holds individual binary files `{rank}_{label}.bin`: a postcard-serialized `SavedStrategy` holding the label and the network. The final generation is always saved; intermediate checkpoints are controlled by `--checkpoint-every`.

**Evolve options:**
- `KIND` (required without `-i`): Strategy kind to evolve
- `-p/--population` (required without `-i`, conflicts with it): Number of random strategies to generate
- `-o/--output` (required): Output directory for evolved strategies
- `-i/--input`: Load strategies from directory (skips random generation)
- `-g/--generations` [10]: Number of generations to evolve
- `-e/--elitism` [2]: Number of top performers preserved each generation
- `--selection` [with-replacement]: How tournament selection draws the individuals it compares (`with-replacement` / `without-replacement`)
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

Every layer works in two phases: the encoder produces a `PositionMap<Encoder::Neighborhood, IN_CHANNELS>` (`[f32; 9]` per channel for `Square3x3`, `[f32; CLUSTERS]` for `ClusterExpansion`, a plain `f32` for `NoNeighbors`), then `Layer::weighted_sums` multiplies each position's flattened values against transposed weights (`[INPUTS_PER_POSITION][OUT_CHANNELS]` layout). The stride `IN_CHANNELS * Neighborhood::COUNT` cannot be written as an array length on stable Rust (`generic_const_exprs` is unstable, verified on 1.98), which is why the neighborhood stays a nested array and `FlatValues` both flattens it to `&[f32]` and supplies `COUNT`; the slice length constant-folds after inlining. `NoNeighbors::encode` returns its input unchanged, so `Layer::apply` takes the input by value and the scoring layer has no gather at all. Weight rows are typed `&[[f32; OUT_CHANNELS]]` via `as_chunks::<OUT_CHANNELS>()`.

**Gather layout and `vgatherqps`:** the two encoders' `encode` loops have deliberately different shapes because LLVM vectorizes them differently. `ClusterExpansion` stages the nine neighbor rows neighbor-major on the stack and sums per channel, so LLVM vectorizes the cluster sums across channels with contiguous loads; staging them channel-major made LLVM emit AVX2 `vgatherqps` for the strided reads and cost 14–48% on cluster-small. `Square3x3` writes each neighbor's channel row straight into the output with strided scalar stores; building the same result through a per-channel closure turned that transpose into gathers too (+15% per move). A shared generic gather on `PositionMap` was tried and removed: it either forced one of these shapes on both encoders or needed two methods with one caller each. After touching either loop, check `perf annotate ... | grep -c gather` is 0.

**Two-pass, not fused:** each layer first encodes the whole board into a `PositionMap<Neighborhood, IN_CHANNELS>` and then runs the weighted sums over it. Fusing the two (encode one position onto the stack, multiply it, next position) removes the intermediate buffer and reads simpler, but measured 2–4% slower on identical games even after removing the copy it introduced and reading one position ahead; the per-position encode compiles less tightly between two vector loops than the standalone pass does. Keep the two passes.

**Benchmarking across weight-layout changes:** a change that alters which values a weight multiplies (e.g. neighbor order) changes the moves a seeded network plays, so `evolve --seed` runs different games and wall time is not comparable. Compare instructions per move instead: `perf stat -e instructions:u` divided by the `choose_move` count from `-l trace 2>&1 | grep -c choose_move`.

**LLVM alias analysis and `&self`:** Hot compute functions must NOT take `&self`. LLVM treats pointers loaded from a struct (e.g., `self.weights.ptr`) as "MayAlias" with fresh heap allocations (like the output buffer), which blocks auto-vectorization. The fix is to extract the weight slice (`&[f32]`) and bias array (`&[f32; OUT_CHANNELS]`) in the caller and pass them as separate function parameters — LLVM's alias analysis can prove that function-parameter pointers don't alias with in-function allocations. See `Layer::weighted_sums` (an associated function, no `&self`).

**Exact lengths in accessors:** `Layer::weights()` asserts the Vec length equals the compile-time constant (`assert_eq!(self.weights.len(), Self::WEIGHT_COUNT)`), which tells LLVM the exact slice length so it can eliminate bounds checks and fully unroll/vectorize loops over these slices. `bias()` goes one step further and returns `&[f32; OUT_CHANNELS]` (via `try_into`), so the length is in the type.
