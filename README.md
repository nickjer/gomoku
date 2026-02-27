# Gomoku

An evolutionary genetic algorithm tool that optimizes neural network strategies for playing Gomoku (five in a row).

## Overview

This tool uses genetic algorithms to evolve AI strategies for the classic board game Gomoku. Two strategy families are available: convolutional (CNN-based) and cluster (D8-equivariant polynomial features).

### Strategy Types

**Conv strategies** use standard 3x3 convolutions with random D8 data augmentation:
- **ConvTiny**: 32 channels, 2 layers (~10K parameters)
- **ConvSmall**: 64 channels, 4 layers (~112K parameters)

**Cluster strategies** use D8-equivariant polynomial features (sums of products of neighbor values grouped by geometric equivalence classes). Features are inherently D8-invariant, so no runtime augmentation is needed:
- **ClusterTiny**: 32 channels, 2 layers (~10K parameters)
- **ClusterSmall**: 64 channels, 4 layers (~112K parameters)

### Evolution Parameters

- **Population**: Number of strategies per generation
- **Generations**: Number of evolutionary cycles
- **Crossover**: Uniform crossover of CNN weights
- **Mutation**: Gaussian noise added to weights (configurable sigma)
- **Selection**: Tournament selection with configurable size
- **Elitism**: Preserve top performers across generations
- **Fitness**: Weighted combination of tournament ranking, threat defense evaluation, and minimax challenge

## Installation

```bash
git clone <repository-url>
cd gomoku
cargo build --release
```

## Usage

```bash
# Evolve ConvTiny CNN strategies
cargo run --release -- evolve conv-tiny -p 16 -o tmp/output -g 10 --seed 42

# Evolve ConvSmall CNN strategies
cargo run --release -- evolve conv-small -p 8 -o tmp/output -g 20 --seed 42

# Evolve ClusterTiny strategies
cargo run --release -- evolve cluster-tiny -p 16 -o tmp/output -g 10 --seed 42

# Evolve ClusterSmall strategies
cargo run --release -- evolve cluster-small -p 8 -o tmp/output -g 20 --seed 42

# Continue evolving from saved strategies
cargo run --release -- evolve conv-small -i tmp/output -o tmp/output2 -g 50

# Evolve with random opening positions (4 pre-placed stones per game)
cargo run --release -- evolve conv-small -p 8 -o tmp/output -g 20 --opening-moves 4 --seed 42

# Inspect a strategy's summary statistics
cargo run --release -- inspect tmp/output/1_*.bin

# Play a game between two strategies (file path or "minimax" / "minimax:DEPTH")
cargo run --release -- play tmp/output/1_*.bin tmp/output/2_*.bin
cargo run --release -- play tmp/output/1_*.bin minimax:6
cargo run --release -- play tmp/output/1_*.bin minimax:6 --opening-moves 4

# Play interactively against a strategy (TUI)
cargo run --release -- interactive tmp/output/1_*.bin
cargo run --release -- interactive tmp/output/1_*.bin --play-as white
cargo run --release -- interactive minimax:4
cargo run --release -- interactive minimax:4 --opening-moves 6
```

### Evolve Options

- `-p/--population` (required without `-i`): Number of random strategies to generate
- `-o/--output` (required): Output directory for evolved strategies
- `-i/--input`: Load strategies from directory (skips random generation)
- `-g/--generations` [10]: Number of generations to evolve
- `-e/--elitism` [2]: Number of top performers preserved each generation
- `--crossover-rate` [0.8]: Crossover probability
- `--mutation-rate` [0.1]: Mutation probability
- `--sigma` [0.01]: Gaussian mutation sigma
- `--tournament-weight` [1.0]: Tournament evaluator weight (0 to disable)
- `--defense-weight` [0.0]: Threat defense evaluator weight (0 to disable)
- `--minimax-weight` [0.0]: Minimax evaluator weight (0 to disable)
- `--checkpoint-every` [0]: Save a checkpoint every N generations (0 to disable)
- `--minimax-depth` [4]: Minimax search depths (space-separated; evaluated smallest to largest with early cutoff)
- `--minimax-scoring-depth`: Depth for per-move minimax scoring (enables move scoring mode when set)
- `--opening-moves` [0]: Pre-place N random stones before each game (0 = start from empty board; applies to tournament and minimax evaluators)
- `--seed`: RNG seed for reproducibility
- `-l/--log-level`: Log level (error/warn/info/debug/trace)

### Interactive Controls

Arrow keys or hjkl to move cursor, Enter or Space to place stone, q or Esc to quit.

### Output Format

Each strategy is saved as an individual binary file `{rank}_{label}.bin` using postcard serialization.

## Development

```bash
cargo build              # Debug build
cargo build --release    # Optimized release build
cargo test               # Run all tests
cargo test <test_name>   # Run specific test
cargo clippy             # Run linter
cargo fmt                # Format code
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
