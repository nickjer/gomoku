# Gomoku

An evolutionary genetic algorithm tool that optimizes CNN-based strategies for playing Gomoku (five in a row).

## Overview

This tool uses genetic algorithms to evolve AI strategies for the classic board game Gomoku. Strategies are convolutional neural networks (CNNs) that evaluate board positions and select moves.

### Strategy Types

- **ConvTiny**: CNN with 3x3 kernels, 32 channels (~10K parameters)
- **ConvSmall**: CNN with 3x3 kernels, 64 channels (~112K parameters)

Move selection:
1. Encode board as 2-channel tensor (own stones, opponent stones)
2. Apply random D8 symmetry transform for data augmentation
3. Forward pass through CNN layers
4. Select position with highest policy output (reservoir sampling for ties)
5. Apply inverse transform to get original coordinates

### Evolution Parameters

- **Population**: Number of strategies per generation
- **Generations**: Number of evolutionary cycles
- **Crossover**: Uniform crossover of CNN weights
- **Mutation**: Gaussian noise added to weights (configurable sigma)
- **Selection**: Tournament selection with configurable size
- **Elitism**: Preserve top performers across generations
- **Fitness**: Weighted combination of tournament ranking and threat defense evaluation

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

# Continue evolving from saved strategies
cargo run --release -- evolve conv-small -i tmp/output -o tmp/output2 -g 50

# Play a game between two strategies
cargo run --release -- play tmp/output/1_*.bin tmp/output/2_*.bin

# Play interactively against a strategy (TUI)
cargo run --release -- interactive tmp/output/1_*.bin
cargo run --release -- interactive tmp/output/1_*.bin --play-as white
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
