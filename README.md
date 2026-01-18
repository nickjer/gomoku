# Gomoku

An evolutionary genetic algorithm tool that optimizes strategies for playing Gomoku (five in a row).

## Overview

This tool uses genetic algorithms to evolve AI strategies for the classic board game Gomoku. Strategies are represented as priority orderings over local board patterns (fingerprints), which capture the stone configuration in neighboring positions.

### Strategy Types

- **NN1**: Considers 4 orthogonal neighbors (~70 fingerprints)
- **NN2**: Adds 4 diagonal neighbors (~1,100 fingerprints)
- **NN3**: Adds 4 extended orthogonal neighbors (~10,000 fingerprints)
- **NN4**: Adds 8 knight-move neighbors (~100,000 fingerprints)

### Evolution Parameters

- **Population**: Number of strategies per generation
- **Generations**: Number of evolutionary cycles
- **Mutation**: Swap, Insert, or Inversion operators
- **Crossover**: Order (OX) or Partially Mapped (PMX) crossover
- **Selection**: Tournament selection with configurable size
- **Elitism**: Preserve top performers across generations

## Installation

```bash
git clone <repository-url>
cd gomoku
cargo build --release
```

## Usage

```bash
# Evolve strategies
gomoku evolve --strategy=nn1 --population=32 --generations=20

# Run tournament
gomoku tournament
```

## Development

```bash
# Build
cargo build

# Run tests
cargo test

# Run lints
cargo clippy

# Format code
cargo fmt
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
