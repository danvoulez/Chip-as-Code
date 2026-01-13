<div align="center">

# 🔲 chip_as_code

**Semantic Chips — Computation as Text DNA**

[![Crates.io](https://img.shields.io/crates/v/chip_as_code.svg)](https://crates.io/crates/chip_as_code)
[![Documentation](https://docs.rs/chip_as_code/badge.svg)](https://docs.rs/chip_as_code)
[![License](https://img.shields.io/crates/l/chip_as_code.svg)](LICENSE-MIT)

*Deterministic boolean circuits you can read, evolve, and prove.*

[Overview](#overview) •
[Installation](#installation) •
[Quick Start](#quick-start) •
[DNA Format](#dna-format) •
[GateBox](#gatebox) •
[Evolution](#evolution) •
[API](#api)

</div>

---

## Overview

**chip_as_code** implements the [Chip as Code](https://logline.foundation) paradigm: policy logic as compilable, auditable, and evolvable text files.

```
┌─────────────────────────────────────────────────────────────┐
│                    SEMANTIC CHIP                            │
│                                                             │
│   CHIP v0                     ┌───┐                        │
│   FEATURES n=3          f0───▶│AND│──┐                     │
│   GATES m=1             f1───▶│   │  │    ┌───┐           │
│   g0 = AND(f0,f1,f2)    f2───▶└───┘  └───▶│OUT│──▶ 0|1    │
│   OUTPUT = g0                        g0──▶│   │           │
│                                           └───┘           │
└─────────────────────────────────────────────────────────────┘
```

### Key Features

| Feature | Description |
|---------|-------------|
| **📝 Text DNA** | Human-readable `.chip` files define boolean circuits |
| **🔐 Deterministic** | Same input → same output, always |
| **🧬 Evolvable** | Discrete evolution learns circuits from data |
| **⚖️ No-Guess Law** | GateBox enforces constitutional checkpoints |
| **🔗 Provable** | BLAKE3 hashes + JSON✯Atomic CIDs for every artifact |

### Part of the LogLine Ecosystem

```toml
[dependencies]
chip_as_code = "0.1"   # This crate
logline = "0.1"        # TDLN + JSON✯Atomic bundle
ubl-ledger = "0.1"     # Append-only canonical ledger
```

> All crates by [danvoulez](https://crates.io/users/danvoulez)

---

## Installation

### As a library

```toml
[dependencies]
chip_as_code = "0.1"
```

### As a CLI

```bash
cargo install chip_as_code
```

---

## Quick Start

### 30 seconds to your first chip

```bash
# Run the demo (commit, ghost, reject scenarios)
chip demo

# Evolve a chip that learns XOR
chip evolve --task xor --seed 1337 --generations 100

# Verify the result
chip replay --task xor --seed 1337 --chip out/best_chip.txt

# Check the golden reference
chip replay --task xor --seed 1337 --chip examples/xor_golden.chip
```

### Use as a library

```rust
use chip_as_code::{Chip, ChipHash};

fn main() -> anyhow::Result<()> {
    // Parse a chip from text
    let source = r#"
        CHIP v0
        FEATURES n=2
        GATES m=3
        g0 = NOT(f1)
        g1 = AND(f0,g0)
        g2 = NOT(f0)
        OUTPUT = g1
    "#;
    
    let chip = Chip::parse(source)?;
    
    // Evaluate against features
    let features = vec![true, false];  // f0=1, f1=0
    let result = chip.eval(&features)?;
    
    println!("Result: {}", result);  // true
    
    // Get the canonical hash
    let hash = chip.hash();
    println!("Hash: blake3:{}", hash);
    
    Ok(())
}
```

---

## DNA Format

Chips are defined in `.chip` files with a simple, deterministic grammar:

```
CHIP v0
FEATURES n=<N>
GATES m=<M>
g0 = OP(inputs...)
g1 = OP(inputs...)
...
OUTPUT = gK
```

### Operators

| Operator | Syntax | Description |
|----------|--------|-------------|
| `AND` | `AND(a,b,c,...)` | All inputs must be true |
| `OR` | `OR(a,b,c,...)` | At least one input true |
| `NOT` | `NOT(a)` | Negate single input |
| `THRESH` | `THRESH(k,a,b,c,...)` | At least k inputs true |

### References

- `f0`, `f1`, ... `fN-1` — Input features
- `g0`, `g1`, ... `gM-1` — Gate outputs (must reference earlier gates only)

### Example: XOR Gate

```
CHIP v0
FEATURES n=2
GATES m=5
g0 = NOT(f1)
g1 = AND(f0,g0)
g2 = NOT(f0)
g3 = AND(g2,f1)
g4 = OR(g1,g3)
OUTPUT = g4
```

This implements: `XOR(f0, f1) = OR(AND(f0, NOT(f1)), AND(NOT(f0), f1))`

### Canonical Identity

```
chip_hash = blake3(json_atomic_canonical_bytes(chip_ast))
```

Whitespace doesn't affect the hash. Two chips with the same logic have the same hash.

---

## GateBox

The **GateBox** is a constitutional checkpoint that enforces the **No-Guess Law**:

```
┌─────────────────────────────────────────────────────────────┐
│                        GATEBOX                              │
│                                                             │
│   Intent ──────┐                                           │
│                │    ┌──────────────────────────────────┐   │
│   Evidence ────┼───▶│  1. Missing evidence? → GHOST    │   │
│                │    │  2. Unanchored?       → REJECT   │   │
│   Policy ──────┘    │  3. Policy fails?     → REJECT   │   │
│   (chip)            │  4. Otherwise         → COMMIT   │   │
│                     └──────────────────────────────────┘   │
│                                    │                        │
│                                    ▼                        │
│                           ┌──────────────┐                 │
│                           │    Ledger    │                 │
│                           │  (NDJSON)    │                 │
│                           └──────────────┘                 │
└─────────────────────────────────────────────────────────────┘
```

### Verdicts

| Verdict | When | Ledger Append |
|---------|------|---------------|
| **Commit** | All evidence present, anchored, policy passes | ✅ Yes |
| **Ghost** | Missing supporting evidence | ✅ Yes (with question) |
| **Reject** | Unanchored evidence OR policy fails | ❌ No |

### Usage

```rust
use chip_as_code::gatebox::{run_gate, GateVerdict};

let verdict = run_gate(
    "intent.json",
    "evidence.json", 
    "policy.chip",
    "ledger.ndjson"
)?;

match verdict {
    GateVerdict::Commit { cid_hex, chip_hash, proof } => {
        println!("✅ Committed: {}", cid_hex);
    }
    GateVerdict::Ghost { question } => {
        println!("❓ Need more info: {}", question);
    }
    GateVerdict::Reject { reason } => {
        println!("❌ Rejected: {}", reason);
    }
}
```

---

## Evolution

**EvoChip** learns boolean circuits from labeled data using discrete evolution — no gradients, no floats in the core.

```
┌─────────────────────────────────────────────────────────────┐
│                      EVOLUTION                              │
│                                                             │
│   Dataset ──▶ ┌──────────┐    ┌──────────┐    ┌────────┐  │
│               │Population│───▶│ Evaluate │───▶│ Select │  │
│               │  (chips) │    │ (fitness)│    │ (elite)│  │
│               └──────────┘    └──────────┘    └────┬───┘  │
│                     ▲                              │       │
│                     │         ┌──────────┐         │       │
│                     └─────────│  Mutate  │◀────────┘       │
│                               │ Crossover│                 │
│                               └──────────┘                 │
│                                                             │
│   Output: best_chip.txt + training_curve.ndjson            │
└─────────────────────────────────────────────────────────────┘
```

### CLI

```bash
chip evolve \
  --task xor \
  --seed 1337 \
  --generations 200 \
  --population 2000 \
  --elite 50 \
  --max-gates 32
```

### Outputs

| File | Description |
|------|-------------|
| `out/best_chip.txt` | Best evolved chip (canonical DNA) |
| `out/training_curve.ndjson` | Per-generation metrics |
| `out/lineage.ndjson` | Offspring lineage (who came from whom) |
| `out/replay_report.json` | Deterministic replay proof |
| `out/dataset.ndjson` | Training/test data (transparency) |

### Determinism Guarantee

```
Same seed + Same command = Identical artifacts
```

- RNG: `ChaCha20Rng` with explicit seed
- Fitness: Integer per10k accuracy (no floats in core)
- Hashes: BLAKE3 over JSON✯Atomic canonical bytes

---

## API

### Core Types

```rust
/// A semantic chip (boolean circuit)
pub struct Chip {
    pub features: usize,
    pub gates: Vec<Gate>,
    pub output: Ref,
}

/// A gate operation
pub enum GateOp {
    And(Vec<Ref>),
    Or(Vec<Ref>),
    Not(Ref),
    Threshold { k: usize, inputs: Vec<Ref> },
}

/// Reference to a feature or gate
pub enum Ref {
    Feature(usize),
    Gate(usize),
}
```

### Key Methods

```rust
impl Chip {
    /// Parse from text DNA
    pub fn parse(text: &str) -> Result<Self, ChipParseError>;
    
    /// Evaluate against feature vector
    pub fn eval(&self, features: &[bool]) -> Result<bool, EvalError>;
    
    /// Get canonical BLAKE3 hash
    pub fn hash(&self) -> ChipHash;
    
    /// Serialize to canonical text DNA
    pub fn to_dna(&self) -> String;
    
    /// Generate random chip
    pub fn random<R: Rng>(rng: &mut R, features: usize, max_gates: usize) -> Self;
    
    /// Mutate chip (for evolution)
    pub fn mutate<R: Rng>(&self, rng: &mut R, rate_per10k: u32) -> Self;
}
```

---

## CLI Reference

```
chip_as_code — Semantic Chips CLI

USAGE:
    chip <COMMAND>

COMMANDS:
    demo      Run 4 GateBox scenarios (commit, ghost, reject×2)
    gate      Single GateBox decision
    verify    Verify ledger CIDs
    evolve    Evolve a chip from data
    replay    Replay chip against dataset
    stats     Show dataset statistics
    help      Print help

OPTIONS:
    -h, --help       Print help
    -V, --version    Print version
```

### Examples

```bash
# Evolve XOR with custom params
chip evolve --task xor --seed 42 --generations 500 --population 5000

# Verify a ledger file
chip verify --ledger out/ledger.ndjson

# Single gate decision
chip gate \
  --intent examples/intent_good.json \
  --evidence examples/evidence_good.json \
  --policy examples/policy.txt \
  --ledger out/ledger.ndjson
```

---

## Architecture

```
chip_as_code/
├── src/
│   ├── lib.rs        # Public API exports
│   ├── chip_ir.rs    # Chip parsing, evaluation, hashing
│   ├── evolve.rs     # Discrete evolution engine
│   ├── gatebox.rs    # No-Guess Law checkpoint
│   ├── gpu_eval.rs   # Optional GPU acceleration
│   └── main.rs       # CLI
├── examples/
│   ├── xor_golden.chip
│   ├── policy.txt
│   ├── intent_*.json
│   └── evidence_*.json
└── out/              # Generated artifacts
```

---

## Contributing

```bash
# Run tests
cargo test

# Run with debug output
cargo run -- evolve --task xor --seed 1337 --debug

# Build with GPU support
cargo build --release --features gpu
```

---

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

---

<div align="center">

**Part of the [LogLine Foundation](https://logline.foundation) ecosystem**

`chip_as_code` • `logline` • `json_atomic` • `tdln-*` • `ubl-ledger`

[crates.io/users/danvoulez](https://crates.io/users/danvoulez)

</div>
