# TDLN Core - Truth-Determining Language Normalizer

**A robust Rust implementation of the TDLN semantic ISA**

[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)

## Overview

TDLN Core is a deterministic, proof-carrying semantic compiler that converts human intention into canonical, provable representations. It serves as a **Semantic Instruction Set Architecture** that bridges natural language and Domain-Specific Languages with executable policy graphs.

This is the reference implementation of the computer-as-a-protocol vision described in the **"Chip as Code"** whitepaper.

## Core Concepts

### PolicyBit: The Semantic Transistor

A PolicyBit is an atomic semantic decision unit that maps context to binary decisions:

```
P_i: context → {0, 1}
```

Just as a transistor is the primitive for electronic logic, the PolicyBit is the primitive for semantic logic, carrying its own provenance and meaning.

### Semantic Chips

Complex logic is built by composing PolicyBits into graphs:
- **Serial Composition**: `P_Output = P_3(P_2(P_1(context)))`
- **Parallel Composition**: `P_Output = AGGREGATOR(P_A, P_B, P_C)`

A TDLN-Chip is a complete specification of a semantic circuit, defined as text and compilable to any backend.

### The Exponential Jump

The semantic behavior of a 200-million-gate silicon chip can be represented in **~50 KB** of TDLN text. This isn't compression—it's a fundamental shift from computing with material gates to computing with semantic decisions.

## Features

- ✅ **Type-safe AST** - Zero-cost abstractions for semantic structures
- ✅ **Deterministic canonicalization** - Same semantic input → same canonical output
- ✅ **SHA-256 content addressing** - Cryptographic integrity for all artifacts
- ✅ **Proof generation** - Translation, validation, and execution proofs
- ✅ **Policy evaluation engine** - Fast, deterministic policy execution
- ✅ **Serde support** - Full JSON serialization/deserialization
- ✅ **Built-in functions** - String, math, array operations
- ✅ **Composability** - Build complex policies from simple primitives
- ✅ **Natural Language Translation** - Convert policy statements to canonical TDLN
- ✅ **DSL Parser** - Define policies in clean, structured text format
- ✅ **Advanced Aggregators** - Weighted, AtLeastN, and custom aggregators (NEW in v0.2.0)
- ✅ **Policy Optimization** - Constant folding, dead code elimination, logic simplification (NEW in v0.2.0)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
tdln-core = "0.1.0"
```

## Quick Start

```rust
use tdln_core::*;
use serde_json::json;

fn main() -> Result<()> {
    // Create a policy: "Premium users can access"
    let policy = PolicyBit::builder()
        .name("is_premium_user")
        .description("Check if user has premium account")
        .condition(Expression::binary(
            Operator::Eq,
            Expression::context_ref(vec!["user", "account_type"]),
            Expression::string("premium"),
        ))
        .build()?;

    // Evaluate against context
    let context = EvaluationContext::new(json!({
        "user": { "account_type": "premium" }
    }));

    let evaluator = Evaluator::new();
    let result = evaluator.evaluate_policy_bit(&policy, &context)?;

    println!("Policy result: {}", result); // true

    Ok(())
}
```

## Examples

### Natural Language Translator (NEW!)

Convert natural language policy statements to TDLN:

```bash
cargo run --example nl_translator
```

This example shows:
- Natural language pattern matching
- Intent extraction and entity recognition
- Automatic PolicyBit generation
- Translation proofs with cryptographic hashing
- Full evaluation and verification

Supported patterns:
- "Premium users can download files"
- "Only admins can delete"
- "Amount must be less than 1000"
- "User must be verified"

### DSL Parser

Define policies using the TDLN Domain-Specific Language:

```bash
cargo run --example dsl_parser
```

This example shows:
- Clean, readable policy syntax
- Support for operators: `==`, `!=`, `<`, `>`, `<=`, `>=`, `&&`, `||`, `!`
- Context references: `user.account_type`
- Literals: strings, numbers, booleans
- Policy composition with aggregators
- Comment support (`//`)

Example DSL:
```tdln
policy PremiumAccess {
    description: "Check premium status"
    condition: user.account_type == "premium" && user.verified == true
}

composition AllChecks {
    type: parallel
    aggregator: all
    policies: [AuthCheck, AmountCheck, FraudCheck]
}
```

### Advanced Aggregators (NEW!)

Demonstrates weighted, AtLeastN, and custom aggregators:

```bash
cargo run --example advanced_aggregators
```

This example shows:
- **Weighted aggregator**: Assign importance weights to different policies
- **AtLeastN aggregator**: Require minimum number of passing policies
- **Custom aggregator**: Define arbitrary aggregation logic using expressions
- Real-world multi-factor authentication scenario

### Policy Optimization (NEW!)

Shows how policy optimization improves performance and reduces complexity:

```bash
cargo run --example policy_optimizer
```

This example shows:
- **Constant folding**: Evaluate constant expressions at compile-time
- **Logic simplification**: Simplify boolean expressions (e.g., `x AND true → x`)
- **Dead code elimination**: Remove unreachable branches
- **Double negation**: Eliminate redundant negations (`!!x → x`)
- Semantics-preserving transformations

### Premium Download Policy

A complete example demonstrating policy composition and evaluation:

```bash
cargo run --example premium_download
```

This example shows:
- Creating individual policy bits
- Composing policies with ALL aggregator
- Evaluating against different contexts
- Generating execution proofs

### Proof System

Demonstrates the cryptographic proof system:

```bash
cargo run --example proof_system
```

This example shows:
- Translation proofs (source → canonical)
- Validation proofs (structural + semantic)
- Execution proofs (deterministic evaluation)
- Complete audit trail

## Architecture

```
┌─────────────────────────────────────────────────┐
│            TDLN Core Architecture               │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌────────────┐      ┌──────────────┐          │
│  │   Source   │─────▶│  Translation │          │
│  │  (NL/DSL)  │      │    + Proof   │          │
│  └────────────┘      └──────┬───────┘          │
│                             │                   │
│                             ▼                   │
│                   ┌──────────────────┐          │
│                   │  Canonical Core  │          │
│                   │   (PolicyBits)   │          │
│                   └────────┬─────────┘          │
│                            │                    │
│          ┌─────────────────┼──────────────┐     │
│          │                 │              │     │
│          ▼                 ▼              ▼     │
│   ┌─────────────┐   ┌──────────┐   ┌─────────┐ │
│   │ Validation  │   │  Hashing │   │ Evaluat │ │
│   │   + Proof   │   │  SHA-256 │   │  ion    │ │
│   └─────────────┘   └──────────┘   └────┬────┘ │
│                                          │      │
│                                          ▼      │
│                                   ┌──────────┐  │
│                                   │   Proof  │  │
│                                   │ + Result │  │
│                                   └──────────┘  │
└─────────────────────────────────────────────────┘
```

## Project Structure

```
tdln-core/
├── src/
│   ├── lib.rs              # Main entry point
│   ├── types.rs            # Core type definitions
│   ├── expression.rs       # Expression AST
│   ├── ast.rs              # PolicyBit, SemanticUnit
│   ├── hash.rs             # SHA-256 hashing
│   ├── canonicalization.rs # Deterministic normalization
│   ├── proof.rs            # Proof system
│   ├── evaluator.rs        # Policy evaluation engine
│   ├── builtin.rs          # Built-in functions
│   └── error.rs            # Error types
├── examples/
│   ├── premium_download.rs # Complete policy example
│   └── proof_system.rs     # Proof generation example
└── tests/
    └── integration_tests.rs
```

## Testing

Run all tests:

```bash
cargo test
```

Run with verbose output:

```bash
cargo test -- --nocapture
```

Run specific test:

```bash
cargo test test_policy_bit_builder
```

## Performance

TDLN Core is designed for deterministic, reproducible evaluation:

- **Canonicalization**: O(n) where n is AST node count
- **Hashing**: SHA-256 (constant time for given input)
- **Evaluation**: O(p) where p is number of policies
- **Proof generation**: O(s) where s is number of steps

## Roadmap

### v0.2.0 ✅ (Completed)
- [x] Natural language to TDLN translation
- [x] DSL parser
- [x] Advanced aggregators (weighted, AtLeastN, custom)
- [x] Policy optimization (constant folding, logic simplification, dead code elimination)

### v0.3.0
- [ ] Backend compilers (WASM, Python, etc.)
- [ ] DNA ledger integration
- [ ] Distributed policy evaluation
- [ ] Policy versioning and migration

### v1.0.0
- [ ] Production-ready stability
- [ ] Complete specification compliance
- [ ] Performance benchmarks
- [ ] Security audit

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is dual-licensed under:

- MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
- Apache License 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

## Authors

**Dan Voulez** - _Initial work_ - [LogLine Foundation](https://github.com/logline-foundation)

## Acknowledgments

- Inspired by the vision of computation as protocol
- Built on the foundations of modern type theory and proof systems
- Thanks to the Rust community for excellent tooling

## Citation

If you use TDLN Core in your research, please cite:

```bibtex
@software{voulez2025tdln,
  author = {Voulez, Dan},
  title = {TDLN Core: A Semantic ISA for Intention-Driven Computing},
  year = {2025},
  publisher = {LogLine Foundation},
  url = {https://github.com/logline-foundation/tdln-core}
}
```

---

**"The jump from computation bound to the physics of silicon to computation defined by the physics of information itself."**
