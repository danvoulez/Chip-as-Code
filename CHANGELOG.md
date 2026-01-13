# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-01-13

### Added

- **Chip IR**: Parser and evaluator for `.chip` text DNA format
  - Support for AND, OR, NOT, THRESH operators
  - BLAKE3 canonical hashing via JSON✯Atomic
  - Deterministic evaluation

- **GateBox**: Constitutional checkpoint implementing No-Guess Law
  - Commit: All evidence present, anchored, policy passes
  - Ghost: Missing evidence → ask exactly one question
  - Reject: Unanchored evidence or policy failure

- **EvoChip**: Discrete evolution engine for learning boolean circuits
  - Integer-only fitness (per10k accuracy)
  - ChaCha20 seeded RNG for reproducibility
  - Tasks: XOR, policy_hidden

- **CLI**: Complete command-line interface
  - `demo` — Run 4 GateBox scenarios
  - `gate` — Single GateBox decision
  - `verify` — Verify ledger CIDs
  - `evolve` — Evolve chips from data
  - `replay` — Replay chip against dataset
  - `stats` — Show dataset statistics

- **GPU support** (optional feature)
  - wgpu-based parallel evaluation

### Dependencies

- Uses [danvoulez](https://crates.io/users/danvoulez) crates:
  - `logline` for TDLN + JSON✯Atomic
  - `ubl-ledger` for append-only canonical ledger

---

[Unreleased]: https://github.com/LogLine-Foundation/chip_as_code/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/LogLine-Foundation/chip_as_code/releases/tag/v0.1.0
