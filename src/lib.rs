//! # chip_as_code
//!
//! **Semantic Chips — Computation as Text DNA**
//!
//! This crate implements the [Chip as Code](https://logline.foundation) paradigm:
//! policy logic as compilable, auditable, and evolvable text files.
//!
//! ## Quick Start
//!
//! ```rust
//! use chip_as_code::{Chip, ChipHash};
//!
//! let chip = Chip::parse(r#"
//!     CHIP v0
//!     FEATURES n=2
//!     GATES m=1
//!     g0 = AND(f0,f1)
//!     OUTPUT = g0
//! "#).unwrap();
//!
//! let result = chip.eval(&[true, true]).unwrap();
//! assert!(result);
//!
//! let hash: ChipHash = chip.hash();
//! println!("Chip identity: blake3:{}", hash);
//! ```
//!
//! ## Key Concepts
//!
//! - **Chip**: A boolean circuit defined as text DNA
//! - **Gate**: Atomic operations (AND, OR, NOT, THRESH)
//! - **GateBox**: Constitutional checkpoint (Commit/Ghost/Reject)
//! - **Evolution**: Discrete evolution to learn circuits from data
//!
//! ## Features
//!
//! - `default` — Core parsing, evaluation, and evolution
//! - `gpu` — GPU-accelerated evaluation via wgpu
//!
//! ## Part of the LogLine Ecosystem
//!
//! All crates by [danvoulez](https://crates.io/users/danvoulez):
//! - [`logline`](https://crates.io/crates/logline) — TDLN + JSON✯Atomic bundle
//! - [`ubl-ledger`](https://crates.io/crates/ubl-ledger) — Append-only ledger
//! - [`json_atomic`](https://crates.io/crates/json_atomic) — Canonical JSON + CIDs

pub mod chip_ir;
pub mod evolve;
#[cfg(feature = "gpu")]
pub mod gpu_eval;
pub mod gatebox;
pub mod zlayer;

pub use chip_ir::{Chip, ChipHash, EvalError};
