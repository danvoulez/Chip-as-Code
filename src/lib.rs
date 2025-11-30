/*!
# TDLN Core - Truth-Determining Language Normalizer

A robust implementation of the TDLN semantic ISA in Rust.

## Overview

TDLN Core is a deterministic, proof-carrying semantic compiler that converts
human intention into canonical, provable representations. It serves as a
**Semantic Instruction Set Architecture** that bridges natural language and
Domain-Specific Languages with executable policy graphs.

## Core Concepts

- **PolicyBit**: Atomic semantic decision unit (`P_i: context → {0,1}`)
- **SemanticUnit**: Composable graph of policy bits
- **Canonicalization**: Deterministic normalization ensuring semantic equivalence
- **Proofs**: Cryptographic attestations of translation and validation

## Example

```rust
use tdln_core::{PolicyBit, Expression, Operator, ContextReference, Literal, Result};

fn create_policy() -> Result<PolicyBit> {
    // Create a simple policy: "User must be premium"
    let policy = PolicyBit::builder()
        .name("is_premium_user")
        .description("Check if user has premium account type")
        .condition(Expression::Binary {
            operator: Operator::Eq,
            left: Box::new(Expression::ContextRef(
                ContextReference::new(vec!["user", "account_type"])
            )),
            right: Box::new(Expression::Literal(
                Literal::String("premium".to_string())
            )),
        })
        .build()?;
    Ok(policy)
}
# fn main() { create_policy().unwrap(); }
```

## Features

- ✅ Type-safe AST representation
- ✅ Deterministic canonicalization
- ✅ SHA-256 based content addressing
- ✅ Cryptographic proof generation
- ✅ Policy evaluation engine
- ✅ Serde serialization support
*/

pub mod ast;
pub mod builtin;
pub mod canonicalization;
pub mod dsl;
pub mod error;
pub mod evaluator;
pub mod expression;
pub mod hash;
pub mod optimizer;
pub mod proof;
pub mod translator;
pub mod types;

// Re-export commonly used types
pub use ast::{PolicyBit, PolicyComposition, SemanticUnit, TdlnNode, PolicyNode, OutputDefinition, Parameter};
pub use error::{Result, TdlnError};
pub use expression::{Expression, ContextReference, Literal};
pub use types::{Operator, ValueType, NodeType, CompositionType, AggregatorType};
pub use canonicalization::Canonicalizer;
pub use hash::{TdlnHash, Hashable};
pub use proof::{TranslationProof, ValidationProof, TranslationStep, ValidationResult, ExecutionProof, ExecutionStep};
pub use evaluator::{Evaluator, EvaluationContext};
pub use optimizer::Optimizer;
pub use translator::{NlTranslator, TranslationResult};
pub use dsl::DslParser;

/// TDLN Core specification version
pub const SPEC_VERSION: &str = "1.0.0";

/// Default configuration for TDLN Core
pub struct TdlnConfig {
    pub normalize_whitespace: bool,
    pub sort_parameters: bool,
    pub standardize_operators: bool,
    pub deduplicate_expressions: bool,
    pub hash_algorithm: String,
}

impl Default for TdlnConfig {
    fn default() -> Self {
        Self {
            normalize_whitespace: true,
            sort_parameters: true,
            standardize_operators: true,
            deduplicate_expressions: true,
            hash_algorithm: "sha256".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spec_version() {
        assert_eq!(SPEC_VERSION, "1.0.0");
    }

    #[test]
    fn test_default_config() {
        let config = TdlnConfig::default();
        assert!(config.normalize_whitespace);
        assert!(config.sort_parameters);
        assert_eq!(config.hash_algorithm, "sha256");
    }
}
