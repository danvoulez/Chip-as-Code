use thiserror::Error;

/// Result type for TDLN operations
pub type Result<T> = std::result::Result<T, TdlnError>;

/// Errors that can occur during TDLN operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum TdlnError {
    /// Invalid node structure
    #[error("Invalid node structure: {0}")]
    InvalidStructure(String),

    /// Hash mismatch during validation
    #[error("Hash mismatch: expected {expected}, got {actual}")]
    HashMismatch { expected: String, actual: String },

    /// Reference to non-existent policy or function
    #[error("Reference not found: {0}")]
    ReferenceNotFound(String),

    /// Type error in expression
    #[error("Type error: {0}")]
    TypeError(String),

    /// Canonicalization failed
    #[error("Canonicalization failed: {0}")]
    CanonicalizationError(String),

    /// Evaluation error
    #[error("Evaluation error: {0}")]
    EvaluationError(String),

    /// Context path not found
    #[error("Context path not found: {0:?}")]
    ContextPathNotFound(Vec<String>),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Deserialization error
    #[error("Deserialization error: {0}")]
    DeserializationError(String),

    /// Invalid operator for given operands
    #[error("Invalid operator {operator:?} for operands")]
    InvalidOperator { operator: String },

    /// Circular dependency detected
    #[error("Circular dependency detected in policy graph")]
    CircularDependency,

    /// Invalid proof
    #[error("Invalid proof: {0}")]
    InvalidProof(String),

    /// Validation failed
    #[error("Validation failed: {0}")]
    ValidationFailed(String),

    /// Builder error
    #[error("Builder error: {0}")]
    BuilderError(String),

    /// Translation error
    #[error("Translation error: {0}")]
    TranslationError(String),

    /// Generic error
    #[error("{0}")]
    Generic(String),
}

impl From<serde_json::Error> for TdlnError {
    fn from(err: serde_json::Error) -> Self {
        TdlnError::SerializationError(err.to_string())
    }
}

impl From<std::io::Error> for TdlnError {
    fn from(err: std::io::Error) -> Self {
        TdlnError::Generic(err.to_string())
    }
}
