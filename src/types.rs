use serde::{Deserialize, Serialize};

/// Node types in the TDLN AST
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeType {
    /// Atomic policy decision
    PolicyBit,
    /// Composition of multiple policies
    PolicyComposition,
    /// Complete semantic unit
    SemanticUnit,
}

/// Types of policy composition
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CompositionType {
    /// Policies execute in sequence
    Sequential,
    /// Policies execute in parallel
    Parallel,
    /// Conditional policy execution
    Conditional,
}

/// Aggregation strategies for parallel policy composition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AggregatorType {
    /// All policies must return true (logical AND)
    All,
    /// At least one policy must return true (logical OR)
    Any,
    /// More than 50% of policies must return true
    Majority,
    /// Weighted aggregation with custom threshold
    Weighted {
        weights: Vec<f64>,
        threshold: f64,
    },
    /// Exactly N policies must return true
    AtLeastN {
        n: usize,
    },
    /// Custom aggregation using an expression
    /// Context will have: results (array of bools), count_true, count_false, total
    #[serde(rename = "CUSTOM")]
    Custom {
        #[serde(skip)]
        expression: Option<Box<crate::expression::Expression>>,
    },
}

/// Value types for policy parameters
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ValueType {
    /// String value
    String,
    /// Numeric value
    Number,
    /// Boolean value
    Boolean,
    /// Context reference
    Context,
    /// Any type
    Any,
}

/// Operators for expressions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Operator {
    // Binary logical operators
    /// Logical AND
    And,
    /// Logical OR
    Or,

    // Comparison operators
    /// Equal to
    Eq,
    /// Not equal to
    Neq,
    /// Greater than
    Gt,
    /// Less than
    Lt,
    /// Greater than or equal to
    Gte,
    /// Less than or equal to
    Lte,
    /// In collection
    In,

    // Unary operators
    /// Logical NOT
    Not,
    /// Exists check
    Exists,
}

impl Operator {
    /// Returns true if this is a binary operator
    pub fn is_binary(&self) -> bool {
        matches!(
            self,
            Operator::And
                | Operator::Or
                | Operator::Eq
                | Operator::Neq
                | Operator::Gt
                | Operator::Lt
                | Operator::Gte
                | Operator::Lte
                | Operator::In
        )
    }

    /// Returns true if this is a unary operator
    pub fn is_unary(&self) -> bool {
        matches!(self, Operator::Not | Operator::Exists)
    }

    /// Returns true if this is a logical operator
    pub fn is_logical(&self) -> bool {
        matches!(self, Operator::And | Operator::Or | Operator::Not)
    }

    /// Returns true if this is a comparison operator
    pub fn is_comparison(&self) -> bool {
        matches!(
            self,
            Operator::Eq
                | Operator::Neq
                | Operator::Gt
                | Operator::Lt
                | Operator::Gte
                | Operator::Lte
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operator_classification() {
        assert!(Operator::And.is_binary());
        assert!(Operator::And.is_logical());

        assert!(Operator::Not.is_unary());
        assert!(Operator::Not.is_logical());

        assert!(Operator::Eq.is_binary());
        assert!(Operator::Eq.is_comparison());

        assert!(!Operator::Gt.is_logical());
        assert!(Operator::Gt.is_comparison());
    }

    #[test]
    fn test_serialization() {
        let op = Operator::Eq;
        let json = serde_json::to_string(&op).unwrap();
        assert_eq!(json, r#""EQ""#);

        let deserialized: Operator = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, Operator::Eq);
    }
}
