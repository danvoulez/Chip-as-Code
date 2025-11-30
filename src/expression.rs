use serde::{Deserialize, Serialize};
use crate::types::Operator;
use std::collections::BTreeMap;

/// Context reference path
pub type ContextPath = Vec<String>;

/// Value types that can appear in expressions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "lowercase")]
pub enum Literal {
    /// String literal
    String(String),
    /// Numeric literal (stored as f64 for flexibility)
    Number(f64),
    /// Boolean literal
    Boolean(bool),
    /// Null value
    Null,
    /// Object literal
    Object(BTreeMap<String, Literal>),
    /// Array literal
    Array(Vec<Literal>),
}

impl Literal {
    /// Returns true if the literal evaluates to a truthy value
    pub fn is_truthy(&self) -> bool {
        match self {
            Literal::Boolean(b) => *b,
            Literal::Null => false,
            Literal::Number(n) => *n != 0.0,
            Literal::String(s) => !s.is_empty(),
            Literal::Array(a) => !a.is_empty(),
            Literal::Object(o) => !o.is_empty(),
        }
    }

    /// Converts the literal to a boolean value
    pub fn to_bool(&self) -> bool {
        self.is_truthy()
    }
}

/// Reference to a value in the evaluation context
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextReference {
    /// Path to the value in the context (e.g., ["user", "account_type"])
    pub path: ContextPath,
    /// Optional fallback value if path not found
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback: Option<Box<Literal>>,
}

impl ContextReference {
    /// Creates a new context reference
    pub fn new<I, S>(path: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            path: path.into_iter().map(|s| s.into()).collect(),
            fallback: None,
        }
    }

    /// Sets a fallback value
    pub fn with_fallback(mut self, fallback: Literal) -> Self {
        self.fallback = Some(Box::new(fallback));
        self
    }
}

/// Expression types in the TDLN AST
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum Expression {
    /// Binary expression (e.g., A AND B, X > Y)
    Binary {
        operator: Operator,
        left: Box<Expression>,
        right: Box<Expression>,
    },

    /// Unary expression (e.g., NOT A, EXISTS X)
    Unary {
        operator: Operator,
        argument: Box<Expression>,
    },

    /// Function call
    FunctionCall {
        function: String,
        arguments: Vec<Expression>,
    },

    /// Reference to a context value
    ContextRef(ContextReference),

    /// Literal value
    Literal(Literal),

    /// Conditional expression (if-then-else)
    Conditional {
        test: Box<Expression>,
        consequent: Box<Expression>,
        alternate: Box<Expression>,
    },
}

impl Expression {
    /// Creates a binary expression
    pub fn binary(operator: Operator, left: Expression, right: Expression) -> Self {
        Expression::Binary {
            operator,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Creates a unary expression
    pub fn unary(operator: Operator, argument: Expression) -> Self {
        Expression::Unary {
            operator,
            argument: Box::new(argument),
        }
    }

    /// Creates a function call expression
    pub fn function_call<S: Into<String>>(function: S, arguments: Vec<Expression>) -> Self {
        Expression::FunctionCall {
            function: function.into(),
            arguments,
        }
    }

    /// Creates a context reference expression
    pub fn context_ref<I, S>(path: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Expression::ContextRef(ContextReference::new(path))
    }

    /// Creates a literal expression
    pub fn literal(value: Literal) -> Self {
        Expression::Literal(value)
    }

    /// Creates a conditional expression
    pub fn conditional(test: Expression, consequent: Expression, alternate: Expression) -> Self {
        Expression::Conditional {
            test: Box::new(test),
            consequent: Box::new(consequent),
            alternate: Box::new(alternate),
        }
    }

    /// Convenience method for creating a string literal
    pub fn string<S: Into<String>>(s: S) -> Self {
        Expression::Literal(Literal::String(s.into()))
    }

    /// Convenience method for creating a number literal
    pub fn number(n: f64) -> Self {
        Expression::Literal(Literal::Number(n))
    }

    /// Convenience method for creating a boolean literal
    pub fn boolean(b: bool) -> Self {
        Expression::Literal(Literal::Boolean(b))
    }

    /// Convenience method for creating a null literal
    pub fn null() -> Self {
        Expression::Literal(Literal::Null)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal_truthy() {
        assert!(Literal::Boolean(true).is_truthy());
        assert!(!Literal::Boolean(false).is_truthy());
        assert!(!Literal::Null.is_truthy());
        assert!(Literal::Number(1.0).is_truthy());
        assert!(!Literal::Number(0.0).is_truthy());
        assert!(Literal::String("hello".to_string()).is_truthy());
        assert!(!Literal::String("".to_string()).is_truthy());
    }

    #[test]
    fn test_context_reference() {
        let ctx_ref = ContextReference::new(vec!["user", "premium"]);
        assert_eq!(ctx_ref.path, vec!["user", "premium"]);
        assert!(ctx_ref.fallback.is_none());

        let ctx_ref_with_fallback =
            ContextReference::new(vec!["user", "premium"]).with_fallback(Literal::Boolean(false));
        assert!(ctx_ref_with_fallback.fallback.is_some());
    }

    #[test]
    fn test_expression_constructors() {
        let expr = Expression::binary(
            Operator::Eq,
            Expression::context_ref(vec!["user", "type"]),
            Expression::string("premium"),
        );

        match expr {
            Expression::Binary { operator, .. } => {
                assert_eq!(operator, Operator::Eq);
            }
            _ => panic!("Expected binary expression"),
        }
    }

    #[test]
    fn test_serialization() {
        let expr = Expression::binary(
            Operator::And,
            Expression::boolean(true),
            Expression::boolean(false),
        );

        let json = serde_json::to_string_pretty(&expr).unwrap();
        let deserialized: Expression = serde_json::from_str(&json).unwrap();

        assert_eq!(expr, deserialized);
    }
}
