use crate::ast::{PolicyBit, PolicyComposition, SemanticUnit};
use crate::error::{Result, TdlnError};
use crate::expression::{Expression, Literal};
use crate::hash::Hashable;
use crate::types::Operator;
use crate::TdlnConfig;

/// Canonicalizer applies deterministic normalization rules to TDLN nodes
pub struct Canonicalizer {
    config: TdlnConfig,
}

impl Canonicalizer {
    /// Creates a new canonicalizer with the given configuration
    pub fn new(config: TdlnConfig) -> Self {
        Self { config }
    }

    /// Creates a canonicalizer with default configuration
    pub fn default() -> Self {
        Self::new(TdlnConfig::default())
    }

    /// Canonicalizes a PolicyBit
    pub fn canonicalize_policy_bit(&self, policy: &PolicyBit) -> Result<PolicyBit> {
        let mut canonical = policy.clone();

        if self.config.normalize_whitespace {
            canonical.name = normalize_whitespace(&canonical.name);
            canonical.description = normalize_whitespace(&canonical.description);
        }

        if self.config.sort_parameters {
            canonical
                .parameters
                .sort_by(|a, b| a.name.cmp(&b.name));
        }

        if self.config.standardize_operators {
            if let Some(condition) = &canonical.condition {
                canonical.condition = Some(self.simplify_expression(condition)?);
            }
        }

        // Recompute hash
        canonical.source_hash = canonical
            .compute_hash()
            .map_err(|e| TdlnError::CanonicalizationError(e.to_string()))?;

        Ok(canonical)
    }

    /// Canonicalizes a PolicyComposition
    pub fn canonicalize_policy_composition(
        &self,
        composition: &PolicyComposition,
    ) -> Result<PolicyComposition> {
        let mut canonical = composition.clone();

        if self.config.normalize_whitespace {
            canonical.name = normalize_whitespace(&canonical.name);
            canonical.description = normalize_whitespace(&canonical.description);
        }

        if self.config.sort_parameters {
            canonical.policies.sort();
        }

        // Recompute hash
        canonical.source_hash = canonical
            .compute_hash()
            .map_err(|e| TdlnError::CanonicalizationError(e.to_string()))?;

        Ok(canonical)
    }

    /// Canonicalizes a SemanticUnit
    pub fn canonicalize_semantic_unit(&self, unit: &SemanticUnit) -> Result<SemanticUnit> {
        let mut canonical = unit.clone();

        if self.config.normalize_whitespace {
            canonical.name = normalize_whitespace(&canonical.name);
            canonical.description = normalize_whitespace(&canonical.description);
        }

        if self.config.sort_parameters {
            canonical.inputs.sort_by(|a, b| a.name.cmp(&b.name));
            canonical.outputs.sort_by(|a, b| a.name.cmp(&b.name));
        }

        // Recompute hash
        canonical.source_hash = canonical
            .compute_hash()
            .map_err(|e| TdlnError::CanonicalizationError(e.to_string()))?;

        Ok(canonical)
    }

    /// Simplifies an expression according to boolean algebra rules
    pub fn simplify_expression(&self, expr: &Expression) -> Result<Expression> {
        match expr {
            Expression::Binary {
                operator,
                left,
                right,
            } => {
                let simplified_left = self.simplify_expression(left)?;
                let simplified_right = self.simplify_expression(right)?;

                // Apply simplification rules
                match operator {
                    // A AND true → A
                    Operator::And => {
                        if matches!(simplified_right, Expression::Literal(Literal::Boolean(true)))
                        {
                            return Ok(simplified_left);
                        }
                        if matches!(simplified_left, Expression::Literal(Literal::Boolean(true))) {
                            return Ok(simplified_right);
                        }
                        // A AND false → false
                        if matches!(
                            simplified_right,
                            Expression::Literal(Literal::Boolean(false))
                        ) || matches!(
                            simplified_left,
                            Expression::Literal(Literal::Boolean(false))
                        ) {
                            return Ok(Expression::Literal(Literal::Boolean(false)));
                        }
                    }

                    // A OR false → A
                    Operator::Or => {
                        if matches!(
                            simplified_right,
                            Expression::Literal(Literal::Boolean(false))
                        ) {
                            return Ok(simplified_left);
                        }
                        if matches!(simplified_left, Expression::Literal(Literal::Boolean(false)))
                        {
                            return Ok(simplified_right);
                        }
                        // A OR true → true
                        if matches!(simplified_right, Expression::Literal(Literal::Boolean(true)))
                            || matches!(
                                simplified_left,
                                Expression::Literal(Literal::Boolean(true))
                            )
                        {
                            return Ok(Expression::Literal(Literal::Boolean(true)));
                        }
                    }

                    // A == A → true (if both sides are identical literals)
                    Operator::Eq => {
                        if let (
                            Expression::Literal(left_lit),
                            Expression::Literal(right_lit),
                        ) = (&simplified_left, &simplified_right)
                        {
                            if left_lit == right_lit {
                                return Ok(Expression::Literal(Literal::Boolean(true)));
                            }
                        }
                    }

                    // A != A → false (if both sides are identical literals)
                    Operator::Neq => {
                        if let (
                            Expression::Literal(left_lit),
                            Expression::Literal(right_lit),
                        ) = (&simplified_left, &simplified_right)
                        {
                            if left_lit == right_lit {
                                return Ok(Expression::Literal(Literal::Boolean(false)));
                            }
                        }
                    }

                    _ => {}
                }

                Ok(Expression::Binary {
                    operator: *operator,
                    left: Box::new(simplified_left),
                    right: Box::new(simplified_right),
                })
            }

            Expression::Unary { operator, argument } => {
                let simplified_arg = self.simplify_expression(argument)?;

                // NOT(NOT(A)) → A
                if let Operator::Not = operator {
                    if let Expression::Unary {
                        operator: Operator::Not,
                        argument: inner,
                    } = &simplified_arg
                    {
                        return Ok((**inner).clone());
                    }

                    // NOT(true) → false, NOT(false) → true
                    if let Expression::Literal(Literal::Boolean(b)) = simplified_arg {
                        return Ok(Expression::Literal(Literal::Boolean(!b)));
                    }
                }

                Ok(Expression::Unary {
                    operator: *operator,
                    argument: Box::new(simplified_arg),
                })
            }

            Expression::Conditional {
                test,
                consequent,
                alternate,
            } => {
                let simplified_test = self.simplify_expression(test)?;
                let simplified_consequent = self.simplify_expression(consequent)?;
                let simplified_alternate = self.simplify_expression(alternate)?;

                // If test is a literal boolean, return the appropriate branch
                if let Expression::Literal(Literal::Boolean(b)) = simplified_test {
                    return Ok(if b {
                        simplified_consequent
                    } else {
                        simplified_alternate
                    });
                }

                Ok(Expression::Conditional {
                    test: Box::new(simplified_test),
                    consequent: Box::new(simplified_consequent),
                    alternate: Box::new(simplified_alternate),
                })
            }

            Expression::FunctionCall { function, arguments } => {
                let simplified_args: Result<Vec<_>> = arguments
                    .iter()
                    .map(|arg| self.simplify_expression(arg))
                    .collect();

                Ok(Expression::FunctionCall {
                    function: function.clone(),
                    arguments: simplified_args?,
                })
            }

            // Literals and context refs are already canonical
            expr @ Expression::Literal(_) | expr @ Expression::ContextRef(_) => Ok(expr.clone()),
        }
    }
}

/// Normalizes whitespace in a string (collapses multiple spaces into one, trims)
fn normalize_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::PolicyBit;
    use crate::expression::{Expression, Literal};
    use crate::types::{Operator, ValueType};

    #[test]
    fn test_normalize_whitespace() {
        assert_eq!(normalize_whitespace("  hello   world  "), "hello world");
        assert_eq!(normalize_whitespace("already normalized"), "already normalized");
        assert_eq!(normalize_whitespace(""), "");
    }

    #[test]
    fn test_simplify_and_true() {
        let canonicalizer = Canonicalizer::default();

        let expr = Expression::Binary {
            operator: Operator::And,
            left: Box::new(Expression::context_ref(vec!["user", "premium"])),
            right: Box::new(Expression::boolean(true)),
        };

        let simplified = canonicalizer.simplify_expression(&expr).unwrap();

        match simplified {
            Expression::ContextRef(_) => {} // Success
            _ => panic!("Expected context ref"),
        }
    }

    #[test]
    fn test_simplify_or_false() {
        let canonicalizer = Canonicalizer::default();

        let expr = Expression::Binary {
            operator: Operator::Or,
            left: Box::new(Expression::context_ref(vec!["user", "active"])),
            right: Box::new(Expression::boolean(false)),
        };

        let simplified = canonicalizer.simplify_expression(&expr).unwrap();

        match simplified {
            Expression::ContextRef(_) => {} // Success
            _ => panic!("Expected context ref"),
        }
    }

    #[test]
    fn test_simplify_double_negation() {
        let canonicalizer = Canonicalizer::default();

        let expr = Expression::Unary {
            operator: Operator::Not,
            argument: Box::new(Expression::Unary {
                operator: Operator::Not,
                argument: Box::new(Expression::boolean(true)),
            }),
        };

        let simplified = canonicalizer.simplify_expression(&expr).unwrap();

        match simplified {
            Expression::Literal(Literal::Boolean(true)) => {} // Success
            _ => panic!("Expected true literal"),
        }
    }

    #[test]
    fn test_simplify_conditional_with_true() {
        let canonicalizer = Canonicalizer::default();

        let expr = Expression::Conditional {
            test: Box::new(Expression::boolean(true)),
            consequent: Box::new(Expression::string("yes")),
            alternate: Box::new(Expression::string("no")),
        };

        let simplified = canonicalizer.simplify_expression(&expr).unwrap();

        match simplified {
            Expression::Literal(Literal::String(s)) if s == "yes" => {} // Success
            _ => panic!("Expected 'yes' literal"),
        }
    }

    #[test]
    fn test_canonicalize_policy_bit() {
        let canonicalizer = Canonicalizer::default();

        let policy = PolicyBit::builder()
            .name("  test_policy  ")
            .description("  A   test   policy  ")
            .build()
            .unwrap();

        let canonical = canonicalizer.canonicalize_policy_bit(&policy).unwrap();

        assert_eq!(canonical.name, "test_policy");
        assert_eq!(canonical.description, "A test policy");
    }
}
