use crate::ast::{PolicyBit, PolicyComposition, PolicyNode, SemanticUnit};
use crate::error::Result;
use crate::expression::{Expression, Literal};
use crate::types::Operator;

/// Optimizer for TDLN policies
pub struct Optimizer {
    // Future: add optimization flags/levels
}

impl Optimizer {
    /// Creates a new optimizer
    pub fn new() -> Self {
        Self {}
    }

    /// Optimizes a semantic unit
    pub fn optimize_semantic_unit(&self, unit: &SemanticUnit) -> Result<SemanticUnit> {
        let mut optimized_unit = unit.clone();

        optimized_unit.policies = unit
            .policies
            .iter()
            .map(|policy| self.optimize_policy_node(policy))
            .collect::<Result<Vec<_>>>()?;

        Ok(optimized_unit)
    }

    /// Optimizes a policy node
    fn optimize_policy_node(&self, node: &PolicyNode) -> Result<PolicyNode> {
        match node {
            PolicyNode::Bit(policy) => {
                let optimized = self.optimize_policy_bit(policy)?;
                Ok(PolicyNode::Bit(optimized))
            }
            PolicyNode::Composition(composition) => {
                // For now, just return the composition as-is
                // Future: optimize policy composition (e.g., remove redundant policies)
                Ok(PolicyNode::Composition(composition.clone()))
            }
        }
    }

    /// Optimizes a policy bit by optimizing its condition expression
    pub fn optimize_policy_bit(&self, policy: &PolicyBit) -> Result<PolicyBit> {
        let mut optimized = policy.clone();

        if let Some(condition) = &policy.condition {
            optimized.condition = Some(self.optimize_expression(condition)?);
        }

        Ok(optimized)
    }

    /// Optimizes an expression
    pub fn optimize_expression(&self, expr: &Expression) -> Result<Expression> {
        // First, recursively optimize sub-expressions
        let expr = self.optimize_recursively(expr)?;

        // Then apply optimizations
        let expr = self.fold_constants(&expr)?;
        let expr = self.simplify_logic(&expr)?;
        let expr = self.eliminate_dead_code(&expr)?;

        Ok(expr)
    }

    /// Recursively optimize sub-expressions
    fn optimize_recursively(&self, expr: &Expression) -> Result<Expression> {
        match expr {
            Expression::Binary { operator, left, right } => {
                Ok(Expression::Binary {
                    operator: *operator,
                    left: Box::new(self.optimize_expression(left)?),
                    right: Box::new(self.optimize_expression(right)?),
                })
            }
            Expression::Unary { operator, argument } => {
                Ok(Expression::Unary {
                    operator: *operator,
                    argument: Box::new(self.optimize_expression(argument)?),
                })
            }
            Expression::Conditional { test, consequent, alternate } => {
                Ok(Expression::Conditional {
                    test: Box::new(self.optimize_expression(test)?),
                    consequent: Box::new(self.optimize_expression(consequent)?),
                    alternate: Box::new(self.optimize_expression(alternate)?),
                })
            }
            Expression::FunctionCall { function, arguments } => {
                let optimized_args = arguments
                    .iter()
                    .map(|arg| self.optimize_expression(arg))
                    .collect::<Result<Vec<_>>>()?;

                Ok(Expression::FunctionCall {
                    function: function.clone(),
                    arguments: optimized_args,
                })
            }
            // Literals and context refs don't need optimization
            Expression::Literal(_) | Expression::ContextRef(_) => Ok(expr.clone()),
        }
    }

    /// Constant folding - evaluate constant expressions at compile time
    fn fold_constants(&self, expr: &Expression) -> Result<Expression> {
        match expr {
            Expression::Binary { operator, left, right } => {
                // If both operands are literals, we can evaluate at compile time
                if let (Expression::Literal(l), Expression::Literal(r)) = (left.as_ref(), right.as_ref()) {
                    if let Some(result) = self.eval_constant_binary(*operator, l, r) {
                        return Ok(Expression::Literal(result));
                    }
                }
                Ok(expr.clone())
            }
            Expression::Unary { operator, argument } => {
                // If operand is a literal, we can evaluate at compile time
                if let Expression::Literal(lit) = argument.as_ref() {
                    if let Some(result) = self.eval_constant_unary(*operator, lit) {
                        return Ok(Expression::Literal(result));
                    }
                }
                Ok(expr.clone())
            }
            _ => Ok(expr.clone()),
        }
    }

    /// Simplify logical expressions
    fn simplify_logic(&self, expr: &Expression) -> Result<Expression> {
        match expr {
            Expression::Binary { operator, left, right } => {
                match operator {
                    // x && true => x
                    Operator::And => {
                        if matches!(right.as_ref(), Expression::Literal(Literal::Boolean(true))) {
                            return Ok((**left).clone());
                        }
                        if matches!(left.as_ref(), Expression::Literal(Literal::Boolean(true))) {
                            return Ok((**right).clone());
                        }
                        // x && false => false
                        if matches!(right.as_ref(), Expression::Literal(Literal::Boolean(false)))
                            || matches!(left.as_ref(), Expression::Literal(Literal::Boolean(false))) {
                            return Ok(Expression::boolean(false));
                        }
                    }
                    // x || false => x
                    Operator::Or => {
                        if matches!(right.as_ref(), Expression::Literal(Literal::Boolean(false))) {
                            return Ok((**left).clone());
                        }
                        if matches!(left.as_ref(), Expression::Literal(Literal::Boolean(false))) {
                            return Ok((**right).clone());
                        }
                        // x || true => true
                        if matches!(right.as_ref(), Expression::Literal(Literal::Boolean(true)))
                            || matches!(left.as_ref(), Expression::Literal(Literal::Boolean(true))) {
                            return Ok(Expression::boolean(true));
                        }
                    }
                    _ => {}
                }
                Ok(expr.clone())
            }
            Expression::Unary { operator, argument } => {
                match operator {
                    // !!x => x (double negation)
                    Operator::Not => {
                        if let Expression::Unary { operator: inner_op, argument: inner_arg } = argument.as_ref() {
                            if *inner_op == Operator::Not {
                                return Ok((**inner_arg).clone());
                            }
                        }
                    }
                    _ => {}
                }
                Ok(expr.clone())
            }
            _ => Ok(expr.clone()),
        }
    }

    /// Eliminate dead code - simplify conditionals with constant test expressions
    fn eliminate_dead_code(&self, expr: &Expression) -> Result<Expression> {
        match expr {
            Expression::Conditional { test, consequent, alternate } => {
                // If test is a constant, we can eliminate the dead branch
                if let Expression::Literal(Literal::Boolean(b)) = test.as_ref() {
                    return if *b {
                        Ok((**consequent).clone())
                    } else {
                        Ok((**alternate).clone())
                    };
                }
                Ok(expr.clone())
            }
            _ => Ok(expr.clone()),
        }
    }

    /// Evaluate a constant binary operation
    fn eval_constant_binary(&self, op: Operator, left: &Literal, right: &Literal) -> Option<Literal> {
        match op {
            Operator::And => Some(Literal::Boolean(left.to_bool() && right.to_bool())),
            Operator::Or => Some(Literal::Boolean(left.to_bool() || right.to_bool())),
            Operator::Eq => Some(Literal::Boolean(left == right)),
            Operator::Neq => Some(Literal::Boolean(left != right)),
            Operator::Gt => {
                if let (Literal::Number(l), Literal::Number(r)) = (left, right) {
                    Some(Literal::Boolean(l > r))
                } else {
                    None
                }
            }
            Operator::Lt => {
                if let (Literal::Number(l), Literal::Number(r)) = (left, right) {
                    Some(Literal::Boolean(l < r))
                } else {
                    None
                }
            }
            Operator::Gte => {
                if let (Literal::Number(l), Literal::Number(r)) = (left, right) {
                    Some(Literal::Boolean(l >= r))
                } else {
                    None
                }
            }
            Operator::Lte => {
                if let (Literal::Number(l), Literal::Number(r)) = (left, right) {
                    Some(Literal::Boolean(l <= r))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Evaluate a constant unary operation
    fn eval_constant_unary(&self, op: Operator, operand: &Literal) -> Option<Literal> {
        match op {
            Operator::Not => Some(Literal::Boolean(!operand.to_bool())),
            Operator::Exists => Some(Literal::Boolean(!matches!(operand, Literal::Null))),
            _ => None,
        }
    }
}

impl Default for Optimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::Expression;

    #[test]
    fn test_constant_folding_binary() {
        let optimizer = Optimizer::new();

        // 5 > 3 => true
        let expr = Expression::binary(
            Operator::Gt,
            Expression::number(5.0),
            Expression::number(3.0),
        );
        let optimized = optimizer.optimize_expression(&expr).unwrap();
        assert_eq!(optimized, Expression::boolean(true));

        // 2 + 3 would require arithmetic support
        // For now, we only optimize comparisons
    }

    #[test]
    fn test_constant_folding_unary() {
        let optimizer = Optimizer::new();

        // !true => false
        let expr = Expression::unary(Operator::Not, Expression::boolean(true));
        let optimized = optimizer.optimize_expression(&expr).unwrap();
        assert_eq!(optimized, Expression::boolean(false));

        // !false => true
        let expr = Expression::unary(Operator::Not, Expression::boolean(false));
        let optimized = optimizer.optimize_expression(&expr).unwrap();
        assert_eq!(optimized, Expression::boolean(true));
    }

    #[test]
    fn test_logic_simplification_and() {
        let optimizer = Optimizer::new();

        // x && true => x
        let x = Expression::context_ref(vec!["x"]);
        let expr = Expression::binary(Operator::And, x.clone(), Expression::boolean(true));
        let optimized = optimizer.optimize_expression(&expr).unwrap();
        assert_eq!(optimized, x);

        // x && false => false
        let expr = Expression::binary(Operator::And, x.clone(), Expression::boolean(false));
        let optimized = optimizer.optimize_expression(&expr).unwrap();
        assert_eq!(optimized, Expression::boolean(false));
    }

    #[test]
    fn test_logic_simplification_or() {
        let optimizer = Optimizer::new();

        // x || false => x
        let x = Expression::context_ref(vec!["x"]);
        let expr = Expression::binary(Operator::Or, x.clone(), Expression::boolean(false));
        let optimized = optimizer.optimize_expression(&expr).unwrap();
        assert_eq!(optimized, x);

        // x || true => true
        let expr = Expression::binary(Operator::Or, x.clone(), Expression::boolean(true));
        let optimized = optimizer.optimize_expression(&expr).unwrap();
        assert_eq!(optimized, Expression::boolean(true));
    }

    #[test]
    fn test_double_negation() {
        let optimizer = Optimizer::new();

        // !!x => x
        let x = Expression::context_ref(vec!["x"]);
        let expr = Expression::unary(
            Operator::Not,
            Expression::unary(Operator::Not, x.clone()),
        );
        let optimized = optimizer.optimize_expression(&expr).unwrap();
        assert_eq!(optimized, x);
    }

    #[test]
    fn test_dead_code_elimination() {
        let optimizer = Optimizer::new();

        // if true then A else B => A
        let a = Expression::context_ref(vec!["a"]);
        let b = Expression::context_ref(vec!["b"]);
        let expr = Expression::conditional(Expression::boolean(true), a.clone(), b.clone());
        let optimized = optimizer.optimize_expression(&expr).unwrap();
        assert_eq!(optimized, a);

        // if false then A else B => B
        let expr = Expression::conditional(Expression::boolean(false), a.clone(), b.clone());
        let optimized = optimizer.optimize_expression(&expr).unwrap();
        assert_eq!(optimized, b);
    }

    #[test]
    fn test_complex_optimization() {
        let optimizer = Optimizer::new();

        // ((x && true) || false) => x
        let x = Expression::context_ref(vec!["x"]);
        let expr = Expression::binary(
            Operator::Or,
            Expression::binary(Operator::And, x.clone(), Expression::boolean(true)),
            Expression::boolean(false),
        );
        let optimized = optimizer.optimize_expression(&expr).unwrap();
        assert_eq!(optimized, x);
    }

    #[test]
    fn test_optimize_policy_bit() {
        let optimizer = Optimizer::new();

        // Create a policy with an unoptimized condition
        let policy = PolicyBit::builder()
            .name("test_policy")
            .condition(Expression::binary(
                Operator::And,
                Expression::context_ref(vec!["user", "premium"]),
                Expression::boolean(true),
            ))
            .build()
            .unwrap();

        let optimized = optimizer.optimize_policy_bit(&policy).unwrap();

        // The condition should be optimized to just the context ref
        assert_eq!(
            optimized.condition,
            Some(Expression::context_ref(vec!["user", "premium"]))
        );
    }
}
