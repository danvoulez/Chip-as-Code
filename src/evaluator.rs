use crate::ast::{PolicyBit, PolicyComposition, PolicyNode, SemanticUnit};
use crate::builtin::BuiltinRegistry;
use crate::error::{Result, TdlnError};
use crate::expression::{ContextReference, Expression, Literal};
use crate::hash::{Hashable, TdlnHash};
use crate::proof::{ExecutionProof, ExecutionStep};
use crate::types::{AggregatorType, CompositionType, Operator};
use std::collections::HashMap;
use std::time::Instant;

/// Context for policy evaluation
#[derive(Debug, Clone)]
pub struct EvaluationContext {
    data: serde_json::Value,
}

impl EvaluationContext {
    /// Creates a new evaluation context from a JSON value
    pub fn new(data: serde_json::Value) -> Self {
        Self { data }
    }

    /// Creates an empty evaluation context
    pub fn empty() -> Self {
        Self {
            data: serde_json::Value::Null,
        }
    }

    /// Gets a value from the context by path
    pub fn get(&self, path: &[String]) -> Option<Literal> {
        let mut current = &self.data;

        for segment in path {
            current = current.get(segment)?;
        }

        json_to_literal(current)
    }

    /// Sets a value in the context
    pub fn set(&mut self, path: &[String], value: Literal) {
        if path.is_empty() {
            self.data = literal_to_json(&value);
            return;
        }

        let mut current = &mut self.data;

        for (i, segment) in path.iter().enumerate() {
            if i == path.len() - 1 {
                if let Some(obj) = current.as_object_mut() {
                    obj.insert(segment.clone(), literal_to_json(&value));
                }
            } else {
                if !current.is_object() {
                    *current = serde_json::json!({});
                }
                current = current
                    .as_object_mut()
                    .unwrap()
                    .entry(segment)
                    .or_insert(serde_json::json!({}));
            }
        }
    }

    /// Computes hash of the context
    pub fn hash(&self) -> Result<TdlnHash> {
        TdlnHash::from_canonical_json(&self.data).map_err(|e| e.into())
    }
}

impl From<serde_json::Value> for EvaluationContext {
    fn from(value: serde_json::Value) -> Self {
        Self::new(value)
    }
}

/// Policy evaluator
pub struct Evaluator {
    builtin_functions: BuiltinRegistry,
}

impl Evaluator {
    /// Creates a new evaluator
    pub fn new() -> Self {
        Self {
            builtin_functions: BuiltinRegistry::new(),
        }
    }

    /// Evaluates a policy bit against a context
    pub fn evaluate_policy_bit(
        &self,
        policy: &PolicyBit,
        context: &EvaluationContext,
    ) -> Result<bool> {
        match &policy.condition {
            Some(condition) => {
                let result = self.evaluate_expression(condition, context)?;
                Ok(result.to_bool())
            }
            None => Ok(policy.fallback),
        }
    }

    /// Evaluates a policy composition
    pub fn evaluate_policy_composition(
        &self,
        composition: &PolicyComposition,
        policies: &HashMap<String, &PolicyBit>,
        context: &EvaluationContext,
    ) -> Result<bool> {
        let results: Result<Vec<bool>> = composition
            .policies
            .iter()
            .map(|policy_id| {
                let policy = policies.get(policy_id).ok_or_else(|| {
                    TdlnError::ReferenceNotFound(format!("Policy not found: {}", policy_id))
                })?;
                self.evaluate_policy_bit(policy, context)
            })
            .collect();

        let results = results?;

        match composition.composition_type {
            CompositionType::Sequential => {
                // All policies must pass in sequence
                Ok(results.iter().all(|&r| r))
            }
            CompositionType::Parallel => {
                if let Some(aggregator) = &composition.aggregator {
                    self.aggregate_results(&results, aggregator)
                } else {
                    // Default to ALL
                    Ok(results.iter().all(|&r| r))
                }
            }
            CompositionType::Conditional => {
                // For conditional, evaluate first policy and use its result to decide
                // This is a simplified implementation
                Ok(results.first().copied().unwrap_or(false))
            }
        }
    }

    /// Evaluates a semantic unit
    pub fn evaluate_semantic_unit(
        &self,
        unit: &SemanticUnit,
        context: &EvaluationContext,
    ) -> Result<HashMap<String, bool>> {
        let _start = Instant::now();

        // Build policy index
        let mut policies: HashMap<String, &PolicyBit> = HashMap::new();
        for policy_node in &unit.policies {
            if let PolicyNode::Bit(policy) = policy_node {
                policies.insert(policy.id.clone(), policy);
            }
        }

        let mut results = HashMap::new();

        // Evaluate all policies
        for policy_node in &unit.policies {
            match policy_node {
                PolicyNode::Bit(policy) => {
                    let result = self.evaluate_policy_bit(policy, context)?;
                    results.insert(policy.id.clone(), result);
                }
                PolicyNode::Composition(composition) => {
                    let result =
                        self.evaluate_policy_composition(composition, &policies, context)?;
                    results.insert(composition.id.clone(), result);
                }
            }
        }

        Ok(results)
    }

    /// Evaluates a semantic unit and produces an execution proof
    pub fn evaluate_with_proof(
        &self,
        unit: &SemanticUnit,
        context: &EvaluationContext,
    ) -> Result<(HashMap<String, bool>, ExecutionProof)> {
        let policy_hash = unit
            .compute_hash()
            .map_err(|e| TdlnError::Generic(e.to_string()))?;
        let context_hash = context
            .hash()
            .map_err(|e| TdlnError::Generic(e.to_string()))?;

        let results = self.evaluate_semantic_unit(unit, context)?;

        // Determine overall result (all outputs must be true)
        let overall_result = results.values().all(|&r| r);

        let mut proof = ExecutionProof::new(policy_hash, context_hash, overall_result);

        // Add execution steps
        for (policy_id, &result) in &results {
            // Find policy name
            let policy_name = unit
                .policies
                .iter()
                .find_map(|p| match p {
                    PolicyNode::Bit(pb) if pb.id == *policy_id => Some(pb.name.clone()),
                    PolicyNode::Composition(pc) if pc.id == *policy_id => Some(pc.name.clone()),
                    _ => None,
                })
                .unwrap_or_else(|| "unknown".to_string());

            proof.add_step(ExecutionStep {
                policy_id: policy_id.clone(),
                policy_name,
                result,
                duration_micros: 0, // Would need proper timing per policy
            });
        }

        Ok((results, proof))
    }

    /// Evaluates an expression against a context
    pub fn evaluate_expression(
        &self,
        expr: &Expression,
        context: &EvaluationContext,
    ) -> Result<Literal> {
        match expr {
            Expression::Literal(lit) => Ok(lit.clone()),

            Expression::ContextRef(ctx_ref) => self.evaluate_context_ref(ctx_ref, context),

            Expression::Binary {
                operator,
                left,
                right,
            } => {
                let left_val = self.evaluate_expression(left, context)?;
                let right_val = self.evaluate_expression(right, context)?;
                self.evaluate_binary_op(*operator, &left_val, &right_val)
            }

            Expression::Unary { operator, argument } => {
                let arg_val = self.evaluate_expression(argument, context)?;
                self.evaluate_unary_op(*operator, &arg_val)
            }

            Expression::Conditional {
                test,
                consequent,
                alternate,
            } => {
                let test_val = self.evaluate_expression(test, context)?;
                if test_val.to_bool() {
                    self.evaluate_expression(consequent, context)
                } else {
                    self.evaluate_expression(alternate, context)
                }
            }

            Expression::FunctionCall { function, arguments } => {
                let arg_vals: Result<Vec<Literal>> = arguments
                    .iter()
                    .map(|arg| self.evaluate_expression(arg, context))
                    .collect();
                let arg_vals = arg_vals?;

                self.builtin_functions.call(function, &arg_vals)
            }
        }
    }

    /// Evaluates a context reference
    fn evaluate_context_ref(
        &self,
        ctx_ref: &ContextReference,
        context: &EvaluationContext,
    ) -> Result<Literal> {
        context.get(&ctx_ref.path).or_else(|| {
            ctx_ref
                .fallback
                .as_ref()
                .map(|fb| (**fb).clone())
        }).ok_or_else(|| {
            TdlnError::ContextPathNotFound(ctx_ref.path.clone())
        })
    }

    /// Evaluates a binary operator
    fn evaluate_binary_op(
        &self,
        operator: Operator,
        left: &Literal,
        right: &Literal,
    ) -> Result<Literal> {
        match operator {
            Operator::And => Ok(Literal::Boolean(left.to_bool() && right.to_bool())),
            Operator::Or => Ok(Literal::Boolean(left.to_bool() || right.to_bool())),

            Operator::Eq => Ok(Literal::Boolean(left == right)),
            Operator::Neq => Ok(Literal::Boolean(left != right)),

            Operator::Gt => match (left, right) {
                (Literal::Number(l), Literal::Number(r)) => Ok(Literal::Boolean(l > r)),
                _ => Err(TdlnError::TypeError(
                    "GT operator requires numeric operands".to_string(),
                )),
            },

            Operator::Lt => match (left, right) {
                (Literal::Number(l), Literal::Number(r)) => Ok(Literal::Boolean(l < r)),
                _ => Err(TdlnError::TypeError(
                    "LT operator requires numeric operands".to_string(),
                )),
            },

            Operator::Gte => match (left, right) {
                (Literal::Number(l), Literal::Number(r)) => Ok(Literal::Boolean(l >= r)),
                _ => Err(TdlnError::TypeError(
                    "GTE operator requires numeric operands".to_string(),
                )),
            },

            Operator::Lte => match (left, right) {
                (Literal::Number(l), Literal::Number(r)) => Ok(Literal::Boolean(l <= r)),
                _ => Err(TdlnError::TypeError(
                    "LTE operator requires numeric operands".to_string(),
                )),
            },

            Operator::In => match (left, right) {
                (item, Literal::Array(arr)) => Ok(Literal::Boolean(arr.contains(item))),
                _ => Err(TdlnError::TypeError(
                    "IN operator requires array as right operand".to_string(),
                )),
            },

            _ => Err(TdlnError::InvalidOperator {
                operator: format!("{:?}", operator),
            }),
        }
    }

    /// Evaluates a unary operator
    fn evaluate_unary_op(&self, operator: Operator, operand: &Literal) -> Result<Literal> {
        match operator {
            Operator::Not => Ok(Literal::Boolean(!operand.to_bool())),
            Operator::Exists => Ok(Literal::Boolean(!matches!(operand, Literal::Null))),
            _ => Err(TdlnError::InvalidOperator {
                operator: format!("{:?}", operator),
            }),
        }
    }

    /// Aggregates boolean results according to aggregator type
    pub fn aggregate_results(&self, results: &[bool], aggregator: &AggregatorType) -> Result<bool> {
        match aggregator {
            AggregatorType::All => Ok(results.iter().all(|&r| r)),
            AggregatorType::Any => Ok(results.iter().any(|&r| r)),
            AggregatorType::Majority => {
                let true_count = results.iter().filter(|&&r| r).count();
                Ok(true_count > results.len() / 2)
            }
            AggregatorType::Weighted { weights, threshold } => {
                if weights.len() != results.len() {
                    return Err(TdlnError::EvaluationError(
                        "Weights count must match results count".to_string(),
                    ));
                }

                let weighted_sum: f64 = results
                    .iter()
                    .zip(weights.iter())
                    .filter(|(&result, _)| result)
                    .map(|(_, &weight)| weight)
                    .sum();

                Ok(weighted_sum >= *threshold)
            }
            AggregatorType::AtLeastN { n } => {
                let true_count = results.iter().filter(|&&r| r).count();
                Ok(true_count >= *n)
            }
            AggregatorType::Custom { expression } => {
                if let Some(expr) = expression {
                    // Build context with aggregation statistics
                    let count_true = results.iter().filter(|&&r| r).count();
                    let count_false = results.len() - count_true;

                    let context = EvaluationContext::new(serde_json::json!({
                        "results": results.to_vec(),
                        "count_true": count_true,
                        "count_false": count_false,
                        "total": results.len(),
                    }));

                    let result = self.evaluate_expression(expr, &context)?;
                    Ok(result.to_bool())
                } else {
                    Err(TdlnError::EvaluationError(
                        "Custom aggregator requires an expression".to_string(),
                    ))
                }
            }
        }
    }
}

impl Default for Evaluator {
    fn default() -> Self {
        Self::new()
    }
}

/// Converts a JSON value to a Literal
fn json_to_literal(value: &serde_json::Value) -> Option<Literal> {
    match value {
        serde_json::Value::Null => Some(Literal::Null),
        serde_json::Value::Bool(b) => Some(Literal::Boolean(*b)),
        serde_json::Value::Number(n) => n.as_f64().map(Literal::Number),
        serde_json::Value::String(s) => Some(Literal::String(s.clone())),
        serde_json::Value::Array(arr) => {
            let literals: Option<Vec<_>> = arr.iter().map(json_to_literal).collect();
            literals.map(Literal::Array)
        }
        serde_json::Value::Object(obj) => {
            let mut map = std::collections::BTreeMap::new();
            for (k, v) in obj {
                if let Some(lit) = json_to_literal(v) {
                    map.insert(k.clone(), lit);
                } else {
                    return None;
                }
            }
            Some(Literal::Object(map))
        }
    }
}

/// Converts a Literal to a JSON value
fn literal_to_json(lit: &Literal) -> serde_json::Value {
    match lit {
        Literal::Null => serde_json::Value::Null,
        Literal::Boolean(b) => serde_json::Value::Bool(*b),
        Literal::Number(n) => {
            serde_json::Number::from_f64(*n).map(serde_json::Value::Number).unwrap_or(serde_json::Value::Null)
        }
        Literal::String(s) => serde_json::Value::String(s.clone()),
        Literal::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(literal_to_json).collect())
        }
        Literal::Object(obj) => {
            let map: serde_json::Map<String, serde_json::Value> =
                obj.iter().map(|(k, v)| (k.clone(), literal_to_json(v))).collect();
            serde_json::Value::Object(map)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::PolicyBit;
    use crate::expression::Expression;
    use crate::types::{Operator, ValueType};
    use serde_json::json;

    #[test]
    fn test_evaluation_context() {
        let ctx = EvaluationContext::new(json!({
            "user": {
                "name": "Alice",
                "age": 30,
                "premium": true
            }
        }));

        let name = ctx.get(&vec!["user".to_string(), "name".to_string()]);
        assert_eq!(name, Some(Literal::String("Alice".to_string())));

        let age = ctx.get(&vec!["user".to_string(), "age".to_string()]);
        assert_eq!(age, Some(Literal::Number(30.0)));
    }

    #[test]
    fn test_evaluate_simple_policy() {
        let evaluator = Evaluator::new();

        let policy = PolicyBit::builder()
            .name("is_premium")
            .condition(Expression::binary(
                Operator::Eq,
                Expression::context_ref(vec!["user", "premium"]),
                Expression::boolean(true),
            ))
            .build()
            .unwrap();

        let ctx = EvaluationContext::new(json!({
            "user": {
                "premium": true
            }
        }));

        let result = evaluator.evaluate_policy_bit(&policy, &ctx).unwrap();
        assert!(result);
    }

    #[test]
    fn test_evaluate_numeric_comparison() {
        let evaluator = Evaluator::new();

        let policy = PolicyBit::builder()
            .name("age_check")
            .condition(Expression::binary(
                Operator::Gte,
                Expression::context_ref(vec!["user", "age"]),
                Expression::number(18.0),
            ))
            .build()
            .unwrap();

        let ctx = EvaluationContext::new(json!({
            "user": {
                "age": 25
            }
        }));

        let result = evaluator.evaluate_policy_bit(&policy, &ctx).unwrap();
        assert!(result);
    }

    #[test]
    fn test_evaluate_with_fallback() {
        let evaluator = Evaluator::new();
        let ctx = EvaluationContext::new(json!({}));

        let policy = PolicyBit::builder()
            .name("test")
            .condition(Expression::context_ref(vec!["missing", "path"]))
            .fallback(true)
            .build()
            .unwrap();

        // Should fail because path doesn't exist and no fallback in context ref
        let result = evaluator.evaluate_policy_bit(&policy, &ctx);
        assert!(result.is_err());
    }

    #[test]
    fn test_aggregator_all() {
        let evaluator = Evaluator::new();

        let result = evaluator
            .aggregate_results(&[true, true, true], &AggregatorType::All)
            .unwrap();
        assert!(result);

        let result = evaluator
            .aggregate_results(&[true, false, true], &AggregatorType::All)
            .unwrap();
        assert!(!result);
    }

    #[test]
    fn test_aggregator_any() {
        let evaluator = Evaluator::new();

        let result = evaluator
            .aggregate_results(&[false, false, true], &AggregatorType::Any)
            .unwrap();
        assert!(result);

        let result = evaluator
            .aggregate_results(&[false, false, false], &AggregatorType::Any)
            .unwrap();
        assert!(!result);
    }

    #[test]
    fn test_aggregator_majority() {
        let evaluator = Evaluator::new();

        let result = evaluator
            .aggregate_results(&[true, true, false], &AggregatorType::Majority)
            .unwrap();
        assert!(result);

        let result = evaluator
            .aggregate_results(&[true, false, false], &AggregatorType::Majority)
            .unwrap();
        assert!(!result);
    }

    #[test]
    fn test_aggregator_weighted() {
        let evaluator = Evaluator::new();

        // Test weighted aggregation with threshold
        let weights = vec![0.5, 0.3, 0.2];
        let threshold = 0.6;

        // 0.5 + 0.3 = 0.8 >= 0.6 -> true
        let result = evaluator
            .aggregate_results(
                &[true, true, false],
                &AggregatorType::Weighted {
                    weights: weights.clone(),
                    threshold,
                },
            )
            .unwrap();
        assert!(result);

        // 0.5 = 0.5 < 0.6 -> false
        let result = evaluator
            .aggregate_results(
                &[true, false, false],
                &AggregatorType::Weighted { weights, threshold },
            )
            .unwrap();
        assert!(!result);
    }

    #[test]
    fn test_aggregator_weighted_mismatch() {
        let evaluator = Evaluator::new();

        let weights = vec![0.5, 0.3]; // Only 2 weights
        let threshold = 0.6;

        // Should fail because weights count doesn't match results count
        let result = evaluator.aggregate_results(
            &[true, true, false], // 3 results
            &AggregatorType::Weighted { weights, threshold },
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_aggregator_at_least_n() {
        let evaluator = Evaluator::new();

        // At least 2 must be true
        let result = evaluator
            .aggregate_results(&[true, true, false], &AggregatorType::AtLeastN { n: 2 })
            .unwrap();
        assert!(result);

        let result = evaluator
            .aggregate_results(&[true, false, false], &AggregatorType::AtLeastN { n: 2 })
            .unwrap();
        assert!(!result);

        // At least 3 must be true
        let result = evaluator
            .aggregate_results(&[true, true, true, false], &AggregatorType::AtLeastN { n: 3 })
            .unwrap();
        assert!(result);
    }

    #[test]
    fn test_aggregator_custom() {
        let evaluator = Evaluator::new();

        // Custom aggregator: count_true > 1
        let expr = Expression::binary(
            Operator::Gt,
            Expression::context_ref(vec!["count_true"]),
            Expression::number(1.0),
        );

        let result = evaluator
            .aggregate_results(
                &[true, true, false],
                &AggregatorType::Custom {
                    expression: Some(Box::new(expr.clone())),
                },
            )
            .unwrap();
        assert!(result); // count_true = 2 > 1

        let result = evaluator
            .aggregate_results(
                &[true, false, false],
                &AggregatorType::Custom {
                    expression: Some(Box::new(expr)),
                },
            )
            .unwrap();
        assert!(!result); // count_true = 1 > 1 is false
    }
}
