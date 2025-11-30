/*!
# TDLN DSL Parser

A parser for the TDLN Domain-Specific Language that allows defining policies
in a clean, structured text format.

## DSL Syntax

### Policy Bit Definition

```tdln
policy MyPolicy {
    description: "Check if user is premium"
    condition: user.account_type == "premium"
}
```

### Operators

- Comparison: `==`, `!=`, `<`, `>`, `<=`, `>=`
- Logical: `&&`, `||`, `!`
- Existence: `exists`

### Expressions

- Context references: `user.account_type`, `transaction.amount`
- Literals: `"string"`, `123`, `45.67`, `true`, `false`, `null`
- Function calls: `length(user.name)`, `contains(roles, "admin")`

### Policy Composition

```tdln
composition AllChecks {
    type: parallel
    aggregator: all
    policies: [CheckAuth, CheckAmount, CheckFraud]
}
```

## Example

```rust
use tdln_core::dsl::DslParser;

let dsl = r#"
policy PremiumAccess {
    description: "Premium users can access"
    condition: user.account_type == "premium"
}
"#;

let parser = DslParser::new();
let policy = parser.parse_policy(dsl)?;
```
*/

use crate::ast::{PolicyBit, PolicyComposition, Parameter};
use crate::error::{Result, TdlnError};
use crate::expression::{ContextReference, Expression, Literal};
use crate::types::{AggregatorType, CompositionType, Operator, ValueType};
use regex::Regex;
use std::collections::HashMap;

/// DSL parser for TDLN policies
pub struct DslParser {
    // Token patterns
    identifier_pattern: Regex,
    string_pattern: Regex,
    number_pattern: Regex,
}

impl DslParser {
    /// Creates a new DSL parser
    pub fn new() -> Self {
        Self {
            identifier_pattern: Regex::new(r"^[a-zA-Z_][a-zA-Z0-9_]*$").unwrap(),
            string_pattern: Regex::new(r#"^"([^"]*)"$"#).unwrap(),
            number_pattern: Regex::new(r"^-?\d+(\.\d+)?$").unwrap(),
        }
    }

    /// Parses a policy definition from DSL text
    pub fn parse_policy(&self, input: &str) -> Result<PolicyBit> {
        let lines = self.preprocess(input);

        // Extract policy block
        if !lines.iter().any(|l| l.trim().starts_with("policy ")) {
            return Err(TdlnError::TranslationError("No policy definition found".to_string()));
        }

        let mut name = String::new();
        let mut description = String::new();
        let mut condition_str = String::new();
        let mut parameters = Vec::new();

        let mut in_policy_block = false;
        let mut brace_count = 0;

        for line in lines {
            let trimmed = line.trim();

            if trimmed.starts_with("policy ") {
                // Extract policy name
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() >= 2 {
                    name = parts[1].trim_end_matches('{').trim().to_string();
                }
                if trimmed.ends_with('{') {
                    in_policy_block = true;
                    brace_count = 1;
                }
                continue;
            }

            if trimmed == "{" {
                brace_count += 1;
                in_policy_block = true;
                continue;
            }

            if trimmed == "}" {
                brace_count -= 1;
                if brace_count == 0 {
                    in_policy_block = false;
                }
                continue;
            }

            if !in_policy_block {
                continue;
            }

            // Parse fields
            if let Some((key, value)) = self.parse_field(trimmed) {
                match key.as_str() {
                    "description" => {
                        description = self.extract_string_value(&value)?;
                    }
                    "condition" => {
                        condition_str = value;
                    }
                    "param" => {
                        if let Some(param) = self.parse_parameter(&value)? {
                            parameters.push(param);
                        }
                    }
                    _ => {} // Ignore unknown fields
                }
            }
        }

        if name.is_empty() {
            return Err(TdlnError::TranslationError("Policy name is required".to_string()));
        }

        // Parse condition expression
        let condition = if !condition_str.is_empty() {
            Some(self.parse_expression(&condition_str)?)
        } else {
            None
        };

        // Build policy
        PolicyBit::builder()
            .name(name)
            .description(description)
            .parameters(parameters)
            .condition(condition.unwrap_or(Expression::boolean(true)))
            .build()
    }

    /// Parses a policy composition from DSL text
    pub fn parse_composition(&self, input: &str) -> Result<PolicyComposition> {
        let lines = self.preprocess(input);

        let mut name = String::new();
        let mut description = String::new();
        let mut comp_type = CompositionType::Parallel;
        let mut aggregator = AggregatorType::All;
        let mut policies = Vec::new();

        let mut in_block = false;

        for line in lines {
            let trimmed = line.trim();

            if trimmed.starts_with("composition ") {
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() >= 2 {
                    name = parts[1].trim_end_matches('{').trim().to_string();
                }
                if trimmed.ends_with('{') {
                    in_block = true;
                }
                continue;
            }

            if trimmed == "{" {
                in_block = true;
                continue;
            }

            if trimmed == "}" {
                in_block = false;
                continue;
            }

            if !in_block {
                continue;
            }

            if let Some((key, value)) = self.parse_field(trimmed) {
                match key.as_str() {
                    "description" => {
                        description = self.extract_string_value(&value)?;
                    }
                    "type" => {
                        comp_type = match value.trim() {
                            "sequential" | "serial" => CompositionType::Sequential,
                            "parallel" => CompositionType::Parallel,
                            "conditional" => CompositionType::Conditional,
                            _ => return Err(TdlnError::TranslationError(format!("Unknown composition type: {}", value))),
                        };
                    }
                    "aggregator" => {
                        aggregator = match value.trim() {
                            "all" => AggregatorType::All,
                            "any" => AggregatorType::Any,
                            "majority" => AggregatorType::Majority,
                            _ => return Err(TdlnError::TranslationError(format!("Unknown aggregator: {}", value))),
                        };
                    }
                    "policies" => {
                        policies = self.parse_array(&value)?;
                    }
                    _ => {}
                }
            }
        }

        PolicyComposition::builder()
            .name(name)
            .description(description)
            .composition_type(comp_type)
            .policies(policies)
            .aggregator(aggregator)
            .build()
    }

    /// Preprocesses input by removing comments and empty lines
    fn preprocess(&self, input: &str) -> Vec<String> {
        input
            .lines()
            .map(|line| {
                // Remove comments
                if let Some(pos) = line.find("//") {
                    &line[..pos]
                } else {
                    line
                }
            })
            .filter(|line| !line.trim().is_empty())
            .map(|s| s.to_string())
            .collect()
    }

    /// Parses a field in the form "key: value"
    fn parse_field(&self, line: &str) -> Option<(String, String)> {
        if let Some(pos) = line.find(':') {
            let key = line[..pos].trim().to_string();
            let value = line[pos + 1..].trim().to_string();
            Some((key, value))
        } else {
            None
        }
    }

    /// Extracts string value from quoted string
    fn extract_string_value(&self, value: &str) -> Result<String> {
        let trimmed = value.trim();
        if let Some(captures) = self.string_pattern.captures(trimmed) {
            Ok(captures[1].to_string())
        } else {
            Err(TdlnError::TranslationError(format!("Invalid string value: {}", value)))
        }
    }

    /// Parses an expression string
    fn parse_expression(&self, expr: &str) -> Result<Expression> {
        let expr = expr.trim();

        // Handle boolean literals
        if expr == "true" {
            return Ok(Expression::boolean(true));
        }
        if expr == "false" {
            return Ok(Expression::boolean(false));
        }
        if expr == "null" {
            return Ok(Expression::null());
        }

        // Handle string literals
        if let Some(captures) = self.string_pattern.captures(expr) {
            return Ok(Expression::string(captures[1].to_string()));
        }

        // Handle number literals
        if let Some(_) = self.number_pattern.captures(expr) {
            if let Ok(num) = expr.parse::<f64>() {
                return Ok(Expression::number(num));
            }
        }

        // Handle logical OR (||)
        if let Some(pos) = expr.rfind("||") {
            let left = self.parse_expression(&expr[..pos])?;
            let right = self.parse_expression(&expr[pos + 2..])?;
            return Ok(Expression::binary(Operator::Or, left, right));
        }

        // Handle logical AND (&&)
        if let Some(pos) = expr.rfind("&&") {
            let left = self.parse_expression(&expr[..pos])?;
            let right = self.parse_expression(&expr[pos + 2..])?;
            return Ok(Expression::binary(Operator::And, left, right));
        }

        // Handle comparison operators
        for (op_str, op) in &[
            ("==", Operator::Eq),
            ("!=", Operator::Neq),
            ("<=", Operator::Lte),
            (">=", Operator::Gte),
            ("<", Operator::Lt),
            (">", Operator::Gt),
        ] {
            if let Some(pos) = expr.find(op_str) {
                let left = self.parse_expression(&expr[..pos])?;
                let right = self.parse_expression(&expr[pos + op_str.len()..])?;
                return Ok(Expression::binary(*op, left, right));
            }
        }

        // Handle negation (!)
        if expr.starts_with('!') {
            let arg = self.parse_expression(&expr[1..])?;
            return Ok(Expression::unary(Operator::Not, arg));
        }

        // Handle function calls
        if let Some(paren_pos) = expr.find('(') {
            if expr.ends_with(')') {
                let func_name = expr[..paren_pos].trim();
                let args_str = &expr[paren_pos + 1..expr.len() - 1];
                let args = self.parse_function_args(args_str)?;
                return Ok(Expression::function_call(func_name, args));
            }
        }

        // Handle context references (e.g., user.account_type)
        if expr.contains('.') {
            let path: Vec<String> = expr.split('.').map(|s| s.trim().to_string()).collect();
            return Ok(Expression::context_ref(path));
        }

        // Simple identifier is a context reference
        if self.identifier_pattern.is_match(expr) {
            return Ok(Expression::context_ref(vec![expr.to_string()]));
        }

        Err(TdlnError::TranslationError(format!("Cannot parse expression: {}", expr)))
    }

    /// Parses function arguments
    fn parse_function_args(&self, args_str: &str) -> Result<Vec<Expression>> {
        if args_str.trim().is_empty() {
            return Ok(Vec::new());
        }

        let args: Vec<&str> = args_str.split(',').collect();
        let mut parsed_args = Vec::new();

        for arg in args {
            parsed_args.push(self.parse_expression(arg.trim())?);
        }

        Ok(parsed_args)
    }

    /// Parses parameter definition
    fn parse_parameter(&self, param_str: &str) -> Result<Option<Parameter>> {
        // Format: "name:type" or "name:type:default"
        let parts: Vec<&str> = param_str.split(':').collect();

        if parts.len() < 2 {
            return Ok(None);
        }

        let name = parts[0].trim().to_string();
        let type_str = parts[1].trim();

        let value_type = match type_str {
            "string" => ValueType::String,
            "number" => ValueType::Number,
            "boolean" => ValueType::Boolean,
            "context" => ValueType::Context,
            "any" => ValueType::Any,
            _ => return Err(TdlnError::TranslationError(format!("Unknown type: {}", type_str))),
        };

        Ok(Some(Parameter::new(name, value_type)))
    }

    /// Parses an array literal
    fn parse_array(&self, array_str: &str) -> Result<Vec<String>> {
        let trimmed = array_str.trim();

        if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
            return Err(TdlnError::TranslationError("Array must be enclosed in brackets".to_string()));
        }

        let inner = &trimmed[1..trimmed.len() - 1];

        if inner.trim().is_empty() {
            return Ok(Vec::new());
        }

        Ok(inner
            .split(',')
            .map(|s| s.trim().to_string())
            .collect())
    }
}

impl Default for DslParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_policy() {
        let dsl = r#"
            policy TestPolicy {
                description: "A test policy"
                condition: user.premium == true
            }
        "#;

        let parser = DslParser::new();
        let policy = parser.parse_policy(dsl).unwrap();

        assert_eq!(policy.name, "TestPolicy");
        assert_eq!(policy.description, "A test policy");
    }

    #[test]
    fn test_parse_policy_with_string_comparison() {
        let dsl = r#"
            policy PremiumCheck {
                description: "Check premium status"
                condition: user.account_type == "premium"
            }
        "#;

        let parser = DslParser::new();
        let policy = parser.parse_policy(dsl).unwrap();

        assert_eq!(policy.name, "PremiumCheck");
    }

    #[test]
    fn test_parse_policy_with_logical_operators() {
        let dsl = r#"
            policy ComplexCheck {
                description: "Complex condition"
                condition: user.premium == true && user.verified == true
            }
        "#;

        let parser = DslParser::new();
        let policy = parser.parse_policy(dsl).unwrap();

        assert_eq!(policy.name, "ComplexCheck");
    }

    #[test]
    fn test_parse_policy_with_numeric_comparison() {
        let dsl = r#"
            policy AmountCheck {
                description: "Validate amount"
                condition: transaction.amount < 1000
            }
        "#;

        let parser = DslParser::new();
        let policy = parser.parse_policy(dsl).unwrap();

        assert_eq!(policy.name, "AmountCheck");
    }

    #[test]
    fn test_parse_composition() {
        let dsl = r#"
            composition AllChecks {
                description: "All security checks"
                type: parallel
                aggregator: all
                policies: [AuthCheck, AmountCheck, FraudCheck]
            }
        "#;

        let parser = DslParser::new();
        let composition = parser.parse_composition(dsl).unwrap();

        assert_eq!(composition.name, "AllChecks");
        assert_eq!(composition.policies.len(), 3);
        assert!(matches!(composition.composition_type, CompositionType::Parallel));
        assert_eq!(composition.aggregator, Some(AggregatorType::All));
    }

    #[test]
    fn test_parse_with_comments() {
        let dsl = r#"
            // This is a comment
            policy TestPolicy {
                description: "Test" // inline comment
                condition: x == 1
            }
        "#;

        let parser = DslParser::new();
        let policy = parser.parse_policy(dsl).unwrap();

        assert_eq!(policy.name, "TestPolicy");
    }

    #[test]
    fn test_parse_negation() {
        let dsl = r#"
            policy NotVerified {
                description: "Check not verified"
                condition: !user.verified
            }
        "#;

        let parser = DslParser::new();
        let policy = parser.parse_policy(dsl).unwrap();

        assert_eq!(policy.name, "NotVerified");
    }
}
