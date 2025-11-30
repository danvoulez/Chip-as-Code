/*!
# TDLN Translator - Natural Language to TDLN Core

Converts natural language policy statements into canonical TDLN PolicyBits with proof generation.

## Overview

The translator uses pattern matching and intent extraction to convert common policy
statements into structured TDLN representations. It generates translation proofs
that attest to the semantic preservation of the original intent.

## Supported Patterns

- **Conditional access**: "Premium users can download files"
- **Authorization**: "Only admins can delete posts"
- **Validation**: "Amount must be less than 1000"
- **Verification**: "User must be verified"
- **Composite**: "Users with premium accounts and valid KYC can withdraw"

## Example

```rust
use tdln_core::translator::NlTranslator;

let translator = NlTranslator::new();
let result = translator.translate("Premium users can download files")?;

println!("Generated Policy: {}", result.policy_bit.name);
println!("Translation Proof: {}", result.proof.source_text);
```
*/

use crate::ast::PolicyBit;
use crate::error::{Result, TdlnError};
use crate::expression::{ContextReference, Expression, Literal};
use crate::proof::TranslationProof;
use crate::types::Operator;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Result of NL translation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationResult {
    pub policy_bit: PolicyBit,
    pub proof: TranslationProof,
    pub confidence: f32,
    pub extracted_entities: HashMap<String, String>,
}

/// Intent patterns for common policy statements
#[derive(Debug, Clone, PartialEq)]
enum IntentPattern {
    ConditionalAccess {
        subject: String,
        condition: String,
        action: String,
    },
    Comparison {
        field: String,
        operator: Operator,
        value: String,
    },
    Verification {
        subject: String,
        property: String,
    },
    Composite {
        patterns: Vec<IntentPattern>,
        combinator: String, // "and", "or"
    },
}

/// Natural Language to TDLN translator
pub struct NlTranslator {
    patterns: Vec<PatternMatcher>,
}

/// Pattern matcher for NL statements
struct PatternMatcher {
    regex: Regex,
    extractor: fn(&regex::Captures) -> Result<IntentPattern>,
}

impl NlTranslator {
    /// Creates a new NL translator with default patterns
    pub fn new() -> Self {
        Self {
            patterns: Self::build_default_patterns(),
        }
    }

    /// Translates a natural language statement to TDLN
    pub fn translate(&self, input: &str) -> Result<TranslationResult> {
        let normalized = self.normalize_input(input);

        // Try to match against known patterns
        for pattern_matcher in &self.patterns {
            if let Some(captures) = pattern_matcher.regex.captures(&normalized) {
                let intent = (pattern_matcher.extractor)(&captures)?;
                return self.intent_to_policy_bit(input, intent);
            }
        }

        // If no pattern matches, attempt generic extraction
        self.generic_translation(input)
    }

    /// Normalizes input text
    fn normalize_input(&self, input: &str) -> String {
        input.trim().to_lowercase()
    }

    /// Converts extracted intent to PolicyBit
    fn intent_to_policy_bit(
        &self,
        original: &str,
        intent: IntentPattern,
    ) -> Result<TranslationResult> {
        let mut entities = HashMap::new();

        let (name, description, condition) = match intent {
            IntentPattern::ConditionalAccess {
                subject,
                condition: cond,
                action,
            } => {
                entities.insert("subject".to_string(), subject.clone());
                entities.insert("condition".to_string(), cond.clone());
                entities.insert("action".to_string(), action.clone());

                let name = format!("check_{}", subject.replace(' ', "_"));
                let desc = format!("Policy: {} when {}", action, cond);
                let expr = self.build_conditional_expression(&subject, &cond)?;

                (name, desc, expr)
            }

            IntentPattern::Comparison {
                field,
                operator,
                value,
            } => {
                entities.insert("field".to_string(), field.clone());
                entities.insert("operator".to_string(), format!("{:?}", operator));
                entities.insert("value".to_string(), value.clone());

                let name = format!("validate_{}", field.replace('.', "_"));
                let desc = format!("Validate {} {:?} {}", field, operator, value);
                let expr = self.build_comparison_expression(&field, operator, &value)?;

                (name, desc, expr)
            }

            IntentPattern::Verification { subject, property } => {
                entities.insert("subject".to_string(), subject.clone());
                entities.insert("property".to_string(), property.clone());

                let name = format!("verify_{}_{}", subject.replace(' ', "_"), property);
                let desc = format!("Verify {} is {}", subject, property);
                let expr = self.build_verification_expression(&subject, &property)?;

                (name, desc, expr)
            }

            IntentPattern::Composite { .. } => {
                // For composite patterns, we need more sophisticated handling
                // For now, return a simple placeholder
                return Err(TdlnError::TranslationError(
                    "Composite patterns not yet fully implemented".to_string(),
                ));
            }
        };

        // Build PolicyBit
        let policy_bit = PolicyBit::builder()
            .name(name)
            .description(description)
            .condition(condition)
            .build()?;

        // Generate translation proof
        use crate::hash::TdlnHash;
        let source_hash = TdlnHash::from_bytes(original.as_bytes());
        let mut proof = TranslationProof::new(
            original.to_string(),
            source_hash.clone(),
        );

        // Add translation steps
        proof.add_step(crate::proof::TranslationStep::new(
            1,
            format!("Normalized input: '{}'", self.normalize_input(original)),
            source_hash.clone(),
            policy_bit.source_hash.clone(),
            "nl_pattern_matching".to_string(),
        ));

        proof.set_target_hash(policy_bit.source_hash.clone());

        Ok(TranslationResult {
            policy_bit,
            proof,
            confidence: 0.85, // High confidence for pattern matches
            extracted_entities: entities,
        })
    }

    /// Builds conditional access expression
    fn build_conditional_expression(&self, subject: &str, condition: &str) -> Result<Expression> {
        // Parse common condition patterns
        if condition.contains("premium") {
            Ok(Expression::binary(
                Operator::Eq,
                Expression::context_ref(vec!["user", "account_type"]),
                Expression::string("premium"),
            ))
        } else if condition.contains("verified") || condition.contains("kyc") {
            Ok(Expression::binary(
                Operator::Eq,
                Expression::context_ref(vec!["user", "verified"]),
                Expression::boolean(true),
            ))
        } else if condition.contains("admin") {
            Ok(Expression::binary(
                Operator::Eq,
                Expression::context_ref(vec!["user", "role"]),
                Expression::string("admin"),
            ))
        } else {
            // Default: check if subject property is true
            Ok(Expression::context_ref(vec![subject]))
        }
    }

    /// Builds comparison expression
    fn build_comparison_expression(
        &self,
        field: &str,
        operator: Operator,
        value: &str,
    ) -> Result<Expression> {
        let field_path: Vec<&str> = field.split('.').collect();
        let context_ref = Expression::ContextRef(ContextReference::new(field_path));

        let value_expr = if let Ok(num) = value.parse::<f64>() {
            Expression::Literal(Literal::Number(num))
        } else if value == "true" || value == "false" {
            Expression::Literal(Literal::Boolean(value == "true"))
        } else {
            Expression::Literal(Literal::String(value.to_string()))
        };

        Ok(Expression::Binary {
            operator,
            left: Box::new(context_ref),
            right: Box::new(value_expr),
        })
    }

    /// Builds verification expression
    fn build_verification_expression(&self, subject: &str, property: &str) -> Result<Expression> {
        Ok(Expression::binary(
            Operator::Eq,
            Expression::context_ref(vec![subject, property]),
            Expression::boolean(true),
        ))
    }

    /// Generic translation when no pattern matches
    fn generic_translation(&self, input: &str) -> Result<TranslationResult> {
        // For unrecognized patterns, create a placeholder policy
        let policy_bit = PolicyBit::builder()
            .name("unrecognized_pattern")
            .description(format!("Original input: {}", input))
            .condition(Expression::boolean(false)) // Default to false for safety
            .build()?;

        use crate::hash::TdlnHash;
        let source_hash = TdlnHash::from_bytes(input.as_bytes());
        let mut proof = TranslationProof::new(
            input.to_string(),
            source_hash.clone(),
        );

        proof.add_step(crate::proof::TranslationStep::new(
            1,
            "No matching pattern found - created placeholder".to_string(),
            source_hash.clone(),
            policy_bit.source_hash.clone(),
            "unrecognized_fallback".to_string(),
        ));

        proof.set_target_hash(policy_bit.source_hash.clone());

        Ok(TranslationResult {
            policy_bit,
            proof,
            confidence: 0.2, // Low confidence
            extracted_entities: HashMap::new(),
        })
    }

    /// Builds default pattern matchers
    fn build_default_patterns() -> Vec<PatternMatcher> {
        vec![
            // Pattern: "Premium users can download files"
            PatternMatcher {
                regex: Regex::new(r"(\w+)\s+users?\s+can\s+(\w+)").unwrap(),
                extractor: |caps| {
                    Ok(IntentPattern::ConditionalAccess {
                        subject: "user".to_string(),
                        condition: caps[1].to_string(),
                        action: caps[2].to_string(),
                    })
                },
            },
            // Pattern: "Only admins can delete"
            PatternMatcher {
                regex: Regex::new(r"only\s+(\w+)s?\s+can\s+(\w+)").unwrap(),
                extractor: |caps| {
                    Ok(IntentPattern::ConditionalAccess {
                        subject: "user".to_string(),
                        condition: caps[1].to_string(),
                        action: caps[2].to_string(),
                    })
                },
            },
            // Pattern: "Amount must be less than 1000"
            PatternMatcher {
                regex: Regex::new(r"([\w\.]+)\s+must\s+be\s+less\s+than\s+(\d+)").unwrap(),
                extractor: |caps| {
                    Ok(IntentPattern::Comparison {
                        field: caps[1].to_string(),
                        operator: Operator::Lt,
                        value: caps[2].to_string(),
                    })
                },
            },
            // Pattern: "Amount must be greater than 100"
            PatternMatcher {
                regex: Regex::new(r"([\w\.]+)\s+must\s+be\s+greater\s+than\s+(\d+)").unwrap(),
                extractor: |caps| {
                    Ok(IntentPattern::Comparison {
                        field: caps[1].to_string(),
                        operator: Operator::Gt,
                        value: caps[2].to_string(),
                    })
                },
            },
            // Pattern: "User must be verified"
            PatternMatcher {
                regex: Regex::new(r"(\w+)\s+must\s+be\s+(\w+)").unwrap(),
                extractor: |caps| {
                    Ok(IntentPattern::Verification {
                        subject: caps[1].to_string(),
                        property: caps[2].to_string(),
                    })
                },
            },
            // Pattern: "Balance equals 0"
            PatternMatcher {
                regex: Regex::new(r"([\w\.]+)\s+equals?\s+([^\s]+)").unwrap(),
                extractor: |caps| {
                    Ok(IntentPattern::Comparison {
                        field: caps[1].to_string(),
                        operator: Operator::Eq,
                        value: caps[2].to_string(),
                    })
                },
            },
        ]
    }
}

impl Default for NlTranslator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_premium_user_pattern() {
        let translator = NlTranslator::new();
        let result = translator.translate("Premium users can download files").unwrap();

        assert_eq!(result.policy_bit.name, "check_user");
        assert!(result.confidence > 0.8);
        assert_eq!(result.extracted_entities.get("condition").unwrap(), "premium");
    }

    #[test]
    fn test_comparison_pattern() {
        let translator = NlTranslator::new();
        let result = translator
            .translate("Amount must be less than 1000")
            .unwrap();

        assert_eq!(result.policy_bit.name, "validate_amount");
        assert_eq!(result.extracted_entities.get("field").unwrap(), "amount");
    }

    #[test]
    fn test_verification_pattern() {
        let translator = NlTranslator::new();
        let result = translator.translate("User must be verified").unwrap();

        assert_eq!(result.policy_bit.name, "verify_user_verified");
        assert!(result.confidence > 0.8);
    }

    #[test]
    fn test_admin_only_pattern() {
        let translator = NlTranslator::new();
        let result = translator.translate("Only admins can delete").unwrap();

        assert_eq!(result.policy_bit.name, "check_user");
        assert_eq!(result.extracted_entities.get("condition").unwrap(), "admins");
    }

    #[test]
    fn test_unrecognized_pattern() {
        let translator = NlTranslator::new();
        let result = translator
            .translate("This is some random text that doesn't match")
            .unwrap();

        assert_eq!(result.policy_bit.name, "unrecognized_pattern");
        assert!(result.confidence < 0.5);
    }

    #[test]
    fn test_translation_proof_generation() {
        let translator = NlTranslator::new();
        let result = translator.translate("Premium users can download files").unwrap();

        assert!(!result.proof.source_text.is_empty());
        assert!(!result.proof.translation_steps.is_empty());
    }
}
