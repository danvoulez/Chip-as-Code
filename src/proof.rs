use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::hash::TdlnHash;

/// A single step in the translation process
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranslationStep {
    pub sequence: u32,
    pub transformation: String,
    pub input_hash: TdlnHash,
    pub output_hash: TdlnHash,
    pub rule_applied: String,
}

impl TranslationStep {
    pub fn new<S: Into<String>>(
        sequence: u32,
        transformation: S,
        input_hash: TdlnHash,
        output_hash: TdlnHash,
        rule_applied: S,
    ) -> Self {
        Self {
            sequence,
            transformation: transformation.into(),
            input_hash,
            output_hash,
            rule_applied: rule_applied.into(),
        }
    }
}

/// Proof that a translation from source to TDLN Core was performed correctly
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranslationProof {
    pub proof_type: String,
    pub source_text: String,
    pub source_hash: TdlnHash,
    pub target_core_hash: TdlnHash,
    pub translation_steps: Vec<TranslationStep>,
    pub canonicalization_config: HashMap<String, serde_json::Value>,
    pub timestamp: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

impl TranslationProof {
    /// Creates a new translation proof
    pub fn new(source_text: String, source_hash: TdlnHash) -> Self {
        Self {
            proof_type: "translation".to_string(),
            source_text,
            source_hash,
            target_core_hash: TdlnHash::new(""),
            translation_steps: Vec::new(),
            canonicalization_config: HashMap::new(),
            timestamp: Utc::now(),
            signature: None,
        }
    }

    /// Adds a translation step
    pub fn add_step(&mut self, step: TranslationStep) {
        self.translation_steps.push(step);
    }

    /// Sets the target core hash
    pub fn set_target_hash(&mut self, hash: TdlnHash) {
        self.target_core_hash = hash;
    }

    /// Sets the canonicalization config
    pub fn set_config(&mut self, config: HashMap<String, serde_json::Value>) {
        self.canonicalization_config = config;
    }

    /// Signs the proof (placeholder for actual cryptographic signing)
    pub fn sign(&mut self, signature: String) {
        self.signature = Some(signature);
    }

    /// Verifies the proof integrity
    pub fn verify(&self) -> bool {
        // Check that steps are sequential
        for (i, step) in self.translation_steps.iter().enumerate() {
            if step.sequence as usize != i + 1 {
                return false;
            }
        }

        // Check that first step starts from source
        if let Some(first) = self.translation_steps.first() {
            if first.input_hash != self.source_hash {
                return false;
            }
        }

        // Check that last step ends at target
        if let Some(last) = self.translation_steps.last() {
            if last.output_hash != self.target_core_hash {
                return false;
            }
        }

        // Check that steps chain together
        for i in 0..self.translation_steps.len().saturating_sub(1) {
            if self.translation_steps[i].output_hash != self.translation_steps[i + 1].input_hash {
                return false;
            }
        }

        true
    }
}

/// Result of a single validation rule
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationResult {
    pub rule: String,
    pub passed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

impl ValidationResult {
    pub fn new<S: Into<String>>(rule: S, passed: bool) -> Self {
        Self {
            rule: rule.into(),
            passed,
            message: None,
        }
    }

    pub fn with_message<S: Into<String>>(mut self, message: S) -> Self {
        self.message = Some(message.into());
        self
    }
}

/// Proof that a TDLN node passed validation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationProof {
    pub proof_type: String,
    pub core_hash: TdlnHash,
    pub validation_rules: Vec<String>,
    pub results: Vec<ValidationResult>,
    pub timestamp: DateTime<Utc>,
}

impl ValidationProof {
    /// Creates a new validation proof
    pub fn new(core_hash: TdlnHash, validation_rules: Vec<String>) -> Self {
        Self {
            proof_type: "validation".to_string(),
            core_hash,
            validation_rules,
            results: Vec::new(),
            timestamp: Utc::now(),
        }
    }

    /// Adds a validation result
    pub fn add_result(&mut self, result: ValidationResult) {
        self.results.push(result);
    }

    /// Returns true if all validation rules passed
    pub fn is_valid(&self) -> bool {
        self.results.iter().all(|r| r.passed)
    }

    /// Returns failed validation rules
    pub fn failed_rules(&self) -> Vec<&ValidationResult> {
        self.results.iter().filter(|r| !r.passed).collect()
    }

    /// Returns a summary of the validation
    pub fn summary(&self) -> String {
        let total = self.results.len();
        let passed = self.results.iter().filter(|r| r.passed).count();
        let failed = total - passed;

        if failed == 0 {
            format!("All {} validation rules passed", total)
        } else {
            format!("{} of {} validation rules failed", failed, total)
        }
    }
}

/// Execution proof for policy evaluation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionProof {
    pub proof_type: String,
    pub policy_hash: TdlnHash,
    pub context_hash: TdlnHash,
    pub result: bool,
    pub execution_trace: Vec<ExecutionStep>,
    pub timestamp: DateTime<Utc>,
}

/// A single step in the execution trace
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionStep {
    pub policy_id: String,
    pub policy_name: String,
    pub result: bool,
    pub duration_micros: u64,
}

impl ExecutionProof {
    pub fn new(policy_hash: TdlnHash, context_hash: TdlnHash, result: bool) -> Self {
        Self {
            proof_type: "execution".to_string(),
            policy_hash,
            context_hash,
            result,
            execution_trace: Vec::new(),
            timestamp: Utc::now(),
        }
    }

    pub fn add_step(&mut self, step: ExecutionStep) {
        self.execution_trace.push(step);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translation_proof_creation() {
        let source = "Premium users can download files".to_string();
        let source_hash = TdlnHash::from_bytes(source.as_bytes());
        let proof = TranslationProof::new(source.clone(), source_hash.clone());

        assert_eq!(proof.source_text, source);
        assert_eq!(proof.source_hash, source_hash);
        assert_eq!(proof.proof_type, "translation");
    }

    #[test]
    fn test_translation_proof_steps() {
        let source = "test".to_string();
        let source_hash = TdlnHash::from_bytes(source.as_bytes());
        let mut proof = TranslationProof::new(source, source_hash.clone());

        let step1_output = TdlnHash::from_bytes(b"step1");
        let step2_output = TdlnHash::from_bytes(b"step2");

        proof.add_step(TranslationStep::new(
            1,
            "parse",
            source_hash.clone(),
            step1_output.clone(),
            "pattern_matching",
        ));

        proof.add_step(TranslationStep::new(
            2,
            "canonicalize",
            step1_output.clone(),
            step2_output.clone(),
            "canonicalization_rules",
        ));

        proof.set_target_hash(step2_output.clone());

        assert!(proof.verify());
    }

    #[test]
    fn test_validation_proof() {
        let hash = TdlnHash::from_bytes(b"test");
        let rules = vec![
            "structural_validity".to_string(),
            "hash_consistency".to_string(),
        ];

        let mut proof = ValidationProof::new(hash, rules);

        proof.add_result(
            ValidationResult::new("structural_validity", true)
                .with_message("Node structure is valid"),
        );

        proof.add_result(
            ValidationResult::new("hash_consistency", true)
                .with_message("Hash matches content"),
        );

        assert!(proof.is_valid());
        assert_eq!(proof.failed_rules().len(), 0);
    }

    #[test]
    fn test_validation_proof_failures() {
        let hash = TdlnHash::from_bytes(b"test");
        let rules = vec!["rule1".to_string(), "rule2".to_string()];

        let mut proof = ValidationProof::new(hash, rules);

        proof.add_result(ValidationResult::new("rule1", true));
        proof.add_result(ValidationResult::new("rule2", false).with_message("Failed"));

        assert!(!proof.is_valid());
        assert_eq!(proof.failed_rules().len(), 1);
    }

    #[test]
    fn test_execution_proof() {
        let policy_hash = TdlnHash::from_bytes(b"policy");
        let context_hash = TdlnHash::from_bytes(b"context");

        let proof = ExecutionProof::new(policy_hash, context_hash, true);

        assert_eq!(proof.proof_type, "execution");
        assert!(proof.result);
    }
}
