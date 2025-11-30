use serde::{Deserialize, Serialize};
use uuid::Uuid;
use crate::error::{Result, TdlnError};
use crate::expression::Expression;
use crate::hash::{Hashable, TdlnHash};
use crate::types::{AggregatorType, CompositionType, NodeType, ValueType};

/// Parameter definition for policies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Parameter {
    pub name: String,
    #[serde(rename = "type")]
    pub value_type: ValueType,
    pub required: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<serde_json::Value>,
}

impl Parameter {
    /// Creates a new required parameter
    pub fn new<S: Into<String>>(name: S, value_type: ValueType) -> Self {
        Self {
            name: name.into(),
            value_type,
            required: true,
            default: None,
        }
    }

    /// Makes the parameter optional with a default value
    pub fn with_default(mut self, default: serde_json::Value) -> Self {
        self.required = false;
        self.default = Some(default);
        self
    }
}

/// Output definition for semantic units
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputDefinition {
    pub name: String,
    pub description: String,
    pub source_policy: String, // Reference to policy ID
}

impl OutputDefinition {
    pub fn new<S: Into<String>>(name: S, description: S, source_policy: S) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            source_policy: source_policy.into(),
        }
    }
}

/// Base trait for all TDLN nodes
pub trait TdlnNode: Hashable {
    fn node_type(&self) -> NodeType;
    fn id(&self) -> &str;
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn version(&self) -> &str;
    fn source_hash(&self) -> &TdlnHash;
}

/// Atomic policy decision unit
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyBit {
    pub node_type: NodeType,
    pub id: String,
    pub name: String,
    pub description: String,
    pub parameters: Vec<Parameter>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condition: Option<Expression>,
    #[serde(default)]
    pub fallback: bool,
    pub source_hash: TdlnHash,
    pub version: String,
}

impl PolicyBit {
    /// Creates a builder for PolicyBit
    pub fn builder() -> PolicyBitBuilder {
        PolicyBitBuilder::default()
    }

    /// Converts to canonical dictionary for hashing
    pub fn to_canonical_dict(&self) -> serde_json::Value {
        let mut params = self.parameters.clone();
        params.sort_by(|a, b| a.name.cmp(&b.name));

        serde_json::json!({
            "node_type": self.node_type,
            "id": self.id,
            "name": self.name.trim(),
            "description": self.description.trim(),
            "parameters": params,
            "condition": self.condition,
            "fallback": self.fallback,
            "version": self.version,
        })
    }
}

impl TdlnNode for PolicyBit {
    fn node_type(&self) -> NodeType {
        self.node_type
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn version(&self) -> &str {
        &self.version
    }

    fn source_hash(&self) -> &TdlnHash {
        &self.source_hash
    }
}

impl Hashable for PolicyBit {
    fn compute_hash(&self) -> std::result::Result<TdlnHash, serde_json::Error> {
        TdlnHash::from_canonical_json(&self.to_canonical_dict())
    }
}

/// Builder for PolicyBit
#[derive(Default)]
pub struct PolicyBitBuilder {
    id: Option<String>,
    name: Option<String>,
    description: Option<String>,
    parameters: Vec<Parameter>,
    condition: Option<Expression>,
    fallback: bool,
    version: String,
}

impl PolicyBitBuilder {
    pub fn id<S: Into<String>>(mut self, id: S) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn name<S: Into<String>>(mut self, name: S) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn description<S: Into<String>>(mut self, description: S) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn parameter(mut self, param: Parameter) -> Self {
        self.parameters.push(param);
        self
    }

    pub fn parameters(mut self, params: Vec<Parameter>) -> Self {
        self.parameters = params;
        self
    }

    pub fn condition(mut self, condition: Expression) -> Self {
        self.condition = Some(condition);
        self
    }

    pub fn fallback(mut self, fallback: bool) -> Self {
        self.fallback = fallback;
        self
    }

    pub fn version<S: Into<String>>(mut self, version: S) -> Self {
        self.version = version.into();
        self
    }

    pub fn build(self) -> Result<PolicyBit> {
        let id = self.id.unwrap_or_else(|| Uuid::new_v4().to_string());
        let name = self.name.ok_or_else(|| TdlnError::BuilderError("name is required".to_string()))?;
        let description = self.description.unwrap_or_default();
        let version = if self.version.is_empty() {
            "1.0.0-policybit".to_string()
        } else {
            self.version
        };

        let mut policy = PolicyBit {
            node_type: NodeType::PolicyBit,
            id,
            name,
            description,
            parameters: self.parameters,
            condition: self.condition,
            fallback: self.fallback,
            source_hash: TdlnHash::new(""),
            version,
        };

        // Compute hash
        policy.source_hash = policy.compute_hash()
            .map_err(|e| TdlnError::Generic(e.to_string()))?;

        Ok(policy)
    }
}

/// Composition of multiple policies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyComposition {
    pub node_type: NodeType,
    pub id: String,
    pub name: String,
    pub description: String,
    pub composition_type: CompositionType,
    pub policies: Vec<String>, // List of policy IDs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aggregator: Option<AggregatorType>,
    pub source_hash: TdlnHash,
    pub version: String,
}

impl PolicyComposition {
    /// Creates a builder for PolicyComposition
    pub fn builder() -> PolicyCompositionBuilder {
        PolicyCompositionBuilder::default()
    }

    /// Converts to canonical dictionary for hashing
    pub fn to_canonical_dict(&self) -> serde_json::Value {
        let mut policies = self.policies.clone();
        policies.sort();

        serde_json::json!({
            "node_type": self.node_type,
            "id": self.id,
            "name": self.name.trim(),
            "description": self.description.trim(),
            "composition_type": self.composition_type,
            "policies": policies,
            "aggregator": self.aggregator,
            "version": self.version,
        })
    }
}

impl TdlnNode for PolicyComposition {
    fn node_type(&self) -> NodeType {
        self.node_type
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn version(&self) -> &str {
        &self.version
    }

    fn source_hash(&self) -> &TdlnHash {
        &self.source_hash
    }
}

impl Hashable for PolicyComposition {
    fn compute_hash(&self) -> std::result::Result<TdlnHash, serde_json::Error> {
        TdlnHash::from_canonical_json(&self.to_canonical_dict())
    }
}

/// Builder for PolicyComposition
#[derive(Default)]
pub struct PolicyCompositionBuilder {
    id: Option<String>,
    name: Option<String>,
    description: Option<String>,
    composition_type: Option<CompositionType>,
    policies: Vec<String>,
    aggregator: Option<AggregatorType>,
    version: String,
}

impl PolicyCompositionBuilder {
    pub fn id<S: Into<String>>(mut self, id: S) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn name<S: Into<String>>(mut self, name: S) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn description<S: Into<String>>(mut self, description: S) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn composition_type(mut self, composition_type: CompositionType) -> Self {
        self.composition_type = Some(composition_type);
        self
    }

    pub fn policy<S: Into<String>>(mut self, policy_id: S) -> Self {
        self.policies.push(policy_id.into());
        self
    }

    pub fn policies(mut self, policies: Vec<String>) -> Self {
        self.policies = policies;
        self
    }

    pub fn aggregator(mut self, aggregator: AggregatorType) -> Self {
        self.aggregator = Some(aggregator);
        self
    }

    pub fn version<S: Into<String>>(mut self, version: S) -> Self {
        self.version = version.into();
        self
    }

    pub fn build(self) -> Result<PolicyComposition> {
        let id = self.id.unwrap_or_else(|| Uuid::new_v4().to_string());
        let name = self.name.ok_or_else(|| TdlnError::BuilderError("name is required".to_string()))?;
        let description = self.description.unwrap_or_default();
        let composition_type = self.composition_type
            .ok_or_else(|| TdlnError::BuilderError("composition_type is required".to_string()))?;
        let version = if self.version.is_empty() {
            "1.0.0-composition".to_string()
        } else {
            self.version
        };

        let mut composition = PolicyComposition {
            node_type: NodeType::PolicyComposition,
            id,
            name,
            description,
            composition_type,
            policies: self.policies,
            aggregator: self.aggregator,
            source_hash: TdlnHash::new(""),
            version,
        };

        // Compute hash
        composition.source_hash = composition.compute_hash()
            .map_err(|e| TdlnError::Generic(e.to_string()))?;

        Ok(composition)
    }
}

/// Complete semantic unit containing policies and their composition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SemanticUnit {
    pub node_type: NodeType,
    pub id: String,
    pub name: String,
    pub description: String,
    pub policies: Vec<PolicyNode>,
    pub inputs: Vec<Parameter>,
    pub outputs: Vec<OutputDefinition>,
    pub source_hash: TdlnHash,
    pub version: String,
}

/// Union type for policy nodes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PolicyNode {
    Bit(PolicyBit),
    Composition(PolicyComposition),
}

impl SemanticUnit {
    /// Creates a builder for SemanticUnit
    pub fn builder() -> SemanticUnitBuilder {
        SemanticUnitBuilder::default()
    }

    /// Converts to canonical dictionary for hashing
    pub fn to_canonical_dict(&self) -> serde_json::Value {
        let mut inputs = self.inputs.clone();
        inputs.sort_by(|a, b| a.name.cmp(&b.name));

        let mut outputs = self.outputs.clone();
        outputs.sort_by(|a, b| a.name.cmp(&b.name));

        serde_json::json!({
            "node_type": self.node_type,
            "id": self.id,
            "name": self.name.trim(),
            "description": self.description.trim(),
            "policies": self.policies,
            "inputs": inputs,
            "outputs": outputs,
            "version": self.version,
        })
    }
}

impl TdlnNode for SemanticUnit {
    fn node_type(&self) -> NodeType {
        self.node_type
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn version(&self) -> &str {
        &self.version
    }

    fn source_hash(&self) -> &TdlnHash {
        &self.source_hash
    }
}

impl Hashable for SemanticUnit {
    fn compute_hash(&self) -> std::result::Result<TdlnHash, serde_json::Error> {
        TdlnHash::from_canonical_json(&self.to_canonical_dict())
    }
}

/// Builder for SemanticUnit
#[derive(Default)]
pub struct SemanticUnitBuilder {
    id: Option<String>,
    name: Option<String>,
    description: Option<String>,
    policies: Vec<PolicyNode>,
    inputs: Vec<Parameter>,
    outputs: Vec<OutputDefinition>,
    version: String,
}

impl SemanticUnitBuilder {
    pub fn id<S: Into<String>>(mut self, id: S) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn name<S: Into<String>>(mut self, name: S) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn description<S: Into<String>>(mut self, description: S) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn policy(mut self, policy: PolicyNode) -> Self {
        self.policies.push(policy);
        self
    }

    pub fn policy_bit(mut self, policy: PolicyBit) -> Self {
        self.policies.push(PolicyNode::Bit(policy));
        self
    }

    pub fn policy_composition(mut self, composition: PolicyComposition) -> Self {
        self.policies.push(PolicyNode::Composition(composition));
        self
    }

    pub fn policies(mut self, policies: Vec<PolicyNode>) -> Self {
        self.policies = policies;
        self
    }

    pub fn input(mut self, input: Parameter) -> Self {
        self.inputs.push(input);
        self
    }

    pub fn inputs(mut self, inputs: Vec<Parameter>) -> Self {
        self.inputs = inputs;
        self
    }

    pub fn output(mut self, output: OutputDefinition) -> Self {
        self.outputs.push(output);
        self
    }

    pub fn outputs(mut self, outputs: Vec<OutputDefinition>) -> Self {
        self.outputs = outputs;
        self
    }

    pub fn version<S: Into<String>>(mut self, version: S) -> Self {
        self.version = version.into();
        self
    }

    pub fn build(self) -> Result<SemanticUnit> {
        let id = self.id.unwrap_or_else(|| Uuid::new_v4().to_string());
        let name = self.name.ok_or_else(|| TdlnError::BuilderError("name is required".to_string()))?;
        let description = self.description.unwrap_or_default();
        let version = if self.version.is_empty() {
            "1.0.0-core".to_string()
        } else {
            self.version
        };

        let mut unit = SemanticUnit {
            node_type: NodeType::SemanticUnit,
            id,
            name,
            description,
            policies: self.policies,
            inputs: self.inputs,
            outputs: self.outputs,
            source_hash: TdlnHash::new(""),
            version,
        };

        // Compute hash
        unit.source_hash = unit.compute_hash()
            .map_err(|e| TdlnError::Generic(e.to_string()))?;

        Ok(unit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression::{Expression, Literal};
    use crate::types::Operator;

    #[test]
    fn test_policy_bit_builder() {
        let policy = PolicyBit::builder()
            .name("test_policy")
            .description("A test policy")
            .condition(Expression::boolean(true))
            .build()
            .unwrap();

        assert_eq!(policy.name, "test_policy");
        assert_eq!(policy.description, "A test policy");
        assert!(!policy.source_hash.as_str().is_empty());
    }

    #[test]
    fn test_policy_composition_builder() {
        let composition = PolicyComposition::builder()
            .name("test_composition")
            .composition_type(CompositionType::Parallel)
            .policy("policy1")
            .policy("policy2")
            .aggregator(AggregatorType::All)
            .build()
            .unwrap();

        assert_eq!(composition.policies.len(), 2);
        assert_eq!(composition.composition_type, CompositionType::Parallel);
    }

    #[test]
    fn test_semantic_unit_builder() {
        let policy = PolicyBit::builder()
            .name("test")
            .build()
            .unwrap();

        let unit = SemanticUnit::builder()
            .name("test_unit")
            .policy_bit(policy)
            .build()
            .unwrap();

        assert_eq!(unit.name, "test_unit");
        assert_eq!(unit.policies.len(), 1);
    }

    #[test]
    fn test_hash_determinism() {
        let policy1 = PolicyBit::builder()
            .id("same-id")
            .name("test")
            .description("desc")
            .build()
            .unwrap();

        let policy2 = PolicyBit::builder()
            .id("same-id")
            .name("test")
            .description("desc")
            .build()
            .unwrap();

        // Hashes should be identical for identical policies
        assert_eq!(policy1.source_hash, policy2.source_hash);
    }
}
