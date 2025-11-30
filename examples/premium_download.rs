/*!
# Premium Download Policy Example

This example demonstrates the TDLN Core implementation for the premium download policy
from the specification:

"Premium users can download files if they have available quota and the file is not restricted."

This showcases:
- Building semantic policies
- Creating policy compositions
- Evaluating against context
- Generating execution proofs
*/

use serde_json::json;
use tdln_core::*;

fn main() -> Result<()> {
    println!("=== TDLN Core Example: Premium Download Policy ===\n");

    // Step 1: Create individual policy bits
    println!("Step 1: Creating individual policy bits...");

    let is_premium = PolicyBit::builder()
        .name("is_premium_user")
        .description("Check if user has premium account type")
        .condition(Expression::binary(
            Operator::Eq,
            Expression::context_ref(vec!["user", "account_type"]),
            Expression::string("premium"),
        ))
        .fallback(false)
        .build()?;

    println!("  ✓ Created: {} (hash: {})", is_premium.name, is_premium.source_hash.short(12));

    let has_quota = PolicyBit::builder()
        .name("has_available_quota")
        .description("Check if user has positive download quota")
        .condition(Expression::binary(
            Operator::Gt,
            Expression::context_ref(vec!["user", "download_quota"]),
            Expression::number(0.0),
        ))
        .fallback(false)
        .build()?;

    println!("  ✓ Created: {} (hash: {})", has_quota.name, has_quota.source_hash.short(12));

    let file_not_restricted = PolicyBit::builder()
        .name("file_not_restricted")
        .description("Check if file is not marked as restricted")
        .condition(Expression::unary(
            Operator::Not,
            Expression::context_ref(vec!["file", "is_restricted"]),
        ))
        .fallback(false)
        .build()?;

    println!("  ✓ Created: {} (hash: {})\n", file_not_restricted.name, file_not_restricted.source_hash.short(12));

    // Step 2: Create policy composition
    println!("Step 2: Creating policy composition...");

    let composition = PolicyComposition::builder()
        .name("premium_download_decision")
        .description("Final decision for premium download access")
        .composition_type(CompositionType::Parallel)
        .policy(is_premium.id.clone())
        .policy(has_quota.id.clone())
        .policy(file_not_restricted.id.clone())
        .aggregator(AggregatorType::All)
        .build()?;

    println!("  ✓ Created composition: {} (hash: {})\n", composition.name, composition.source_hash.short(12));

    // Step 3: Create semantic unit
    println!("Step 3: Creating semantic unit...");

    let semantic_unit = SemanticUnit::builder()
        .name("premium_download_policy")
        .description("Premium users can download files if they have available quota and the file is not restricted")
        .policy_bit(is_premium)
        .policy_bit(has_quota)
        .policy_bit(file_not_restricted)
        .policy(PolicyNode::Composition(composition.clone()))
        .output(OutputDefinition::new(
            "allow_download",
            "Whether to allow file download",
            &composition.id,
        ))
        .build()?;

    println!("  ✓ Created semantic unit: {}", semantic_unit.name);
    println!("  ✓ Unit hash: {}", semantic_unit.source_hash);
    println!("  ✓ Contains {} policies\n", semantic_unit.policies.len());

    // Step 4: Evaluate against different contexts
    println!("Step 4: Evaluating against different contexts...\n");

    let evaluator = Evaluator::new();

    // Scenario 1: Premium user with quota and unrestricted file (should succeed)
    println!("Scenario 1: Premium user with quota and unrestricted file");
    let context1 = EvaluationContext::new(json!({
        "user": {
            "account_type": "premium",
            "download_quota": 100
        },
        "file": {
            "is_restricted": false
        }
    }));

    let (results1, proof1) = evaluator.evaluate_with_proof(&semantic_unit, &context1)?;
    println!("  Results: {:?}", results1);
    println!("  Overall: {}", if proof1.result { "✓ ALLOWED" } else { "✗ DENIED" });
    println!("  Execution trace:");
    for step in &proof1.execution_trace {
        println!("    - {}: {}", step.policy_name, if step.result { "✓" } else { "✗" });
    }
    println!();

    // Scenario 2: Non-premium user (should fail)
    println!("Scenario 2: Non-premium user");
    let context2 = EvaluationContext::new(json!({
        "user": {
            "account_type": "free",
            "download_quota": 100
        },
        "file": {
            "is_restricted": false
        }
    }));

    let (results2, proof2) = evaluator.evaluate_with_proof(&semantic_unit, &context2)?;
    println!("  Results: {:?}", results2);
    println!("  Overall: {}", if proof2.result { "✓ ALLOWED" } else { "✗ DENIED" });
    println!("  Execution trace:");
    for step in &proof2.execution_trace {
        println!("    - {}: {}", step.policy_name, if step.result { "✓" } else { "✗" });
    }
    println!();

    // Scenario 3: Premium user with no quota (should fail)
    println!("Scenario 3: Premium user with no quota");
    let context3 = EvaluationContext::new(json!({
        "user": {
            "account_type": "premium",
            "download_quota": 0
        },
        "file": {
            "is_restricted": false
        }
    }));

    let (results3, proof3) = evaluator.evaluate_with_proof(&semantic_unit, &context3)?;
    println!("  Results: {:?}", results3);
    println!("  Overall: {}", if proof3.result { "✓ ALLOWED" } else { "✗ DENIED" });
    println!("  Execution trace:");
    for step in &proof3.execution_trace {
        println!("    - {}: {}", step.policy_name, if step.result { "✓" } else { "✗" });
    }
    println!();

    // Step 5: Demonstrate canonicalization
    println!("Step 5: Demonstrating canonicalization...");

    let canonicalizer = Canonicalizer::default();
    let canonical_unit = canonicalizer.canonicalize_semantic_unit(&semantic_unit)?;

    println!("  Original hash:   {}", semantic_unit.source_hash);
    println!("  Canonical hash:  {}", canonical_unit.source_hash);
    println!("  Hashes match: {}\n", semantic_unit.source_hash == canonical_unit.source_hash);

    // Step 6: Serialize to JSON
    println!("Step 6: Serializing semantic unit to JSON...");

    let json_output = serde_json::to_string_pretty(&semantic_unit)?;
    println!("  JSON size: {} bytes", json_output.len());
    println!("  First 500 chars:\n{}\n", &json_output[..json_output.len().min(500)]);

    // Step 7: Demonstrate hash-based content addressing
    println!("Step 7: Hash-based content addressing...");
    println!("  Semantic Unit ID: {}", semantic_unit.id);
    println!("  Content Hash: {}", semantic_unit.source_hash);
    println!("  Version: {}", semantic_unit.version);
    println!("\n  This unit can be uniquely identified and retrieved by its hash!");
    println!("  Any change to the content will produce a different hash.\n");

    println!("=== Example completed successfully! ===");

    Ok(())
}
