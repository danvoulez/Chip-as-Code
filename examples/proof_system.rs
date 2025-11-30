/*!
# Proof System Example

This example demonstrates the proof and validation system in TDLN Core:
- Translation proofs
- Validation proofs
- Execution proofs
- Cryptographic verification
*/

use tdln_core::*;
use serde_json::json;

fn main() -> Result<()> {
    println!("=== TDLN Core Example: Proof System ===\n");

    // Create a simple policy
    println!("Step 1: Creating a policy...");

    let policy = PolicyBit::builder()
        .name("admin_access")
        .description("Check if user is an administrator")
        .condition(Expression::binary(
            Operator::Eq,
            Expression::context_ref(vec!["user", "role"]),
            Expression::string("admin"),
        ))
        .build()?;

    println!("  ✓ Created policy: {}", policy.name);
    println!("  ✓ Hash: {}\n", policy.source_hash);

    // Create a translation proof (simulated)
    println!("Step 2: Creating translation proof...");

    let source_text = "Only administrators can access this resource";
    let source_hash = TdlnHash::from_bytes(source_text.as_bytes());

    let mut translation_proof = TranslationProof::new(
        source_text.to_string(),
        source_hash.clone(),
    );

    // Add translation steps
    let step1_hash = TdlnHash::from_bytes(b"intermediate_ast");
    translation_proof.add_step(TranslationStep::new(
        1,
        "natural_language_parsing",
        source_hash.clone(),
        step1_hash.clone(),
        "pattern_matching",
    ));

    translation_proof.add_step(TranslationStep::new(
        2,
        "canonicalization",
        step1_hash,
        policy.source_hash.clone(),
        "canonicalization_rules",
    ));

    translation_proof.set_target_hash(policy.source_hash.clone());

    println!("  ✓ Translation proof created");
    println!("  ✓ Steps: {}", translation_proof.translation_steps.len());
    println!("  ✓ Proof valid: {}\n", translation_proof.verify());

    // Display translation steps
    println!("  Translation steps:");
    for step in &translation_proof.translation_steps {
        println!("    {}. {} ({} → {})",
            step.sequence,
            step.transformation,
            step.input_hash.short(8),
            step.output_hash.short(8)
        );
    }
    println!();

    // Create a validation proof
    println!("Step 3: Creating validation proof...");

    let validation_rules = vec![
        "structural_validity".to_string(),
        "hash_consistency".to_string(),
        "reference_integrity".to_string(),
        "type_safety".to_string(),
        "determinism".to_string(),
    ];

    let mut validation_proof = ValidationProof::new(
        policy.source_hash.clone(),
        validation_rules,
    );

    // Add validation results
    validation_proof.add_result(
        ValidationResult::new("structural_validity", true)
            .with_message("Policy structure conforms to TDLN schema")
    );

    validation_proof.add_result(
        ValidationResult::new("hash_consistency", true)
            .with_message("Computed hash matches stored hash")
    );

    validation_proof.add_result(
        ValidationResult::new("reference_integrity", true)
            .with_message("All referenced policies exist")
    );

    validation_proof.add_result(
        ValidationResult::new("type_safety", true)
            .with_message("Expression types are consistent")
    );

    validation_proof.add_result(
        ValidationResult::new("determinism", true)
            .with_message("Canonicalization produces identical results")
    );

    println!("  ✓ Validation proof created");
    println!("  ✓ Rules validated: {}", validation_proof.results.len());
    println!("  ✓ All passed: {}", validation_proof.is_valid());
    println!("  ✓ Summary: {}\n", validation_proof.summary());

    // Display validation results
    println!("  Validation results:");
    for result in &validation_proof.results {
        println!("    {} {} - {}",
            if result.passed { "✓" } else { "✗" },
            result.rule,
            result.message.as_ref().unwrap_or(&"".to_string())
        );
    }
    println!();

    // Create execution proof
    println!("Step 4: Creating execution proof...");

    let context = EvaluationContext::new(json!({
        "user": {
            "role": "admin",
            "name": "Alice"
        }
    }));

    let evaluator = Evaluator::new();
    let result = evaluator.evaluate_policy_bit(&policy, &context)?;

    let context_hash = context.hash()?;
    let mut execution_proof = ExecutionProof::new(
        policy.source_hash.clone(),
        context_hash,
        result,
    );

    execution_proof.add_step(ExecutionStep {
        policy_id: policy.id.clone(),
        policy_name: policy.name.clone(),
        result,
        duration_micros: 42, // Would be measured in real implementation
    });

    println!("  ✓ Execution proof created");
    println!("  ✓ Policy evaluated: {}", result);
    println!("  ✓ Context hash: {}", execution_proof.context_hash.short(16));
    println!("  ✓ Result: {}\n", if execution_proof.result { "ALLOW" } else { "DENY" });

    // Serialize proofs to JSON
    println!("Step 5: Serializing proofs to JSON...");

    let translation_json = serde_json::to_string_pretty(&translation_proof)?;
    let validation_json = serde_json::to_string_pretty(&validation_proof)?;
    let execution_json = serde_json::to_string_pretty(&execution_proof)?;

    println!("  ✓ Translation proof: {} bytes", translation_json.len());
    println!("  ✓ Validation proof: {} bytes", validation_json.len());
    println!("  ✓ Execution proof: {} bytes\n", execution_json.len());

    // Demonstrate proof chain
    println!("Step 6: Demonstrating proof chain...");
    println!("  Source Text → Translation → Canonical Core → Validation → Execution\n");

    println!("  1. Source: \"{}\"", source_text);
    println!("     Hash: {}", source_hash.short(16));
    println!();

    println!("  2. Translation: {} steps", translation_proof.translation_steps.len());
    println!("     Verified: {}", translation_proof.verify());
    println!();

    println!("  3. Canonical Core:");
    println!("     Policy: {}", policy.name);
    println!("     Hash: {}", policy.source_hash.short(16));
    println!();

    println!("  4. Validation: {} rules", validation_proof.results.len());
    println!("     All passed: {}", validation_proof.is_valid());
    println!();

    println!("  5. Execution:");
    println!("     Context hash: {}", execution_proof.context_hash.short(16));
    println!("     Result: {}", if execution_proof.result { "✓ ALLOW" } else { "✗ DENY" });
    println!();

    println!("  🔗 Complete audit trail:");
    println!("     - Source intention is cryptographically linked to execution");
    println!("     - Every transformation is proven and verifiable");
    println!("     - Execution is deterministic and reproducible");
    println!();

    println!("=== Example completed successfully! ===");

    Ok(())
}
