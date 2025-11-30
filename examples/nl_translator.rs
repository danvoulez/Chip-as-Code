/*!
# Natural Language to TDLN Translation Example

This example demonstrates how to use the NL translator to convert natural language
policy statements into canonical TDLN PolicyBits with full proof generation.
*/

use tdln_core::{NlTranslator, Evaluator, EvaluationContext, Result};
use serde_json::json;

fn main() -> Result<()> {
    println!("🔮 TDLN Natural Language Translator Demo");
    println!("{}", "=".repeat(60));
    println!();

    // Create translator
    let translator = NlTranslator::new();

    // Example 1: Conditional Access
    println!("Example 1: Conditional Access Policy");
    println!("{}", "-".repeat(60));
    let nl1 = "Premium users can download files";
    println!("Natural Language: \"{}\"", nl1);

    let result1 = translator.translate(nl1)?;
    println!("\n✓ Generated PolicyBit:");
    println!("  Name: {}", result1.policy_bit.name);
    println!("  Description: {}", result1.policy_bit.description);
    println!("  Hash: {}", result1.policy_bit.source_hash.short(8));
    println!("  Confidence: {:.1}%", result1.confidence * 100.0);

    println!("\n✓ Extracted Entities:");
    for (key, value) in &result1.extracted_entities {
        println!("  {}: {}", key, value);
    }

    println!("\n✓ Translation Proof:");
    println!("  Source Hash: {}", result1.proof.source_hash.short(8));
    println!("  Target Hash: {}", result1.proof.target_core_hash.short(8));
    println!("  Steps: {}", result1.proof.translation_steps.len());

    // Test the generated policy
    let evaluator = Evaluator::new();

    let premium_context = EvaluationContext::new(json!({
        "user": { "account_type": "premium" }
    }));

    let basic_context = EvaluationContext::new(json!({
        "user": { "account_type": "basic" }
    }));

    println!("\n✓ Evaluation:");
    let premium_result = evaluator.evaluate_policy_bit(&result1.policy_bit, &premium_context)?;
    println!("  Premium user: {} ✓", premium_result);

    let basic_result = evaluator.evaluate_policy_bit(&result1.policy_bit, &basic_context)?;
    println!("  Basic user: {} ✗", basic_result);

    println!("\n");

    // Example 2: Numeric Comparison
    println!("Example 2: Numeric Validation Policy");
    println!("{}", "-".repeat(60));
    let nl2 = "Amount must be less than 1000";
    println!("Natural Language: \"{}\"", nl2);

    let result2 = translator.translate(nl2)?;
    println!("\n✓ Generated PolicyBit:");
    println!("  Name: {}", result2.policy_bit.name);
    println!("  Description: {}", result2.policy_bit.description);
    println!("  Hash: {}", result2.policy_bit.source_hash.short(8));
    println!("  Confidence: {:.1}%", result2.confidence * 100.0);

    // Test with different amounts
    let valid_amount = EvaluationContext::new(json!({
        "amount": 500
    }));

    let invalid_amount = EvaluationContext::new(json!({
        "amount": 1500
    }));

    println!("\n✓ Evaluation:");
    let valid_result = evaluator.evaluate_policy_bit(&result2.policy_bit, &valid_amount)?;
    println!("  Amount 500: {} ✓", valid_result);

    let invalid_result = evaluator.evaluate_policy_bit(&result2.policy_bit, &invalid_amount)?;
    println!("  Amount 1500: {} ✗", invalid_result);

    println!("\n");

    // Example 3: Verification Policy
    println!("Example 3: Verification Policy");
    println!("{}", "-".repeat(60));
    let nl3 = "User must be verified";
    println!("Natural Language: \"{}\"", nl3);

    let result3 = translator.translate(nl3)?;
    println!("\n✓ Generated PolicyBit:");
    println!("  Name: {}", result3.policy_bit.name);
    println!("  Description: {}", result3.policy_bit.description);
    println!("  Hash: {}", result3.policy_bit.source_hash.short(8));

    let verified_user = EvaluationContext::new(json!({
        "user": { "verified": true }
    }));

    let unverified_user = EvaluationContext::new(json!({
        "user": { "verified": false }
    }));

    println!("\n✓ Evaluation:");
    let verified_result = evaluator.evaluate_policy_bit(&result3.policy_bit, &verified_user)?;
    println!("  Verified user: {} ✓", verified_result);

    let unverified_result = evaluator.evaluate_policy_bit(&result3.policy_bit, &unverified_user)?;
    println!("  Unverified user: {} ✗", unverified_result);

    println!("\n");

    // Example 4: Serialization
    println!("Example 4: Proof Serialization");
    println!("{}", "-".repeat(60));

    let json_output = serde_json::to_string_pretty(&result1)?;
    println!("Generated policy and proof as JSON:");
    println!("{}", json_output);

    println!("\n");
    println!("{}", "=".repeat(60));
    println!("✓ Natural Language Translation Complete!");
    println!("\nKey Features Demonstrated:");
    println!("  • Pattern-based NL parsing");
    println!("  • Intent extraction and entity recognition");
    println!("  • Canonical PolicyBit generation");
    println!("  • Translation proof with cryptographic hashing");
    println!("  • Full evaluation and verification");
    println!("  • JSON serialization for auditability");

    Ok(())
}
