/*!
# TDLN DSL Parser Example

This example demonstrates how to use the TDLN DSL to define policies in a clean,
structured text format.
*/

use tdln_core::{DslParser, Evaluator, EvaluationContext, Result};
use serde_json::json;

fn main() -> Result<()> {
    println!("📝 TDLN DSL Parser Demo");
    println!("{}", "=".repeat(60));
    println!();

    let parser = DslParser::new();
    let evaluator = Evaluator::new();

    // Example 1: Simple Policy
    println!("Example 1: Simple Policy Definition");
    println!("{}", "-".repeat(60));

    let dsl1 = r#"
        policy PremiumAccess {
            description: "Check if user has premium access"
            condition: user.account_type == "premium"
        }
    "#;

    println!("DSL Input:");
    println!("{}", dsl1);

    let policy1 = parser.parse_policy(dsl1)?;
    println!("\n✓ Parsed PolicyBit:");
    println!("  Name: {}", policy1.name);
    println!("  Description: {}", policy1.description);
    println!("  Hash: {}", policy1.source_hash.short(8));

    // Test the policy
    let premium_ctx = EvaluationContext::new(json!({
        "user": { "account_type": "premium" }
    }));

    let result = evaluator.evaluate_policy_bit(&policy1, &premium_ctx)?;
    println!("\n✓ Evaluation:");
    println!("  Premium user: {} ✓", result);

    println!("\n");

    // Example 2: Logical Operators
    println!("Example 2: Policy with Logical Operators");
    println!("{}", "-".repeat(60));

    let dsl2 = r#"
        policy SecureTransaction {
            description: "Verify user is authenticated and verified"
            condition: user.authenticated == true && user.verified == true
        }
    "#;

    println!("DSL Input:");
    println!("{}", dsl2);

    let policy2 = parser.parse_policy(dsl2)?;
    println!("\n✓ Parsed PolicyBit:");
    println!("  Name: {}", policy2.name);
    println!("  Description: {}", policy2.description);

    let secure_ctx = EvaluationContext::new(json!({
        "user": {
            "authenticated": true,
            "verified": true
        }
    }));

    let result = evaluator.evaluate_policy_bit(&policy2, &secure_ctx)?;
    println!("\n✓ Evaluation:");
    println!("  Authenticated & Verified: {} ✓", result);

    println!("\n");

    // Example 3: Numeric Comparison
    println!("Example 3: Numeric Validation");
    println!("{}", "-".repeat(60));

    let dsl3 = r#"
        // Validate transaction amounts
        policy AmountLimit {
            description: "Ensure transaction is below limit"
            condition: transaction.amount < 1000
        }
    "#;

    println!("DSL Input:");
    println!("{}", dsl3);

    let policy3 = parser.parse_policy(dsl3)?;
    println!("\n✓ Parsed PolicyBit:");
    println!("  Name: {}", policy3.name);
    println!("  Description: {}", policy3.description);

    let valid_amount_ctx = EvaluationContext::new(json!({
        "transaction": { "amount": 500 }
    }));

    let invalid_amount_ctx = EvaluationContext::new(json!({
        "transaction": { "amount": 1500 }
    }));

    println!("\n✓ Evaluation:");
    let valid = evaluator.evaluate_policy_bit(&policy3, &valid_amount_ctx)?;
    println!("  Amount 500: {} ✓", valid);

    let invalid = evaluator.evaluate_policy_bit(&policy3, &invalid_amount_ctx)?;
    println!("  Amount 1500: {} ✗", invalid);

    println!("\n");

    // Example 4: Negation
    println!("Example 4: Negation Operator");
    println!("{}", "-".repeat(60));

    let dsl4 = r#"
        policy BlockUnverified {
            description: "Block unverified users"
            condition: !user.verified
        }
    "#;

    println!("DSL Input:");
    println!("{}", dsl4);

    let policy4 = parser.parse_policy(dsl4)?;
    println!("\n✓ Parsed PolicyBit:");
    println!("  Name: {}", policy4.name);

    let unverified_ctx = EvaluationContext::new(json!({
        "user": { "verified": false }
    }));

    let result = evaluator.evaluate_policy_bit(&policy4, &unverified_ctx)?;
    println!("\n✓ Evaluation:");
    println!("  Unverified user blocked: {} ✓", result);

    println!("\n");

    // Example 5: Policy Composition
    println!("Example 5: Policy Composition");
    println!("{}", "-".repeat(60));

    let dsl5 = r#"
        composition AllSecurityChecks {
            description: "All security policies must pass"
            type: parallel
            aggregator: all
            policies: [AuthCheck, VerificationCheck, AmountCheck]
        }
    "#;

    println!("DSL Input:");
    println!("{}", dsl5);

    let composition = parser.parse_composition(dsl5)?;
    println!("\n✓ Parsed PolicyComposition:");
    println!("  Name: {}", composition.name);
    println!("  Description: {}", composition.description);
    println!("  Type: {:?}", composition.composition_type);
    println!("  Aggregator: {:?}", composition.aggregator);
    println!("  Policies: {:?}", composition.policies);

    println!("\n");

    // Example 6: Complex Multi-line DSL
    println!("Example 6: Complete DSL Document");
    println!("{}", "-".repeat(60));

    let complete_dsl = r#"
        // Premium download access policy
        policy PremiumDownload {
            description: "Premium users with sufficient balance can download"
            condition: user.tier == "premium" && user.balance >= 10
        }

        // VIP access with multiple conditions
        policy VipAccess {
            description: "VIP access requires premium and verification"
            condition: user.tier == "premium" && user.verified == true || user.tier == "vip"
        }
    "#;

    println!("DSL Document:");
    println!("{}", complete_dsl);

    let premium_download = parser.parse_policy(complete_dsl)?;
    println!("\n✓ First Policy Parsed:");
    println!("  Name: {}", premium_download.name);
    println!("  Description: {}", premium_download.description);
    println!("  Hash: {}", premium_download.source_hash.short(8));

    println!("\n");

    // Example 7: JSON Serialization
    println!("Example 7: Serialization");
    println!("{}", "-".repeat(60));

    let json_output = serde_json::to_string_pretty(&policy1)?;
    println!("PolicyBit as JSON:");
    println!("{}", json_output);

    println!("\n");
    println!("{}", "=".repeat(60));
    println!("✓ DSL Parsing Complete!");
    println!("\nKey Features Demonstrated:");
    println!("  • Clean, readable policy definitions");
    println!("  • Support for comparison operators (==, !=, <, >, <=, >=)");
    println!("  • Logical operators (&&, ||, !)");
    println!("  • Context references (user.account_type)");
    println!("  • String, number, and boolean literals");
    println!("  • Policy composition with aggregators");
    println!("  • Comment support (//)");
    println!("  • Full evaluation and verification");

    Ok(())
}
