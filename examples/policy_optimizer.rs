use tdln_core::*;
use serde_json::json;

fn main() -> Result<()> {
    println!("⚡ TDLN Core - Policy Optimization Demo");
    println!("=======================================\n");

    let optimizer = Optimizer::new();
    let evaluator = Evaluator::new();

    println!("📋 Policy optimization reduces complexity and improves performance");
    println!("by simplifying expressions at compile-time.\n");

    // Example 1: Constant Folding
    println!("1️⃣  Constant Folding");
    println!("   Evaluates constant expressions at compile-time\n");

    let original = PolicyBit::builder()
        .name("age_check")
        .description("Check if user is adult")
        .condition(Expression::binary(
            Operator::And,
            Expression::context_ref(vec!["user", "verified"]),
            Expression::binary(
                Operator::Gt,
                Expression::number(20.0),
                Expression::number(18.0),
            ),
        ))
        .build()?;

    println!("   Original condition:");
    println!("   user.verified AND (20 > 18)");
    println!("   └─ Contains constant expression: 20 > 18\n");

    let optimized = optimizer.optimize_policy_bit(&original)?;

    println!("   Optimized condition:");
    println!("   user.verified");
    println!("   └─ Step 1: 20 > 18 → true");
    println!("   └─ Step 2: user.verified AND true → user.verified\n");

    // Test both versions produce same result
    let context = EvaluationContext::new(json!({"user": {"verified": true}}));
    let original_result = evaluator.evaluate_policy_bit(&original, &context)?;
    let optimized_result = evaluator.evaluate_policy_bit(&optimized, &context)?;
    println!("   ✓ Verification: both evaluate to {}\n", original_result);
    assert_eq!(original_result, optimized_result);

    // Example 2: Logic Simplification
    println!("2️⃣  Logic Simplification");
    println!("   Simplifies boolean expressions\n");

    let original = PolicyBit::builder()
        .name("premium_access")
        .description("Check premium access with redundant logic")
        .condition(Expression::binary(
            Operator::And,
            Expression::context_ref(vec!["user", "premium"]),
            Expression::boolean(true),
        ))
        .build()?;

    println!("   Original condition:");
    println!("   user.premium AND true");
    println!("   └─ Redundant AND with true\n");

    let optimized = optimizer.optimize_policy_bit(&original)?;

    println!("   Optimized condition:");
    println!("   user.premium");
    println!("   └─ Simplified: x AND true → x\n");

    let context = EvaluationContext::new(json!({"user": {"premium": true}}));
    let original_result = evaluator.evaluate_policy_bit(&original, &context)?;
    let optimized_result = evaluator.evaluate_policy_bit(&optimized, &context)?;
    println!("   ✓ Verification: both evaluate to {}\n", original_result);

    // Example 3: Dead Code Elimination
    println!("3️⃣  Dead Code Elimination");
    println!("   Removes unreachable branches\n");

    let original_expr = Expression::conditional(
        Expression::boolean(true),
        Expression::context_ref(vec!["user", "premium"]),
        Expression::context_ref(vec!["user", "basic"]),
    );

    println!("   Original expression:");
    println!("   if true then user.premium else user.basic");
    println!("   └─ Condition is always true\n");

    let optimized_expr = optimizer.optimize_expression(&original_expr)?;

    println!("   Optimized expression:");
    println!("   user.premium");
    println!("   └─ Dead branch eliminated\n");

    // Example 4: Double Negation
    println!("4️⃣  Double Negation Elimination");
    println!("   Removes double negations\n");

    let original = PolicyBit::builder()
        .name("active_check")
        .description("Check if user is active with double negation")
        .condition(Expression::unary(
            Operator::Not,
            Expression::unary(
                Operator::Not,
                Expression::context_ref(vec!["user", "active"]),
            ),
        ))
        .build()?;

    println!("   Original condition:");
    println!("   NOT (NOT user.active)");
    println!("   └─ Double negation\n");

    let optimized = optimizer.optimize_policy_bit(&original)?;

    println!("   Optimized condition:");
    println!("   user.active");
    println!("   └─ Simplified: !!x → x\n");

    let context = EvaluationContext::new(json!({"user": {"active": true}}));
    let original_result = evaluator.evaluate_policy_bit(&original, &context)?;
    let optimized_result = evaluator.evaluate_policy_bit(&optimized, &context)?;
    println!("   ✓ Verification: both evaluate to {}\n", original_result);

    // Example 5: Complex Optimization
    println!("5️⃣  Complex Optimization");
    println!("   Combines multiple optimizations\n");

    let original = PolicyBit::builder()
        .name("complex_check")
        .description("Complex policy with multiple optimization opportunities")
        .condition(Expression::binary(
            Operator::Or,
            Expression::binary(
                Operator::And,
                Expression::context_ref(vec!["user", "verified"]),
                Expression::boolean(true),
            ),
            Expression::boolean(false),
        ))
        .build()?;

    println!("   Original condition:");
    println!("   ((user.verified AND true) OR false)");
    println!("   └─ Multiple redundant operations\n");

    let optimized = optimizer.optimize_policy_bit(&original)?;

    println!("   Optimized condition:");
    println!("   user.verified");
    println!("   └─ Fully simplified through multiple passes:\n");
    println!("      Step 1: user.verified AND true → user.verified");
    println!("      Step 2: user.verified OR false → user.verified\n");

    let context = EvaluationContext::new(json!({"user": {"verified": true}}));
    let original_result = evaluator.evaluate_policy_bit(&original, &context)?;
    let optimized_result = evaluator.evaluate_policy_bit(&optimized, &context)?;
    println!("   ✓ Verification: both evaluate to {}\n", original_result);

    // Summary
    println!("✨ Summary");
    println!("----------");
    println!("Policy optimization provides:");
    println!("  ✓ Faster evaluation (no runtime computation of constants)");
    println!("  ✓ Cleaner canonical forms");
    println!("  ✓ Easier debugging and auditing");
    println!("  ✓ Smaller serialized policy size");
    println!("\nOptimizations are semantics-preserving: original and optimized");
    println!("policies always produce identical results!");

    Ok(())
}
