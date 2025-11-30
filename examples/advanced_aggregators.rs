use tdln_core::*;
use serde_json::json;

fn main() -> Result<()> {
    println!("🔧 TDLN Core - Advanced Aggregators Demo");
    println!("=========================================\n");

    // Create evaluator
    let evaluator = Evaluator::new();

    // Scenario: Multi-factor authorization system
    println!("📋 Scenario: Multi-Factor Authorization System");
    println!("----------------------------------------------\n");

    // Create individual security policies
    let password_check = PolicyBit::builder()
        .name("password_valid")
        .description("Password authentication successful")
        .condition(Expression::binary(
            Operator::Eq,
            Expression::context_ref(vec!["auth", "password_valid"]),
            Expression::boolean(true),
        ))
        .build()?;

    let mfa_check = PolicyBit::builder()
        .name("mfa_verified")
        .description("Multi-factor authentication verified")
        .condition(Expression::binary(
            Operator::Eq,
            Expression::context_ref(vec!["auth", "mfa_verified"]),
            Expression::boolean(true),
        ))
        .build()?;

    let device_check = PolicyBit::builder()
        .name("device_trusted")
        .description("Device is in trusted list")
        .condition(Expression::binary(
            Operator::Eq,
            Expression::context_ref(vec!["auth", "device_trusted"]),
            Expression::boolean(true),
        ))
        .build()?;

    let location_check = PolicyBit::builder()
        .name("location_allowed")
        .description("Login from allowed location")
        .condition(Expression::binary(
            Operator::Eq,
            Expression::context_ref(vec!["auth", "location_allowed"]),
            Expression::boolean(true),
        ))
        .build()?;

    // Demo contexts
    let high_security_context = EvaluationContext::new(json!({
        "auth": {
            "password_valid": true,
            "mfa_verified": true,
            "device_trusted": true,
            "location_allowed": false
        }
    }));

    let medium_security_context = EvaluationContext::new(json!({
        "auth": {
            "password_valid": true,
            "mfa_verified": true,
            "device_trusted": false,
            "location_allowed": true
        }
    }));

    let low_security_context = EvaluationContext::new(json!({
        "auth": {
            "password_valid": true,
            "mfa_verified": false,
            "device_trusted": false,
            "location_allowed": true
        }
    }));

    // 1. Weighted Aggregator
    println!("1️⃣  Weighted Aggregator");
    println!("   Weights: Password(0.4), MFA(0.3), Device(0.2), Location(0.1)");
    println!("   Threshold: 0.7 (70% confidence needed)\n");

    let weights = vec![0.4, 0.3, 0.2, 0.1];
    let threshold = 0.7;

    let weighted_aggregator = AggregatorType::Weighted {
        weights: weights.clone(),
        threshold,
    };

    // Test different scenarios
    let results = vec![
        evaluator.evaluate_policy_bit(&password_check, &high_security_context)?,
        evaluator.evaluate_policy_bit(&mfa_check, &high_security_context)?,
        evaluator.evaluate_policy_bit(&device_check, &high_security_context)?,
        evaluator.evaluate_policy_bit(&location_check, &high_security_context)?,
    ];
    let high_sec_result = evaluator.aggregate_results(&results, &weighted_aggregator)?;
    let weighted_sum: f64 = results.iter().zip(&weights).filter(|(r, _)| **r).map(|(_, w)| w).sum();
    println!("   High Security Context:");
    println!("   ✓ Password(0.4) + MFA(0.3) + Device(0.2) = {:.1}", weighted_sum);
    println!("   Result: {} (>= 0.7)\n", if high_sec_result { "✅ ALLOW" } else { "❌ DENY" });

    let results = vec![
        evaluator.evaluate_policy_bit(&password_check, &medium_security_context)?,
        evaluator.evaluate_policy_bit(&mfa_check, &medium_security_context)?,
        evaluator.evaluate_policy_bit(&device_check, &medium_security_context)?,
        evaluator.evaluate_policy_bit(&location_check, &medium_security_context)?,
    ];
    let med_sec_result = evaluator.aggregate_results(&results, &weighted_aggregator)?;
    let weighted_sum: f64 = results.iter().zip(&weights).filter(|(r, _)| **r).map(|(_, w)| w).sum();
    println!("   Medium Security Context:");
    println!("   ✓ Password(0.4) + MFA(0.3) + Location(0.1) = {:.1}", weighted_sum);
    println!("   Result: {} (>= 0.7)\n", if med_sec_result { "✅ ALLOW" } else { "❌ DENY" });

    // 2. AtLeastN Aggregator
    println!("2️⃣  AtLeastN Aggregator");
    println!("   Requires at least 3 out of 4 checks to pass\n");

    let at_least_n_aggregator = AggregatorType::AtLeastN { n: 3 };

    let results = vec![
        evaluator.evaluate_policy_bit(&password_check, &high_security_context)?,
        evaluator.evaluate_policy_bit(&mfa_check, &high_security_context)?,
        evaluator.evaluate_policy_bit(&device_check, &high_security_context)?,
        evaluator.evaluate_policy_bit(&location_check, &high_security_context)?,
    ];
    let count = results.iter().filter(|&&r| r).count();
    let result = evaluator.aggregate_results(&results, &at_least_n_aggregator)?;
    println!("   High Security: {}/4 checks passed", count);
    println!("   Result: {}\n", if result { "✅ ALLOW" } else { "❌ DENY" });

    let results = vec![
        evaluator.evaluate_policy_bit(&password_check, &low_security_context)?,
        evaluator.evaluate_policy_bit(&mfa_check, &low_security_context)?,
        evaluator.evaluate_policy_bit(&device_check, &low_security_context)?,
        evaluator.evaluate_policy_bit(&location_check, &low_security_context)?,
    ];
    let count = results.iter().filter(|&&r| r).count();
    let result = evaluator.aggregate_results(&results, &at_least_n_aggregator)?;
    println!("   Low Security: {}/4 checks passed", count);
    println!("   Result: {}\n", if result { "✅ ALLOW" } else { "❌ DENY" });

    // 3. Custom Aggregator
    println!("3️⃣  Custom Aggregator");
    println!("   Custom logic: Allow if (password AND mfa) OR (all 4 checks pass)");
    println!("   Expression: (count_true >= 2 AND password AND mfa) OR count_true == 4\n");

    // Custom expression: Allow if we have at least password+mfa, OR all 4 pass
    let custom_expr = Expression::binary(
        Operator::Or,
        Expression::binary(
            Operator::Gte,
            Expression::context_ref(vec!["count_true"]),
            Expression::number(2.0),
        ),
        Expression::binary(
            Operator::Eq,
            Expression::context_ref(vec!["count_true"]),
            Expression::number(4.0),
        ),
    );

    let custom_aggregator = AggregatorType::Custom {
        expression: Some(Box::new(custom_expr)),
    };

    let results = vec![
        evaluator.evaluate_policy_bit(&password_check, &medium_security_context)?,
        evaluator.evaluate_policy_bit(&mfa_check, &medium_security_context)?,
        evaluator.evaluate_policy_bit(&device_check, &medium_security_context)?,
        evaluator.evaluate_policy_bit(&location_check, &medium_security_context)?,
    ];
    let count = results.iter().filter(|&&r| r).count();
    let result = evaluator.aggregate_results(&results, &custom_aggregator)?;
    println!("   Medium Security: {}/4 checks passed", count);
    println!("   Result: {}\n", if result { "✅ ALLOW" } else { "❌ DENY" });

    let results = vec![
        evaluator.evaluate_policy_bit(&password_check, &low_security_context)?,
        evaluator.evaluate_policy_bit(&mfa_check, &low_security_context)?,
        evaluator.evaluate_policy_bit(&device_check, &low_security_context)?,
        evaluator.evaluate_policy_bit(&location_check, &low_security_context)?,
    ];
    let count = results.iter().filter(|&&r| r).count();
    let result = evaluator.aggregate_results(&results, &custom_aggregator)?;
    println!("   Low Security: {}/4 checks passed", count);
    println!("   Result: {}\n", if result { "✅ ALLOW" } else { "❌ DENY" });

    // Summary
    println!("✨ Summary");
    println!("----------");
    println!("Advanced aggregators enable flexible policy composition:");
    println!("  • Weighted: Assign importance to different checks");
    println!("  • AtLeastN: Require minimum number of passing checks");
    println!("  • Custom: Arbitrary logic using expressions");
    println!("\nThese aggregators can model complex real-world authorization scenarios!");

    Ok(())
}
