use crate::error::{Result, TdlnError};
use crate::expression::Literal;
use std::collections::HashMap;

/// Type alias for built-in function implementations
pub type BuiltinFunction = fn(&[Literal]) -> Result<Literal>;

/// Registry of built-in functions
pub struct BuiltinRegistry {
    functions: HashMap<String, BuiltinFunction>,
}

impl BuiltinRegistry {
    /// Creates a new registry with all standard built-in functions
    pub fn new() -> Self {
        let mut registry = Self {
            functions: HashMap::new(),
        };

        // Type checking functions
        registry.register("is_string", is_string);
        registry.register("is_number", is_number);
        registry.register("is_boolean", is_boolean);
        registry.register("is_array", is_array);
        registry.register("is_object", is_object);

        // String operations
        registry.register("string_length", string_length);
        registry.register("string_contains", string_contains);
        registry.register("string_starts_with", string_starts_with);
        registry.register("string_ends_with", string_ends_with);

        // Numeric operations
        registry.register("math_abs", math_abs);
        registry.register("math_floor", math_floor);
        registry.register("math_ceil", math_ceil);

        // Array operations
        registry.register("array_length", array_length);
        registry.register("array_contains", array_contains);

        registry
    }

    /// Registers a new built-in function
    pub fn register<S: Into<String>>(&mut self, name: S, func: BuiltinFunction) {
        self.functions.insert(name.into(), func);
    }

    /// Calls a built-in function by name
    pub fn call(&self, name: &str, args: &[Literal]) -> Result<Literal> {
        let func = self
            .functions
            .get(name)
            .ok_or_else(|| TdlnError::EvaluationError(format!("Unknown function: {}", name)))?;

        func(args)
    }

    /// Returns true if a function exists
    pub fn has_function(&self, name: &str) -> bool {
        self.functions.contains_key(name)
    }

    /// Returns all function names
    pub fn function_names(&self) -> Vec<&String> {
        self.functions.keys().collect()
    }
}

impl Default for BuiltinRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Type Checking Functions
// ============================================================================

fn is_string(args: &[Literal]) -> Result<Literal> {
    if args.len() != 1 {
        return Err(TdlnError::EvaluationError(
            "is_string expects 1 argument".to_string(),
        ));
    }

    Ok(Literal::Boolean(matches!(args[0], Literal::String(_))))
}

fn is_number(args: &[Literal]) -> Result<Literal> {
    if args.len() != 1 {
        return Err(TdlnError::EvaluationError(
            "is_number expects 1 argument".to_string(),
        ));
    }

    Ok(Literal::Boolean(matches!(args[0], Literal::Number(_))))
}

fn is_boolean(args: &[Literal]) -> Result<Literal> {
    if args.len() != 1 {
        return Err(TdlnError::EvaluationError(
            "is_boolean expects 1 argument".to_string(),
        ));
    }

    Ok(Literal::Boolean(matches!(args[0], Literal::Boolean(_))))
}

fn is_array(args: &[Literal]) -> Result<Literal> {
    if args.len() != 1 {
        return Err(TdlnError::EvaluationError(
            "is_array expects 1 argument".to_string(),
        ));
    }

    Ok(Literal::Boolean(matches!(args[0], Literal::Array(_))))
}

fn is_object(args: &[Literal]) -> Result<Literal> {
    if args.len() != 1 {
        return Err(TdlnError::EvaluationError(
            "is_object expects 1 argument".to_string(),
        ));
    }

    Ok(Literal::Boolean(matches!(args[0], Literal::Object(_))))
}

// ============================================================================
// String Operations
// ============================================================================

fn string_length(args: &[Literal]) -> Result<Literal> {
    if args.len() != 1 {
        return Err(TdlnError::EvaluationError(
            "string_length expects 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Literal::String(s) => Ok(Literal::Number(s.len() as f64)),
        _ => Err(TdlnError::TypeError(
            "string_length expects a string argument".to_string(),
        )),
    }
}

fn string_contains(args: &[Literal]) -> Result<Literal> {
    if args.len() != 2 {
        return Err(TdlnError::EvaluationError(
            "string_contains expects 2 arguments".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Literal::String(s), Literal::String(substring)) => {
            Ok(Literal::Boolean(s.contains(substring.as_str())))
        }
        _ => Err(TdlnError::TypeError(
            "string_contains expects string arguments".to_string(),
        )),
    }
}

fn string_starts_with(args: &[Literal]) -> Result<Literal> {
    if args.len() != 2 {
        return Err(TdlnError::EvaluationError(
            "string_starts_with expects 2 arguments".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Literal::String(s), Literal::String(prefix)) => {
            Ok(Literal::Boolean(s.starts_with(prefix.as_str())))
        }
        _ => Err(TdlnError::TypeError(
            "string_starts_with expects string arguments".to_string(),
        )),
    }
}

fn string_ends_with(args: &[Literal]) -> Result<Literal> {
    if args.len() != 2 {
        return Err(TdlnError::EvaluationError(
            "string_ends_with expects 2 arguments".to_string(),
        ));
    }

    match (&args[0], &args[1]) {
        (Literal::String(s), Literal::String(suffix)) => {
            Ok(Literal::Boolean(s.ends_with(suffix.as_str())))
        }
        _ => Err(TdlnError::TypeError(
            "string_ends_with expects string arguments".to_string(),
        )),
    }
}

// ============================================================================
// Numeric Operations
// ============================================================================

fn math_abs(args: &[Literal]) -> Result<Literal> {
    if args.len() != 1 {
        return Err(TdlnError::EvaluationError(
            "math_abs expects 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Literal::Number(n) => Ok(Literal::Number(n.abs())),
        _ => Err(TdlnError::TypeError(
            "math_abs expects a number argument".to_string(),
        )),
    }
}

fn math_floor(args: &[Literal]) -> Result<Literal> {
    if args.len() != 1 {
        return Err(TdlnError::EvaluationError(
            "math_floor expects 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Literal::Number(n) => Ok(Literal::Number(n.floor())),
        _ => Err(TdlnError::TypeError(
            "math_floor expects a number argument".to_string(),
        )),
    }
}

fn math_ceil(args: &[Literal]) -> Result<Literal> {
    if args.len() != 1 {
        return Err(TdlnError::EvaluationError(
            "math_ceil expects 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Literal::Number(n) => Ok(Literal::Number(n.ceil())),
        _ => Err(TdlnError::TypeError(
            "math_ceil expects a number argument".to_string(),
        )),
    }
}

// ============================================================================
// Array Operations
// ============================================================================

fn array_length(args: &[Literal]) -> Result<Literal> {
    if args.len() != 1 {
        return Err(TdlnError::EvaluationError(
            "array_length expects 1 argument".to_string(),
        ));
    }

    match &args[0] {
        Literal::Array(arr) => Ok(Literal::Number(arr.len() as f64)),
        _ => Err(TdlnError::TypeError(
            "array_length expects an array argument".to_string(),
        )),
    }
}

fn array_contains(args: &[Literal]) -> Result<Literal> {
    if args.len() != 2 {
        return Err(TdlnError::EvaluationError(
            "array_contains expects 2 arguments".to_string(),
        ));
    }

    match &args[0] {
        Literal::Array(arr) => Ok(Literal::Boolean(arr.contains(&args[1]))),
        _ => Err(TdlnError::TypeError(
            "array_contains expects an array as first argument".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_checking() {
        let registry = BuiltinRegistry::new();

        let result = registry
            .call("is_string", &[Literal::String("test".to_string())])
            .unwrap();
        assert_eq!(result, Literal::Boolean(true));

        let result = registry.call("is_number", &[Literal::Number(42.0)]).unwrap();
        assert_eq!(result, Literal::Boolean(true));

        let result = registry
            .call("is_boolean", &[Literal::Boolean(true)])
            .unwrap();
        assert_eq!(result, Literal::Boolean(true));
    }

    #[test]
    fn test_string_operations() {
        let registry = BuiltinRegistry::new();

        let result = registry
            .call("string_length", &[Literal::String("hello".to_string())])
            .unwrap();
        assert_eq!(result, Literal::Number(5.0));

        let result = registry
            .call(
                "string_contains",
                &[
                    Literal::String("hello world".to_string()),
                    Literal::String("world".to_string()),
                ],
            )
            .unwrap();
        assert_eq!(result, Literal::Boolean(true));
    }

    #[test]
    fn test_math_operations() {
        let registry = BuiltinRegistry::new();

        let result = registry.call("math_abs", &[Literal::Number(-42.0)]).unwrap();
        assert_eq!(result, Literal::Number(42.0));

        let result = registry
            .call("math_floor", &[Literal::Number(3.7)])
            .unwrap();
        assert_eq!(result, Literal::Number(3.0));

        let result = registry.call("math_ceil", &[Literal::Number(3.2)]).unwrap();
        assert_eq!(result, Literal::Number(4.0));
    }

    #[test]
    fn test_array_operations() {
        let registry = BuiltinRegistry::new();

        let arr = Literal::Array(vec![
            Literal::Number(1.0),
            Literal::Number(2.0),
            Literal::Number(3.0),
        ]);

        let result = registry.call("array_length", &[arr.clone()]).unwrap();
        assert_eq!(result, Literal::Number(3.0));

        let result = registry
            .call("array_contains", &[arr, Literal::Number(2.0)])
            .unwrap();
        assert_eq!(result, Literal::Boolean(true));
    }
}
