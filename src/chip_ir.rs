//! # Chip IR — Intermediate Representation for Semantic Chips
//!
//! This module provides the core data structures and algorithms for parsing,
//! evaluating, and hashing semantic chips defined in the `.chip` text DNA format.
//!
//! ## Example
//!
//! ```rust
//! use chip_as_code::Chip;
//!
//! let chip = Chip::parse(r#"
//!     CHIP v0
//!     FEATURES n=2
//!     GATES m=1
//!     g0 = AND(f0,f1)
//!     OUTPUT = g0
//! "#).unwrap();
//!
//! assert!(chip.eval(&[true, true]).unwrap());
//! assert!(!chip.eval(&[true, false]).unwrap());
//! ```

use blake3::Hasher;
use logline::json_atomic;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fmt::Write as FmtWrite;
use std::str::FromStr;
use thiserror::Error;

/// Hash of a chip (hex-encoded BLAKE3 of JSON✯Atomic canonical bytes).
///
/// This is the canonical identity of a chip — two chips with the same
/// semantic behavior will have the same hash.
pub type ChipHash = String;

#[derive(Serialize)]
struct ChipAtom<'a> {
    kind: &'static str,
    version: u32,
    features: usize,
    gates: &'a [Gate],
    output: &'a Ref,
}

/// A semantic chip — a boolean circuit defined as a DAG of gates.
///
/// # DNA Format
///
/// ```text
/// CHIP v0
/// FEATURES n=<N>
/// GATES m=<M>
/// g0 = OP(inputs...)
/// ...
/// OUTPUT = gK
/// ```
///
/// # Example
///
/// ```rust
/// use chip_as_code::Chip;
///
/// let xor = Chip::parse(r#"
///     CHIP v0
///     FEATURES n=2
///     GATES m=5
///     g0 = NOT(f1)
///     g1 = AND(f0,g0)
///     g2 = NOT(f0)
///     g3 = AND(g2,f1)
///     g4 = OR(g1,g3)
///     OUTPUT = g4
/// "#).unwrap();
///
/// // XOR truth table
/// assert!(!xor.eval(&[false, false]).unwrap());
/// assert!(xor.eval(&[true, false]).unwrap());
/// assert!(xor.eval(&[false, true]).unwrap());
/// assert!(!xor.eval(&[true, true]).unwrap());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chip {
    /// Number of input features (f0, f1, ..., fN-1)
    pub features: usize,
    /// Gates in evaluation order (g0, g1, ..., gM-1)
    pub gates: Vec<Gate>,
    /// Output reference (which gate or feature produces the final result)
    pub output: Ref,
}

/// A single gate in the chip circuit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gate {
    /// The gate operation
    pub op: GateOp,
}

/// Gate operation types.
///
/// These are the primitive operations that make up a semantic chip.
/// They correspond to the "semantic transistors" in the Chip as Code paradigm.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateOp {
    /// Logical AND: all inputs must be true
    And(Vec<Ref>),
    /// Logical OR: at least one input must be true
    Or(Vec<Ref>),
    /// Logical NOT: negate single input
    Not(Ref),
    /// Threshold: at least k inputs must be true
    Threshold { k: usize, inputs: Vec<Ref> },
}

/// Reference to a feature or gate output.
///
/// Gates can only reference earlier gates (forming a DAG).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Ref {
    /// Reference to input feature fN
    Feature(usize),
    /// Reference to gate output gN
    Gate(usize),
}

/// Errors that can occur when parsing a chip.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ChipParseError {
    /// Input was empty or whitespace-only
    #[error("empty input")]
    Empty,
    /// Missing or invalid "CHIP v0" header
    #[error("invalid header")]
    InvalidHeader,
    /// Missing or invalid "FEATURES n=N" line
    #[error("invalid features line")]
    InvalidFeatures,
    /// Missing or invalid "GATES m=M" line
    #[error("invalid gates line")]
    InvalidGates,
    /// Invalid gate definition line
    #[error("invalid gate line: {0}")]
    InvalidGate(String),
    /// Missing or invalid "OUTPUT = gK" line
    #[error("invalid output line")]
    InvalidOutput,
    /// Gate references a feature or gate that doesn't exist
    #[error("reference out of range")]
    RefOutOfRange,
}

#[derive(Debug, Error)]
pub enum EvalError {
    #[error("feature length mismatch: expected {expected}, got {got}")]
    FeatureMismatch { expected: usize, got: usize },
}

impl Chip {
    pub fn parse(text: &str) -> Result<Self, ChipParseError> {
        if text.trim().is_empty() {
            return Err(ChipParseError::Empty);
        }
        let normalized: Vec<&str> = text
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect();
        if normalized.len() < 4 {
            return Err(ChipParseError::InvalidHeader);
        }
        if normalized[0] != "CHIP v0" {
            return Err(ChipParseError::InvalidHeader);
        }
        let features = parse_kv_line(&normalized[1], "FEATURES", "n=")
            .ok_or(ChipParseError::InvalidFeatures)?;
        let gates_count = parse_kv_line(&normalized[2], "GATES", "m=")
            .ok_or(ChipParseError::InvalidGates)?;
        let mut gates = Vec::with_capacity(gates_count);
        for (idx, line) in normalized.iter().enumerate().skip(3) {
            if line.starts_with("OUTPUT") {
                break;
            }
            let gate_idx = idx - 3;
            if gate_idx >= gates_count {
                break;
            }
            let parsed = parse_gate_line(line, gate_idx)?;
            validate_gate_refs(&parsed.op, features, gate_idx)?;
            gates.push(parsed);
        }
        if gates.len() != gates_count {
            return Err(ChipParseError::InvalidGates);
        }
        let output_line = normalized
            .iter()
            .find(|l| l.starts_with("OUTPUT"))
            .ok_or(ChipParseError::InvalidOutput)?;
        let output = parse_output_line(output_line)?;
        validate_ref(&output, features, gates.len())?;
        Ok(Chip {
            features,
            gates,
            output,
        })
    }

    pub fn canonical_text(&self) -> String {
        let mut buf = String::new();
        writeln!(&mut buf, "CHIP v0").unwrap();
        writeln!(&mut buf, "FEATURES n={}", self.features).unwrap();
        writeln!(&mut buf, "GATES m={}", self.gates.len()).unwrap();
        for (idx, gate) in self.gates.iter().enumerate() {
            let op_text = gate.op.format();
            writeln!(&mut buf, "g{} = {}", idx, op_text).unwrap();
        }
        writeln!(&mut buf, "OUTPUT = {}", format_ref(&self.output)).unwrap();
        buf
    }

    pub fn hash(&self) -> ChipHash {
        let atom = ChipAtom {
            kind: "chip",
            version: 0,
            features: self.features,
            gates: &self.gates,
            output: &self.output,
        };
        let canon = json_atomic::canonize(&atom).expect("chip canonize");
        let mut hasher = Hasher::new();
        hasher.update(&canon);
        let digest = hasher.finalize();
        hex::encode(digest.as_bytes())
    }

    pub fn eval(&self, features: &[bool]) -> Result<bool, EvalError> {
        if features.len() != self.features {
            return Err(EvalError::FeatureMismatch {
                expected: self.features,
                got: features.len(),
            });
        }
        let mut gate_values: Vec<bool> = Vec::with_capacity(self.gates.len());
        for gate in &self.gates {
            let value = match &gate.op {
                GateOp::And(args) => args.iter().all(|r| resolve_ref(r, features, &gate_values)),
                GateOp::Or(args) => args.iter().any(|r| resolve_ref(r, features, &gate_values)),
                GateOp::Not(r) => !resolve_ref(r, features, &gate_values),
                GateOp::Threshold { k, inputs } => {
                    let mut count = 0usize;
                    for r in inputs {
                        if resolve_ref(r, features, &gate_values) {
                            count += 1;
                        }
                    }
                    count >= *k
                }
            };
            gate_values.push(value);
        }
        Ok(resolve_ref(&self.output, features, &gate_values))
    }

    pub fn live_gate_count(&self) -> usize {
        let mut live = vec![false; self.gates.len()];
        fn mark(idx: usize, live: &mut [bool], gates: &[Gate]) {
            if idx >= gates.len() || live[idx] {
                return;
            }
            live[idx] = true;
            match &gates[idx].op {
                GateOp::And(args) | GateOp::Or(args) => {
                    for r in args {
                        if let Ref::Gate(g) = r {
                            mark(*g, live, gates);
                        }
                    }
                }
                GateOp::Not(r) => {
                    if let Ref::Gate(g) = r {
                        mark(*g, live, gates);
                    }
                }
                GateOp::Threshold { inputs, .. } => {
                    for r in inputs {
                        if let Ref::Gate(g) = r {
                            mark(*g, live, gates);
                        }
                    }
                }
            }
        }
        if let Ref::Gate(g) = self.output {
            mark(g, &mut live, &self.gates);
        }
        live.into_iter().filter(|b| *b).count()
    }

    pub fn mutate_random<R: Rng>(
        &self,
        rng: &mut R,
        max_gates: usize,
        bias_primary_inputs: bool,
    ) -> Self {
        let mut next = self.clone();
        #[derive(Clone, Copy)]
        enum MutKind {
            SwapOutput,
            ReplaceInput,
            ToggleOp,
            AddGate,
            RemoveGate,
            /// Replaces the gate's operation with NOT(random_ref). Does NOT wrap the existing gate.
            ReplaceWithNot,
            AdjustThresh,
        }
        let choices = [
            MutKind::SwapOutput,
            MutKind::ReplaceInput,
            MutKind::ToggleOp,
            MutKind::AddGate,
            MutKind::RemoveGate,
            MutKind::ReplaceWithNot,
            MutKind::AdjustThresh,
        ];
        let choice = choices[rng.gen_range(0..choices.len())];
        match choice {
            MutKind::SwapOutput => {
                if !next.gates.is_empty() {
                    let idx = rng.gen_range(0..(next.gates.len() + next.features));
                    let picked = biased_ref(idx, next.features, bias_primary_inputs);
                    next.output = picked
                        .unwrap_or_else(|| ref_from_index(idx, next.features));
                }
            }
            MutKind::ReplaceInput => {
                if next.gates.is_empty() {
                    return next;
                }
                let idx = rng.gen_range(0..next.gates.len());
                if let Some(gate) = next.gates.get_mut(idx) {
                    gate.op.replace_random_input(
                        rng,
                        next.features,
                        idx,
                        bias_primary_inputs,
                    );
                }
            }
            MutKind::ToggleOp => {
                let len = next.gates.len().max(1);
                let idx = rng.gen_range(0..len);
                if let Some(gate) = next.gates.get_mut(idx) {
                    gate.op.toggle_logic();
                }
            }
            MutKind::AddGate => {
                if next.gates.len() < max_gates {
                    let new_gate = Gate {
                        op: GateOp::random(
                            rng,
                            next.features,
                            next.gates.len(),
                            bias_primary_inputs,
                        ),
                    };
                    next.gates.push(new_gate);
                }
            }
            MutKind::RemoveGate => {
                if !next.gates.is_empty() {
                    next.gates.pop();
                    if let Ref::Gate(idx) = next.output {
                        if idx >= next.gates.len() {
                            next.output = Ref::Feature(idx % next.features);
                        }
                    }
                }
            }
            MutKind::ReplaceWithNot => {
                if next.gates.is_empty() {
                    return next;
                }
                let idx = rng.gen_range(0..next.gates.len());
                if let Some(gate) = next.gates.get_mut(idx) {
                    // Replace gate op with NOT(random_ref), does not preserve original logic
                    let r = biased_random_ref(rng, next.features, idx, bias_primary_inputs)
                        .unwrap_or_else(|| random_ref(rng, next.features, idx));
                    gate.op = GateOp::Not(r);
                }
            }
            MutKind::AdjustThresh => {
                if next.gates.is_empty() {
                    return next;
                }
                let idx = rng.gen_range(0..next.gates.len());
                if let Some(Gate {
                    op: GateOp::Threshold { k, inputs },
                }) = next.gates.get_mut(idx)
                {
                    if *k > 0 && rng.gen_bool(0.5) {
                        *k -= 1;
                    } else {
                        *k += 1;
                        if *k > inputs.len() {
                            *k = inputs.len();
                        }
                    }
                }
            }
        }
        if bias_primary_inputs && !next.depends_on_primary_inputs() && rng.gen_bool(0.2) {
            // Nudge exploration: occasionally force output to touch f0/f1 when bias is on.
            next.output = if rng.gen_bool(0.5) {
                Ref::Feature(0)
            } else {
                Ref::Feature(1)
            };
        }
        next
    }

    pub fn depends_on_primary_inputs(&self) -> bool {
        fn ref_is_primary(r: &Ref) -> bool {
            matches!(r, Ref::Feature(0) | Ref::Feature(1))
        }

        fn visit(idx: usize, chip: &Chip, seen: &mut [bool]) -> bool {
            if idx >= chip.gates.len() || seen[idx] {
                return false;
            }
            seen[idx] = true;
            match &chip.gates[idx].op {
                GateOp::And(args) | GateOp::Or(args) => {
                    for r in args {
                        if ref_is_primary(r) {
                            return true;
                        }
                        if let Ref::Gate(g) = r {
                            if visit(*g, chip, seen) {
                                return true;
                            }
                        }
                    }
                }
                GateOp::Not(r) => {
                    if ref_is_primary(r) {
                        return true;
                    }
                    if let Ref::Gate(g) = r {
                        if visit(*g, chip, seen) {
                            return true;
                        }
                    }
                }
                GateOp::Threshold { inputs, .. } => {
                    for r in inputs {
                        if ref_is_primary(r) {
                            return true;
                        }
                        if let Ref::Gate(g) = r {
                            if visit(*g, chip, seen) {
                                return true;
                            }
                        }
                    }
                }
            }
            false
        }

        if ref_is_primary(&self.output) {
            return true;
        }
        if let Ref::Gate(g) = self.output {
            let mut seen = vec![false; self.gates.len()];
            return visit(g, self, &mut seen);
        }
        false
    }
}

impl GateOp {
    fn format(&self) -> String {
        match self {
            GateOp::And(args) => format!("AND({})", join_refs(args)),
            GateOp::Or(args) => format!("OR({})", join_refs(args)),
            GateOp::Not(r) => format!("NOT({})", format_ref(r)),
            GateOp::Threshold { k, inputs } => format!("THRESH({},{})", k, join_refs(inputs)),
        }
    }

    fn replace_random_input<R: Rng>(
        &mut self,
        rng: &mut R,
        features: usize,
        gate_idx: usize,
        bias_primary_inputs: bool,
    ) {
        match self {
            GateOp::And(args) | GateOp::Or(args) | GateOp::Threshold { inputs: args, .. } => {
                if args.is_empty() {
                    return;
                }
                let pos = rng.gen_range(0..args.len());
                args[pos] = biased_random_ref(rng, features, gate_idx, bias_primary_inputs)
                    .unwrap_or_else(|| random_ref(rng, features, gate_idx));
            }
            GateOp::Not(arg) => {
                *arg = biased_random_ref(rng, features, gate_idx, bias_primary_inputs)
                    .unwrap_or_else(|| random_ref(rng, features, gate_idx));
            }
        }
    }

    fn toggle_logic(&mut self) {
        match self {
            GateOp::And(args) => *self = GateOp::Or(args.clone()),
            GateOp::Or(args) => *self = GateOp::And(args.clone()),
            GateOp::Not(arg) => *self = GateOp::Not(arg.clone()),
            GateOp::Threshold { k, inputs } => {
                let new_k = if *k > 0 { *k - 1 } else { *k + 1 };
                *self = GateOp::Threshold {
                    k: new_k.clamp(0, inputs.len()),
                    inputs: inputs.clone(),
                };
            }
        }
    }

    fn random<R: Rng>(
        rng: &mut R,
        features: usize,
        gate_idx: usize,
        bias_primary_inputs: bool,
    ) -> Self {
        let choice = rng.gen_range(0..4);
        match choice {
            0 => GateOp::And(vec![
                biased_random_ref(rng, features, gate_idx, bias_primary_inputs)
                    .unwrap_or_else(|| random_ref(rng, features, gate_idx)),
                biased_random_ref(rng, features, gate_idx, bias_primary_inputs)
                    .unwrap_or_else(|| random_ref(rng, features, gate_idx)),
            ]),
            1 => GateOp::Or(vec![
                biased_random_ref(rng, features, gate_idx, bias_primary_inputs)
                    .unwrap_or_else(|| random_ref(rng, features, gate_idx)),
                biased_random_ref(rng, features, gate_idx, bias_primary_inputs)
                    .unwrap_or_else(|| random_ref(rng, features, gate_idx)),
            ]),
            2 => GateOp::Not(
                biased_random_ref(rng, features, gate_idx, bias_primary_inputs)
                    .unwrap_or_else(|| random_ref(rng, features, gate_idx)),
            ),
            _ => {
                let inputs = vec![
                    biased_random_ref(rng, features, gate_idx, bias_primary_inputs)
                        .unwrap_or_else(|| random_ref(rng, features, gate_idx)),
                    biased_random_ref(rng, features, gate_idx, bias_primary_inputs)
                        .unwrap_or_else(|| random_ref(rng, features, gate_idx)),
                ];
                GateOp::Threshold { k: 1, inputs }
            }
        }
    }
}

fn parse_kv_line(line: &str, prefix: &str, key: &str) -> Option<usize> {
    if !line.starts_with(prefix) {
        return None;
    }
    let parts: Vec<&str> = line.split_whitespace().collect();
    for part in parts {
        if let Some(value) = part.strip_prefix(key) {
            return usize::from_str(value).ok();
        }
    }
    None
}

fn parse_gate_line(line: &str, idx: usize) -> Result<Gate, ChipParseError> {
    let mut split = line.splitn(2, '=');
    let lhs = split.next().unwrap().trim();
    let rhs = split.next().ok_or_else(|| ChipParseError::InvalidGate(line.to_string()))?.trim();
    if lhs != format!("g{}", idx) {
        return Err(ChipParseError::InvalidGate(line.to_string()));
    }
    let op = parse_gate_op(rhs).ok_or_else(|| ChipParseError::InvalidGate(line.to_string()))?;
    Ok(Gate { op })
}

fn parse_gate_op(text: &str) -> Option<GateOp> {
    if let Some(rest) = text.strip_prefix("AND(") {
        let args = parse_ref_list(rest.strip_suffix(')')?)?;
        return Some(GateOp::And(args));
    }
    if let Some(rest) = text.strip_prefix("OR(") {
        let args = parse_ref_list(rest.strip_suffix(')')?)?;
        return Some(GateOp::Or(args));
    }
    if let Some(rest) = text.strip_prefix("NOT(") {
        let arg = parse_ref(rest.strip_suffix(')')?)?;
        return Some(GateOp::Not(arg));
    }
    if let Some(rest) = text.strip_prefix("THRESH(") {
        let inner = rest.strip_suffix(')')?;
        let mut parts = inner.splitn(2, ',');
        let k = usize::from_str(parts.next()?).ok()?;
        let args = parse_ref_list(parts.next()?)?;
        return Some(GateOp::Threshold { k, inputs: args });
    }
    None
}

fn parse_ref_list(text: &str) -> Option<Vec<Ref>> {
    let mut out = Vec::new();
    for part in text.split(',') {
        out.push(parse_ref(part.trim())?);
    }
    Some(out)
}

fn parse_ref(text: &str) -> Option<Ref> {
    if let Some(f) = text.strip_prefix('f') {
        return usize::from_str(f).ok().map(Ref::Feature);
    }
    if let Some(g) = text.strip_prefix('g') {
        return usize::from_str(g).ok().map(Ref::Gate);
    }
    None
}

fn parse_output_line(line: &str) -> Result<Ref, ChipParseError> {
    let mut split = line.splitn(2, '=');
    let lhs = split.next().unwrap().trim();
    let rhs = split.next().ok_or(ChipParseError::InvalidOutput)?.trim();
    if lhs != "OUTPUT" {
        return Err(ChipParseError::InvalidOutput);
    }
    parse_ref(rhs).ok_or(ChipParseError::InvalidOutput)
}

fn format_ref(r: &Ref) -> String {
    match r {
        Ref::Feature(i) => format!("f{}", i),
        Ref::Gate(i) => format!("g{}", i),
    }
}

fn join_refs(list: &[Ref]) -> String {
    let mut buf = String::new();
    for (idx, r) in list.iter().enumerate() {
        if idx > 0 {
            buf.push(',');
        }
        buf.push_str(&format_ref(r));
    }
    buf
}

fn validate_gate_refs(op: &GateOp, features: usize, gate_idx: usize) -> Result<(), ChipParseError> {
    let check = |r: &Ref| validate_ref(r, features, gate_idx);
    match op {
        GateOp::And(list) | GateOp::Or(list) => list.iter().try_for_each(check),
        GateOp::Not(r) => check(r),
        GateOp::Threshold { inputs, .. } => inputs.iter().try_for_each(check),
    }
}

fn validate_ref(r: &Ref, features: usize, gate_idx: usize) -> Result<(), ChipParseError> {
    match r {
        Ref::Feature(i) if *i < features => Ok(()),
        Ref::Gate(i) if *i < gate_idx => Ok(()),
        _ => Err(ChipParseError::RefOutOfRange),
    }
}

fn resolve_ref(r: &Ref, features: &[bool], gates: &[bool]) -> bool {
    match r {
        Ref::Feature(i) => features.get(*i).copied().unwrap_or(false),
        Ref::Gate(i) => gates.get(*i).copied().unwrap_or(false),
    }
}

fn random_ref<R: Rng>(rng: &mut R, features: usize, gate_idx: usize) -> Ref {
    let total = features + gate_idx;
    if total == 0 {
        return Ref::Feature(0);
    }
    let idx = rng.gen_range(0..total);
    ref_from_index(idx, features)
}

fn ref_from_index(idx: usize, features: usize) -> Ref {
    if idx < features {
        Ref::Feature(idx)
    } else {
        Ref::Gate(idx - features)
    }
}

fn biased_ref(idx: usize, features: usize, bias_primary: bool) -> Option<Ref> {
    if !bias_primary || features < 2 {
        return None;
    }
    if idx < 2 {
        return Some(Ref::Feature(idx));
    }
    None
}

fn biased_random_ref<R: Rng>(
    rng: &mut R,
    features: usize,
    gate_idx: usize,
    bias_primary: bool,
) -> Option<Ref> {
    if !bias_primary || features < 2 {
        return None;
    }
    if rng.gen_bool(0.5) {
        let idx = rng.gen_range(0..2);
        return Some(Ref::Feature(idx));
    }
    let total = features + gate_idx;
    if total == 0 {
        return Some(Ref::Feature(0));
    }
    let idx = rng.gen_range(0..total);
    Some(ref_from_index(idx, features))
}

pub fn random_chip<R: Rng>(
    rng: &mut R,
    features: usize,
    gates: usize,
    bias_primary_inputs: bool,
) -> Chip {
    let mut chip = Chip {
        features,
        gates: Vec::new(),
        output: Ref::Feature(0),
    };
    for idx in 0..gates {
        let op = GateOp::random(rng, features, idx, bias_primary_inputs);
        chip.gates.push(Gate { op });
    }
    if gates > 0 {
        chip.output = Ref::Gate(gates - 1);
    }
    chip
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn parse_canonical_idempotent() {
        let text = "CHIP v0\nFEATURES n=3\nGATES m=2\ng0 = AND(f0,f1)\ng1 = NOT(g0)\nOUTPUT = g1\n";
        let chip = Chip::parse(text).unwrap();
        let canon = chip.canonical_text();
        let again = Chip::parse(&canon).unwrap();
        assert_eq!(canon, again.canonical_text());
        assert_eq!(chip.hash(), again.hash());
    }

    #[test]
    fn hash_ignores_whitespace_variants() {
        let a = "  CHIP v0\nFEATURES  n=2\nGATES m=1\n g0 = OR( f0 , f1 )\nOUTPUT   = g0\n";
        let b = "CHIP v0\nFEATURES n=2\nGATES m=1\ng0 = OR(f0,f1)\nOUTPUT = g0\n";
        let ca = Chip::parse(a).unwrap();
        let cb = Chip::parse(b).unwrap();
        assert_eq!(ca.hash(), cb.hash());
    }

    #[test]
    fn biased_random_ref_can_pick_gates() {
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(42);
        let mut saw_gate = false;
        for _ in 0..50 {
            if let Some(Ref::Gate(_)) = biased_random_ref(&mut rng, 2, 3, true) {
                saw_gate = true;
                break;
            }
        }
        assert!(saw_gate, "biased_random_ref should occasionally pick gates");
    }

    #[test]
    fn eval_ops() {
        let chip = Chip::parse(
            "CHIP v0\nFEATURES n=2\nGATES m=2\ng0 = AND(f0,f1)\ng1 = THRESH(1,g0,f0)\nOUTPUT = g1\n",
        )
        .unwrap();
        assert_eq!(chip.eval(&[true, true]).unwrap(), true);
        assert_eq!(chip.eval(&[false, true]).unwrap(), false);
        assert_eq!(chip.eval(&[true, false]).unwrap(), true);
    }

    #[test]
    fn eval_threshold() {
        let chip = Chip::parse(
            "CHIP v0\nFEATURES n=3\nGATES m=1\ng0 = THRESH(2,f0,f1,f2)\nOUTPUT = g0\n",
        )
        .unwrap();
        assert!(chip.eval(&[true, true, false]).unwrap());
        assert!(!chip.eval(&[true, false, false]).unwrap());
    }
}
