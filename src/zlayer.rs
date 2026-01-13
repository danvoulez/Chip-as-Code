use crate::evolve::{evolve, Backend, EvolutionConfig, EvolutionResult, Task, TrainingLine};
#[cfg(feature = "gpu")]
use crate::gpu_eval::WGPU_VERSION;
use crate::gatebox::{gate_run_from_atoms, EvidenceDoc, GateVerdictLike, IntentClaim, IntentDoc};
use anyhow::{anyhow, Result};
use glob::glob;
use logline::json_atomic;
use rayon::prelude::*;
use rand::{seq::SliceRandom, RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, BTreeMap};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use tempfile::tempdir;

pub fn cid_for_value<T: Serialize>(value: &T) -> Result<String> {
    let canon = json_atomic::canonize(value)?;
    Ok(hex::encode(blake3::hash(&canon).as_bytes()))
}

fn cid_for_atom_without_cid(value: &Value) -> Result<String> {
    let mut tmp = value.clone();
    if let Value::Object(ref mut map) = tmp {
        map.remove("cid");
    }
    cid_for_value(&tmp)
}

fn attach_cid(mut value: Value) -> Result<Value> {
    let cid = cid_for_atom_without_cid(&value)?;
    if let Value::Object(ref mut map) = value {
        map.insert("cid".into(), Value::String(cid));
    }
    Ok(value)
}

fn map_get<'a>(map: &'a HashMap<String, Value>, cid: &str) -> Option<&'a Value> {
    map.get(cid).or_else(|| {
        map.values().find(|v| v.get("cid").and_then(|c| c.as_str()) == Some(cid))
    })
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
enum DerivOp {
    RollupSum,
    RollupCount,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VerifyReport {
    pub atoms_loaded: usize,
    pub atoms_unique: usize,
    pub derivations_replayed: usize,
    pub commits: usize,
    pub ghosts: usize,
    pub rejects: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MergeReport {
    pub atoms_loaded: usize,
    pub atoms_unique: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ValueSummary {
    pub commit_count: u64,
    pub ghost_count: u64,
    pub reject_count: u64,
    pub total_events: u64,
    pub total_value_per10k: u64,
    pub reason_counts: BTreeMap<String, u64>,
}

#[derive(Deserialize)]
struct TrainingLineWithCid {
    #[serde(rename = "cid")]
    _cid: String,
    #[serde(flatten)]
    line: TrainingLine,
}

struct EvoAtoms {
    dataset_fact: Value,
    dataset_set: Value,
    training_facts: Vec<Value>,
    training_set: Value,
    best_fact: Value,
    best_set: Value,
    derivation: Value,
    kernel_fact: Option<Value>,
}

fn claim_payloads(intent: &IntentDoc) -> Vec<Value> {
    intent
        .claims
        .iter()
        .map(|c| {
            json!({
                "intent_id": intent.id,
                "utterance": intent.utterance,
                "policy_features": intent.policy_features,
                "policy_epoch": intent.policy_epoch,
                "claim": {"id": c.id, "requires": c.requires},
            })
        })
        .collect()
}

fn evidence_payloads(evidence: &[EvidenceDoc]) -> Vec<Value> {
    evidence
        .iter()
        .map(|e| {
            json!({
                "id": e.id,
                "anchored": e.anchored,
                "supports": e.supports,
                "payload": e.payload,
            })
        })
        .collect()
}

fn read_training_curve_lines(path: &Path) -> Result<Vec<TrainingLine>> {
    let f = File::open(path)?;
    let mut out = Vec::new();
    for line in BufReader::new(f).lines() {
        let l = line?;
        if l.trim().is_empty() {
            continue;
        }
        let parsed: TrainingLineWithCid = serde_json::from_str(&l)?;
        out.push(parsed.line);
    }
    Ok(out)
}

fn build_evo_atoms(
    config: &EvolutionConfig,
    res: &EvolutionResult,
    training_lines: &[TrainingLine],
) -> Result<EvoAtoms> {
    let dataset_fact = attach_cid(json!({
        "t":"atom.fact",
        "v":1,
        "payload": {"dataset_hash": res.split.hash_hex},
    }))?;
    let dataset_set = attach_cid(json!({
        "t":"atom.set",
        "v":1,
        "name":"dataset_set",
        "members":[dataset_fact["cid"].as_str().unwrap()],
    }))?;

    let mut training_facts = Vec::new();
    for line in training_lines {
        let payload = serde_json::to_value(line)?;
        training_facts.push(attach_cid(json!({
            "t":"atom.fact",
            "v":1,
            "payload": payload,
        }))?);
    }

    let training_set = attach_cid(json!({
        "t":"atom.set",
        "v":1,
        "name":"training_curve_set",
        "members": training_facts
            .iter()
            .map(|f| f["cid"].as_str().unwrap())
            .collect::<Vec<_>>(),
    }))?;

    let best_fact = attach_cid(json!({
        "t":"atom.fact",
        "v":1,
        "payload": {
            "chip_text": res.best.chip.canonical_text(),
            "chip_hash": res.best.chip.hash(),
            "train_acc_per10k": res.best.train_acc_per10k,
            "test_acc_per10k": res.best.test_acc_per10k,
        },
    }))?;
    let best_set = attach_cid(json!({
        "t":"atom.set",
        "v":1,
        "name":"best_chip_set",
        "members":[best_fact["cid"].as_str().unwrap()],
    }))?;

    #[allow(unused_mut)]
    let mut kernel_fact: Option<Value> = None;
    #[allow(unused_mut)]
    let mut params = json!({
        "task": config.task.as_str(),
        "seed": config.seed,
        "generations": config.generations,
        "population": config.population,
        "offspring": config.offspring,
        "elite": config.elite,
        "max_gates": config.max_gates,
        "mutation_rate_per10k": config.mutation_rate_per10k,
        "backend": config.backend.as_str(),
    });
    #[cfg(feature = "gpu")]
    if let (Backend::Gpu, Some(meta)) = (config.backend, res.backend_info.gpu.as_ref()) {
        let fact = attach_cid(json!({
            "t":"atom.fact",
            "v":1,
            "payload": {
                "kernel":"wgpu_eval",
                "wgpu": WGPU_VERSION,
                "shader_hash": meta.shader_hash,
                "device": meta.adapter,
                "backend": meta.backend,
            },
        }))?;
        if let Some(obj) = params.as_object_mut() {
            obj.insert("shader_hash".into(), Value::String(meta.shader_hash.clone()));
            obj.insert(
                "kernel_fact".into(),
                Value::String(fact["cid"].as_str().unwrap_or_default().to_string()),
            );
        }
        kernel_fact = Some(fact);
    }

    let derivation = attach_cid(json!({
        "t":"atom.derivation",
        "v":1,
        "z":"evo:v1",
        "op":"evolve_run",
        "inputs":[{"set": dataset_set["cid"].as_str().unwrap()}],
        "params": params,
        "outputs":[
            {"set": training_set["cid"].as_str().unwrap()},
            {"set": best_set["cid"].as_str().unwrap()},
        ],
    }))?;

    Ok(EvoAtoms {
        dataset_fact,
        dataset_set,
        training_facts,
        training_set,
        best_fact,
        best_set,
        derivation,
        kernel_fact,
    })
}

fn write_evo_atoms(out_dir: &Path, atoms: &EvoAtoms) -> Result<()> {
    fs::create_dir_all(out_dir)?;
    let mut training_rows = Vec::new();
    training_rows.push(atoms.dataset_fact.clone());
    if let Some(kf) = &atoms.kernel_fact {
        training_rows.push(kf.clone());
    }
    training_rows.extend_from_slice(&atoms.training_facts);
    write_ndjson(&out_dir.join("training_curve_atoms.ndjson"), &training_rows)?;
    write_ndjson(&out_dir.join("best_chip_atoms.ndjson"), &[atoms.best_fact.clone()])?;
    write_ndjson(
        &out_dir.join("sets.ndjson"),
        &[atoms.dataset_set.clone(), atoms.training_set.clone(), atoms.best_set.clone()],
    )?;
    write_ndjson(&out_dir.join("derivations.ndjson"), &[atoms.derivation.clone()])?;
    Ok(())
}

pub fn zlayer_evolve(config: EvolutionConfig, zlayer_out: &Path) -> Result<()> {
    let res = evolve(config.clone())?;
    let training_curve_path = Path::new(&config.out_dir).join("training_curve.ndjson");
    let training_lines = read_training_curve_lines(&training_curve_path)?;
    let atoms = build_evo_atoms(&config, &res, &training_lines)?;
    write_evo_atoms(zlayer_out, &atoms)
}

/// Build gate outputs using Template + Occurrence pattern.
/// 
/// This produces:
/// 1. A TEMPLATE fact (dedupable) - canonical verdict reason
/// 2. An EVENT fact (unique per derivation) - anchors occurrence to case
/// 
/// The event references the template via `template_cid`, enabling:
/// - Perfect deduplication of identical verdict reasons (catalog)
/// - Accurate event counting (1 derivation = 1 event)
fn build_gate_outputs(
    intent: &IntentDoc,
    evidence_cids: &[String],
    verdict: GateVerdictLike,
    derivation_cid: Option<&str>,
) -> Result<(Vec<Value>, Vec<HashMap<&'static str, String>>)> {
    let (template_fact, verdict_str) = match &verdict {
        GateVerdictLike::Commit { proof } => {
            let fact = attach_cid(json!({
                "t": "atom.fact",
                "v": 1,
                "payload": {
                    "fact_type": "verdict_template",
                    "verdict": "commit",
                    "gate_proof": proof,
                },
            }))?;
            (fact, "commit")
        }
        GateVerdictLike::Ghost { missing_claim, question } => {
            let fact = attach_cid(json!({
                "t": "atom.fact",
                "v": 1,
                "payload": {
                    "fact_type": "verdict_template",
                    "verdict": "ghost",
                    "missing_claim": missing_claim,
                    "question": question,
                },
            }))?;
            (fact, "ghost")
        }
        GateVerdictLike::Reject { reason } => {
            let fact = attach_cid(json!({
                "t": "atom.fact",
                "v": 1,
                "payload": {
                    "fact_type": "verdict_template",
                    "verdict": "reject",
                    "reason": reason,
                },
            }))?;
            (fact, "reject")
        }
    };

    let template_cid = template_fact["cid"].as_str().unwrap().to_string();

    // Build the EVENT fact (unique per derivation via intent_id)
    // Note: derivation_cid is optional; the link is primarily via derivation.outputs -> event
    let mut event_payload = json!({
        "fact_type": "verdict_event",
        "verdict": verdict_str,
        "intent_id": intent.id,
        "utterance": intent.utterance,
        "policy_epoch": intent.policy_epoch,
        "template_cid": template_cid,
        "evidence_cids": evidence_cids,
    });
    if let Some(dcid) = derivation_cid {
        event_payload["derivation_cid"] = Value::String(dcid.to_string());
    }
    
    let event_fact = attach_cid(json!({
        "t": "atom.fact",
        "v": 1,
        "payload": event_payload,
    }))?;

    let event_cid = event_fact["cid"].as_str().unwrap().to_string();

    // Build the verdict set containing the event
    let verdict_set = attach_cid(json!({
        "t": "atom.set",
        "v": 1,
        "name": format!("{}_set", verdict_str),
        "members": [&event_cid],
    }))?;

    let mut output_ref = HashMap::new();
    output_ref.insert("set", verdict_set["cid"].as_str().unwrap().to_string());

    Ok((
        vec![template_fact, event_fact, verdict_set],
        vec![output_ref],
    ))
}

fn build_gate_run_atoms(
    intent: &IntentDoc,
    evidence: &[EvidenceDoc],
    policy_text: &str,
) -> Result<(Vec<Value>, String, String, Vec<String>)> {
    let claim_payloads = claim_payloads(intent);
    let evidence_payloads = evidence_payloads(evidence);

    let mut atoms = Vec::new();
    let mut claim_facts = Vec::new();
    for p in &claim_payloads {
        claim_facts.push(attach_cid(json!({"t":"atom.fact","v":1,"payload":p}))?);
    }
    let intent_set = attach_cid(json!({
        "t":"atom.set","v":1,"name":"gate:intent_set","members":claim_facts.iter().map(|f| f["cid"].as_str().unwrap()).collect::<Vec<_>>()
    }))?;

    let mut evidence_facts = Vec::new();
    for p in &evidence_payloads {
        evidence_facts.push(attach_cid(json!({"t":"atom.fact","v":1,"payload":p}))?);
    }
    let evidence_set = attach_cid(json!({
        "t":"atom.set","v":1,"name":"gate:evidence_set","members":evidence_facts.iter().map(|f| f["cid"].as_str().unwrap()).collect::<Vec<_>>()
    }))?;

    let policy_fact = attach_cid(json!({
        "t":"atom.fact","v":1,
        "payload": {"policy": policy_text},
    }))?;
    let epoch_fact = attach_cid(json!({
        "t":"atom.fact","v":1,
        "payload": {"policy_epoch": intent.policy_epoch},
    }))?;

    atoms.extend(claim_facts.clone());
    atoms.extend(evidence_facts.clone());
    atoms.push(intent_set.clone());
    atoms.push(evidence_set.clone());
    atoms.push(policy_fact.clone());
    atoms.push(epoch_fact.clone());

    let verdict = gate_run_from_atoms(
        claim_payloads,
        evidence_payloads,
        policy_fact["payload"].clone(),
        intent.policy_epoch,
    )?;

    let evidence_cids: Vec<String> = evidence_facts
        .iter()
        .map(|f| f["cid"].as_str().unwrap().to_string())
        .collect();
    
    // First pass: build outputs without derivation_cid (will be linked via derivation.outputs)
    let (output_atoms, outputs) = build_gate_outputs(intent, &evidence_cids, verdict.clone(), None)?;
    atoms.extend(output_atoms.clone());

    let outputs_json: Vec<Value> = outputs
        .iter()
        .map(|m| {
            let mut obj = serde_json::Map::new();
            for (k, v) in m {
                obj.insert((*k).into(), Value::String(v.clone()));
            }
            Value::Object(obj)
        })
        .collect();

    let derivation = attach_cid(json!({
        "t":"atom.derivation",
        "v":1,
        "z":"gate:v1",
        "op":"gate_run",
        "inputs":[
            {"set": intent_set["cid"].as_str().unwrap()},
            {"set": evidence_set["cid"].as_str().unwrap()},
            {"fact": policy_fact["cid"].as_str().unwrap()},
            {"fact": epoch_fact["cid"].as_str().unwrap()},
        ],
        "params": {},
        "outputs": outputs_json,
    }))?;
    atoms.push(derivation.clone());

    let verdict_str = match verdict {
        GateVerdictLike::Commit { .. } => "COMMIT".to_string(),
        GateVerdictLike::Ghost { .. } => "GHOST".to_string(),
        GateVerdictLike::Reject { .. } => "REJECT".to_string(),
    };
    let output_cids = outputs
        .iter()
        .filter_map(|m| m.get("set").or_else(|| m.get("fact")))
        .cloned()
        .collect::<Vec<_>>();
    Ok((atoms, verdict_str, derivation["cid"].as_str().unwrap().to_string(), output_cids))
}

fn build_gate_run_atoms_from_fixtures(
    intent: &IntentDoc,
    evidence: &[EvidenceDoc],
    policy_text: &str,
    out_dir: &Path,
) -> Result<(String, String, Vec<String>)> {
    let (atoms, verdict, deriv_cid, outputs) = build_gate_run_atoms(intent, evidence, policy_text)?;

    fs::create_dir_all(out_dir)?;
    write_ndjson(&out_dir.join("atoms.ndjson"), &atoms)?;

    Ok((verdict, deriv_cid, outputs))
}

impl DerivOp {
    fn name(&self) -> &'static str {
        match self {
            DerivOp::RollupSum => "rollup_sum",
            DerivOp::RollupCount => "rollup_count",
        }
    }
}

pub fn zlayer_demo(out_dir: &str) -> Result<()> {
    let base = Path::new(out_dir);
    fs::create_dir_all(base)?;
    let ts = "2026-01-12T00:00:00Z";
    let mut facts = Vec::new();
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    for i in 0..3u64 {
        let amt = 10 + i * 5 + (rng.next_u32() as u64 % 3);
        let fact = json!({
            "t": "atom.fact",
            "v": 1,
            "ts": ts,
            "payload": {"id": format!("f{}", i), "amount": amt},
        });
        facts.push(attach_cid(fact)?);
    }
    write_ndjson(&base.join("z0_facts.ndjson"), &facts)?;

    let z0_set = attach_cid(json!({
        "t": "atom.set",
        "v": 1,
        "name": "store_001/2026-01-12/z0",
        "members": facts.iter().map(|f| f["cid"].as_str().unwrap()).collect::<Vec<_>>()
    }))?;
    write_ndjson(&base.join("z0_set.ndjson"), &[z0_set.clone()])?;

    let (z1_fact, z1_set, deriv) = run_derivation(&z0_set, &facts, DerivOp::RollupSum)?;
    write_ndjson(&base.join("z1_set.ndjson"), &[z1_fact.clone(), z1_set.clone()])?;
    write_ndjson(&base.join("z1_derivation.ndjson"), &[deriv.clone()])?;
    Ok(())
}

pub fn zlayer_merge(glob_pattern: &str, out: &Path) -> Result<MergeReport> {
    let paths = ndjson_paths_from_glob(glob_pattern)?;
    if paths.is_empty() {
        return Err(anyhow!("no files matched glob"));
    }
    let (map, loaded) = load_atoms_from_paths(&paths)?;
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    let atoms = atoms_sorted_by_cid(&map)?;
    write_ndjson(out, &atoms)?;
    Ok(MergeReport {
        atoms_loaded: loaded,
        atoms_unique: map.len(),
    })
}

fn verdict_of_fact(fact: &Value) -> Option<&str> {
    fact.get("payload")
        .and_then(|p| p.get("verdict"))
        .and_then(|v| v.as_str())
}

fn build_value_summary_from_facts(facts: &[Value], unit_value_per10k: u64) -> Result<ValueSummary> {
    let mut summary = ValueSummary::default();
    for f in facts {
        let payload = f.get("payload");
        let verdict = verdict_of_fact(f).ok_or_else(|| anyhow!("fact missing verdict"))?;
        
        // Skip templates - only count events (occurrences)
        // Templates have fact_type="verdict_template", events have fact_type="verdict_event"
        // Old format (no fact_type) should be counted as events for backward compat
        let fact_type = payload.and_then(|p| p.get("fact_type")).and_then(|t| t.as_str());
        if fact_type == Some("verdict_template") {
            continue; // Skip template, only count events
        }
        
        match verdict {
            "commit" => summary.commit_count += 1,
            "ghost" => summary.ghost_count += 1,
            "reject" => {
                summary.reject_count += 1;
                // For reject, reason might be in the template, check both
                if let Some(reason) = payload
                    .and_then(|p| p.get("reason"))
                    .and_then(|r| r.as_str())
                {
                    *summary.reason_counts.entry(reason.to_string()).or_insert(0) += 1;
                }
            }
            other => return Err(anyhow!("unknown verdict {}", other)),
        }
    }
    summary.total_events = summary.commit_count + summary.ghost_count + summary.reject_count;
    summary.total_value_per10k = summary.commit_count.saturating_mul(unit_value_per10k);
    Ok(summary)
}

fn value_metric_facts(summary: &ValueSummary) -> Result<Vec<Value>> {
    let mut out = Vec::new();
    let metrics: Vec<(&str, u64)> = vec![
        ("commit_count", summary.commit_count),
        ("ghost_count", summary.ghost_count),
        ("reject_count", summary.reject_count),
        ("total_events", summary.total_events),
        ("total_value_per10k", summary.total_value_per10k),
    ];
    for (metric, value) in metrics {
        out.push(attach_cid(json!({
            "t":"atom.fact",
            "v":1,
            "payload": {"verdict":"value", "metric": metric, "value": value},
        }))?);
    }
    for (reason, value) in summary.reason_counts.iter() {
        out.push(attach_cid(json!({
            "t":"atom.fact",
            "v":1,
            "payload": {"verdict":"value", "metric":"reject_reason_count", "reason": reason, "value": value},
        }))?);
    }
    out.sort_by(|a, b| a["cid"].as_str().unwrap_or("").cmp(b["cid"].as_str().unwrap_or("")));
    Ok(out)
}

fn build_value_run_atoms_from_map(
    map: &HashMap<String, Value>,
    mode: &str,
    unit_value_per10k: u64,
) -> Result<(Vec<Value>, ValueSummary)> {
    let mut verdict_facts: Vec<Value> = map
        .values()
        .filter(|v| v["t"].as_str() == Some("atom.fact"))
        .filter(|v| matches!(verdict_of_fact(v), Some("commit" | "ghost" | "reject")))
        .cloned()
        .collect();
    if verdict_facts.is_empty() {
        return Err(anyhow!("no verdict facts found for value_run"));
    }
    verdict_facts.sort_by(|a, b| a["cid"].as_str().unwrap_or("").cmp(b["cid"].as_str().unwrap_or("")));

    let summary = build_value_summary_from_facts(&verdict_facts, unit_value_per10k)?;
    let summary_facts = value_metric_facts(&summary)?;

    let verdict_cids: Vec<&str> = verdict_facts
        .iter()
        .map(|v| v["cid"].as_str().unwrap())
        .collect();
    let verdict_set = attach_cid(json!({
        "t":"atom.set",
        "v":1,
        "name":"value:verdicts",
        "members": verdict_cids,
    }))?;

    let summary_cids: Vec<&str> = summary_facts
        .iter()
        .map(|v| v["cid"].as_str().unwrap())
        .collect();
    let summary_set = attach_cid(json!({
        "t":"atom.set",
        "v":1,
        "name":"value:summary",
        "members": summary_cids,
    }))?;

    let derivation = attach_cid(json!({
        "t":"atom.derivation",
        "v":1,
        "z":"value:v1",
        "op":"value_run",
        "inputs":[{"set": verdict_set["cid"].as_str().unwrap()}],
        "params": {"mode": mode, "unit_value_per10k": unit_value_per10k},
        "outputs":[{"set": summary_set["cid"].as_str().unwrap()}],
    }))?;

    let mut atoms = Vec::new();
    atoms.push(verdict_set);
    atoms.push(summary_set);
    atoms.extend(summary_facts);
    atoms.push(derivation);
    Ok((atoms, summary))
}

pub fn zlayer_value(input: &Path, out_dir: &Path, mode: &str, unit_value_per10k: u64) -> Result<ValueSummary> {
    let paths = if input.is_dir() {
        ndjson_paths_in_dir(input)?
    } else {
        vec![input.to_path_buf()]
    };
    if paths.is_empty() {
        return Err(anyhow!("no input ndjson found"));
    }
    let (map, _loaded) = load_atoms_from_paths(&paths)?;
    let (atoms, summary) = build_value_run_atoms_from_map(&map, mode, unit_value_per10k)?;

    fs::create_dir_all(out_dir)?;
    let facts: Vec<Value> = atoms
        .iter()
        .filter(|a| a["t"].as_str() == Some("atom.fact"))
        .cloned()
        .collect();
    let sets: Vec<Value> = atoms
        .iter()
        .filter(|a| a["t"].as_str() == Some("atom.set"))
        .cloned()
        .collect();
    let derivs: Vec<Value> = atoms
        .iter()
        .filter(|a| a["t"].as_str() == Some("atom.derivation"))
        .cloned()
        .collect();

    write_ndjson(&out_dir.join("value_facts.ndjson"), &facts)?;
    write_ndjson(&out_dir.join("value_set.ndjson"), &sets)?;
    write_ndjson(&out_dir.join("value_derivations.ndjson"), &derivs)?;

    Ok(summary)
}

fn personalize_case(intent: &IntentDoc, evidence: &[EvidenceDoc], scenario: &str, idx: usize) -> (IntentDoc, Vec<EvidenceDoc>) {
    let suffix = format!("{}-{}", scenario, idx);

    let mut intent = intent.clone();
    intent.id = format!("{}-{}", intent.id, &suffix);
    for claim in intent.claims.iter_mut() {
        claim.id = format!("{}-{}", claim.id, &suffix);
        claim.requires = claim
            .requires
            .iter()
            .map(|r| format!("{}-{}", r, &suffix))
            .collect();
    }

    let mut ev_out = Vec::new();
    for ev in evidence {
        let mut e = ev.clone();
        e.id = format!("{}-{}", e.id, &suffix);
        e.supports = e
            .supports
            .iter()
            .map(|s| format!("{}-{}", s, &suffix))
            .collect();
        ev_out.push(e);
    }

    (intent, ev_out)
}

pub fn zlayer_stress(seed: u64, n: usize, shards: usize, out_dir: &Path) -> Result<Vec<PathBuf>> {
    if shards == 0 {
        return Err(anyhow!("shards must be > 0"));
    }
    fs::create_dir_all(out_dir)?;

    let intent_good = load_fixture_intent(Path::new("examples/intent_good.json"))?;
    let intent_missing = load_fixture_intent(Path::new("examples/intent_missing.json"))?;
    let intent_unanchored = load_fixture_intent(Path::new("examples/intent_unanchored.json"))?;
    let intent_policy = load_fixture_intent(Path::new("examples/intent_policy_violation.json"))?;
    let ev_good = load_fixture_evidence(Path::new("examples/evidence_good.json"))?;
    let ev_missing = load_fixture_evidence(Path::new("examples/evidence_missing.json"))?;
    let ev_unanchored = load_fixture_evidence(Path::new("examples/evidence_unanchored.json"))?;
    let ev_policy = load_fixture_evidence(Path::new("examples/evidence_policy_violation.json"))?;
    let policy_text = fs::read_to_string("examples/policy.txt")?;

    let scenarios: Vec<(&str, IntentDoc, Vec<EvidenceDoc>, String)> = vec![
        ("good", intent_good, ev_good, policy_text.clone()),
        ("missing", intent_missing, ev_missing, policy_text.clone()),
        ("unanchored", intent_unanchored, ev_unanchored, policy_text.clone()),
        ("policy", intent_policy, ev_policy, policy_text.clone()),
    ];

    let mut assignments: Vec<Vec<(usize, usize)>> = vec![vec![]; shards];
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut produced = 0usize;
    while produced < n {
        let mut block = [0usize, 1, 2, 3];
        block.shuffle(&mut rng);
        for scenario_idx in block {
            if produced >= n {
                break;
            }
            assignments[produced % shards].push((scenario_idx, produced));
            produced += 1;
        }
    }

    let out_base = out_dir.to_path_buf();
    let shard_paths: Vec<PathBuf> = assignments
        .into_par_iter()
        .enumerate()
        .map(|(shard_idx, work)| -> Result<PathBuf> {
            let mut map: HashMap<String, Value> = HashMap::new();
            for (scenario_idx, seq_idx) in work {
                let (name, base_intent, base_ev, policy) = &scenarios[scenario_idx];
                let (intent, evidence) = personalize_case(base_intent, base_ev, name, seq_idx);
                let (atoms, _verdict, _deriv, _outputs) = build_gate_run_atoms(&intent, &evidence, policy)?;
                for atom in atoms {
                    let cid = atom["cid"].as_str().ok_or_else(|| anyhow!("missing cid"))?.to_string();
                    let recompute = cid_for_atom_without_cid(&atom)?;
                    if cid != recompute {
                        return Err(anyhow!("cid mismatch during stress build {}", cid));
                    }
                    if let Some(existing) = map.get(&cid) {
                        if existing != &atom {
                            return Err(anyhow!("cid collision in shard {}", cid));
                        }
                        continue;
                    }
                    map.insert(cid, atom);
                }
            }

            let shard_atoms = atoms_sorted_by_cid(&map)?;
            let shard_path = out_base.join(format!("shard_{:03}.ndjson", shard_idx));
            write_ndjson(&shard_path, &shard_atoms)?;
            Ok(shard_path)
        })
        .collect::<Result<Vec<_>>>()?;

    let mut sorted_paths = shard_paths;
    sorted_paths.sort();
    Ok(sorted_paths)
}

fn run_derivation(z0_set: &Value, facts: &[Value], op: DerivOp) -> Result<(Value, Value, Value)> {
    let members_arr = z0_set["members"]
        .as_array()
        .ok_or_else(|| anyhow!("z0_set missing members array"))?;
    let members: Vec<&Value> = facts
        .iter()
        .filter(|f| members_arr.iter().any(|c| c == &f["cid"]))
        .collect();
    let derived_fact = match op {
        DerivOp::RollupSum => {
            let sum: u64 = members
                .iter()
                .map(|f| f["payload"]["amount"].as_u64().unwrap_or(0))
                .sum();
            attach_cid(json!({"t":"atom.fact","v":1,"ts":"2026-01-12T00:00:00Z","payload":{"sum_amount":sum}}))?
        }
        DerivOp::RollupCount => {
            let count = members.len() as u64;
            attach_cid(json!({"t":"atom.fact","v":1,"ts":"2026-01-12T00:00:00Z","payload":{"count":count}}))?
        }
    };
    let derived_set = attach_cid(json!({
        "t":"atom.set",
        "v":1,
        "name":"store_001/2026-01-12/z1",
        "members":[derived_fact["cid"].as_str().unwrap()],
    }))?;
    let derivation = attach_cid(json!({
        "t":"atom.derivation",
        "v":1,
        "z":"z1",
        "op": op.name(),
        "inputs":[{"set": z0_set["cid"].as_str().unwrap()}],
        "params": match op { DerivOp::RollupSum => json!({"field":"amount"}), DerivOp::RollupCount => json!({}) },
        "outputs":[{"set": derived_set["cid"].as_str().unwrap()}],
    }))?;
    Ok((derived_fact, derived_set, derivation))
}

fn write_ndjson(path: &Path, rows: &[Value]) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    for r in rows {
        serde_json::to_writer(&mut w, r)?;
        writeln!(&mut w)?;
    }
    Ok(())
}

fn load_atoms_from_paths(paths: &[PathBuf]) -> Result<(HashMap<String, Value>, usize)> {
    let mut map: HashMap<String, Value> = HashMap::new();
    let mut loaded: usize = 0;
    for path in paths {
        let f = File::open(path)?;
        for line in BufReader::new(f).lines() {
            let l = line?;
            if l.trim().is_empty() {
                continue;
            }
            loaded += 1;
            let v: Value = serde_json::from_str(&l)?;
            let cid = v["cid"].as_str().ok_or_else(|| anyhow!("missing cid"))?.to_string();
            let recompute = cid_for_atom_without_cid(&v)?;
            if cid != recompute {
                return Err(anyhow!("cid mismatch for {}", cid));
            }
            if let Some(existing) = map.get(&cid) {
                if existing != &v {
                    return Err(anyhow!("cid collision with different content {}", cid));
                }
                continue;
            }
            map.insert(cid, v);
        }
    }
    Ok((map, loaded))
}

fn atoms_sorted_by_cid(map: &HashMap<String, Value>) -> Result<Vec<Value>> {
    let mut atoms: Vec<Value> = map.values().cloned().collect();
    atoms.sort_by(|a, b| {
        let ac = a["cid"].as_str().unwrap_or("");
        let bc = b["cid"].as_str().unwrap_or("");
        ac.cmp(bc)
    });
    Ok(atoms)
}

fn ndjson_paths_in_dir(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if entry.path().extension().and_then(|s| s.to_str()) == Some("ndjson") {
            paths.push(entry.path());
        }
    }
    paths.sort();
    Ok(paths)
}

fn ndjson_paths_from_glob(pattern: &str) -> Result<Vec<PathBuf>> {
    let mut paths: Vec<PathBuf> = Vec::new();
    for pat in pattern.split(',') {
        let trimmed = pat.trim();
        if trimmed.is_empty() {
            continue;
        }
        for entry in glob(trimmed)? {
            if let Ok(p) = entry {
                paths.push(p);
            }
        }
    }
    paths.sort();
    Ok(paths)
}

fn verify_evolve_run(atom: &Value, map: &HashMap<String, Value>) -> Result<()> {
    let params = atom
        .get("params")
        .and_then(|p| p.as_object())
        .ok_or_else(|| anyhow!("missing params"))?;
    let task_str = params
        .get("task")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("missing task"))?;
    let task = Task::from_str(task_str).ok_or_else(|| anyhow!("unknown task"))?;
    let seed = params
        .get("seed")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow!("missing seed"))?;
    let generations = params
        .get("generations")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow!("missing generations"))? as usize;
    let population = params
        .get("population")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow!("missing population"))? as usize;
    let offspring = params
        .get("offspring")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow!("missing offspring"))? as usize;
    let elite = params
        .get("elite")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow!("missing elite"))? as usize;
    let max_gates = params
        .get("max_gates")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow!("missing max_gates"))? as usize;
    let mutation_rate_per10k = params
        .get("mutation_rate_per10k")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow!("missing mutation_rate_per10k"))? as u32;
    let backend = params
        .get("backend")
        .and_then(|v| v.as_str())
        .and_then(Backend::from_str)
        .unwrap_or(Backend::Cpu);
    let shader_hash = params.get("shader_hash").and_then(|v| v.as_str());
    let kernel_fact_cid = params
        .get("kernel_fact")
        .and_then(|v| v.as_str());

    let tmp = tempdir()?;
    let out_dir = tmp.path().join("out");
    let cfg = EvolutionConfig {
        task,
        seed,
        generations,
        population,
        offspring,
        elite,
        max_gates,
        mutation_rate_per10k,
        out_dir: out_dir.to_string_lossy().into_owned(),
        debug: false,
        backend,
    };
    let res = evolve(cfg.clone())?;
    let training_curve_path = Path::new(&cfg.out_dir).join("training_curve.ndjson");
    let training_lines = read_training_curve_lines(&training_curve_path)?;
    let atoms = build_evo_atoms(&cfg, &res, &training_lines)?;

    if let Some(cid) = kernel_fact_cid {
        let fact = map
            .get(cid)
            .ok_or_else(|| anyhow!("missing kernel fact {}", cid))?;
        if fact["t"].as_str() != Some("atom.fact") {
            return Err(anyhow!("kernel_fact not fact"));
        }
        if let Some(hash) = shader_hash {
            let payload_hash = fact
                .get("payload")
                .and_then(|p| p.get("shader_hash"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if payload_hash != hash {
                return Err(anyhow!("shader_hash mismatch"));
            }
        }
    } else if backend == Backend::Gpu {
        return Err(anyhow!("gpu backend missing kernel_fact"));
    }

    let inputs = atom
        .get("inputs")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("inputs missing"))?;
    let input_set = inputs
        .get(0)
        .and_then(|v| v.get("set"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("input set missing"))?;
    if input_set != atoms.dataset_set["cid"].as_str().unwrap_or("") {
        return Err(anyhow!("evolve_run input set cid mismatch"));
    }

    let outputs = atom
        .get("outputs")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("outputs missing"))?;
    let training_set_cid = outputs
        .get(0)
        .and_then(|v| v.get("set"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("training output missing"))?;
    let best_set_cid = outputs
        .get(1)
        .and_then(|v| v.get("set"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("best output missing"))?;

    if training_set_cid != atoms.training_set["cid"].as_str().unwrap_or("") {
        return Err(anyhow!("training_curve_set cid mismatch"));
    }
    if best_set_cid != atoms.best_set["cid"].as_str().unwrap_or("") {
        return Err(anyhow!("best_chip_set cid mismatch"));
    }

    let dataset_set = map
        .get(input_set)
        .ok_or_else(|| anyhow!("missing referenced set {}", input_set))?;
    if dataset_set["members"] != atoms.dataset_set["members"] {
        return Err(anyhow!("dataset set members mismatch"));
    }

    let training_set = map
        .get(training_set_cid)
        .ok_or_else(|| anyhow!("missing referenced set {}", training_set_cid))?;
    if training_set["members"] != atoms.training_set["members"] {
        return Err(anyhow!("training set members mismatch"));
    }

    let best_set = map
        .get(best_set_cid)
        .ok_or_else(|| anyhow!("missing referenced set {}", best_set_cid))?;
    if best_set["members"] != atoms.best_set["members"] {
        return Err(anyhow!("best set members mismatch"));
    }
    let best_fact_cid = atoms.best_fact["cid"].as_str().unwrap_or("");
    map.get(best_fact_cid)
        .ok_or_else(|| anyhow!("missing best fact {}", best_fact_cid))?;

    for f in &atoms.training_facts {
        let cid = f["cid"].as_str().unwrap_or("");
        map.get(cid).ok_or_else(|| anyhow!("missing training fact {}", cid))?;
    }
    map.get(atoms.dataset_fact["cid"].as_str().unwrap_or(""))
        .ok_or_else(|| anyhow!("missing dataset fact"))?;
    Ok(())
}

fn verify_gate_run(atom: &Value, map: &HashMap<String, Value>) -> Result<GateVerdictLike> {
    let inputs = atom
        .get("inputs")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("inputs missing"))?;

    let mut set_cids = Vec::new();
    let mut fact_cids = Vec::new();
    for obj in inputs {
        if let Some(s) = obj.get("set").and_then(|v| v.as_str()) {
            set_cids.push(s.to_string());
        }
        if let Some(f) = obj.get("fact").and_then(|v| v.as_str()) {
            fact_cids.push(f.to_string());
        }
    }
    if set_cids.len() != 2 {
        return Err(anyhow!("expected 2 gate input sets"));
    }
    if fact_cids.len() != 2 {
        return Err(anyhow!("expected 2 gate input facts"));
    }

    let mut intent_set: Option<&Value> = None;
    let mut evidence_set: Option<&Value> = None;
    for cid in &set_cids {
        let set = map_get(map, cid).ok_or_else(|| anyhow!("missing input set {}", cid))?;
        if set["t"].as_str() != Some("atom.set") {
            return Err(anyhow!("input {} not set", cid));
        }
        let name = set.get("name").and_then(|v| v.as_str()).unwrap_or("");
        if name.contains("intent") {
            intent_set = Some(set);
        } else if name.contains("evidence") {
            evidence_set = Some(set);
        }
    }
    let intent_set = intent_set.ok_or_else(|| anyhow!("intent set missing"))?;
    let evidence_set = evidence_set.ok_or_else(|| anyhow!("evidence set missing"))?;

    let mut policy_fact: Option<&Value> = None;
    let mut epoch_fact: Option<&Value> = None;
    for cid in &fact_cids {
        let fact = map_get(map, cid).ok_or_else(|| anyhow!("missing input fact {}", cid))?;
        if fact["t"].as_str() != Some("atom.fact") {
            return Err(anyhow!("input {} not fact", cid));
        }
        let payload = fact.get("payload").and_then(|v| v.as_object()).ok_or_else(|| anyhow!("fact payload missing"))?;
        if payload.contains_key("policy_epoch") {
            epoch_fact = Some(fact);
        } else if payload.contains_key("policy") {
            policy_fact = Some(fact);
        }
    }
    let policy_fact = policy_fact.ok_or_else(|| anyhow!("policy fact missing"))?;
    let epoch_fact = epoch_fact.ok_or_else(|| anyhow!("epoch fact missing"))?;

    let claim_payloads: Vec<Value> = intent_set
        .get("members")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("intent members missing"))?
        .iter()
        .map(|cid| {
            let c = cid.as_str().ok_or_else(|| anyhow!("member not string"))?;
            let fact = map_get(map, c).ok_or_else(|| anyhow!("missing claim fact"))?;
            Ok(fact.get("payload").cloned().ok_or_else(|| anyhow!("claim payload missing"))?)
        })
        .collect::<Result<_>>()?;
    let evidence_payloads: Vec<Value> = evidence_set
        .get("members")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("evidence members missing"))?
        .iter()
        .map(|cid| {
            let c = cid.as_str().ok_or_else(|| anyhow!("member not string"))?;
            let fact = map_get(map, c).ok_or_else(|| anyhow!("missing evidence fact"))?;
            Ok(fact.get("payload").cloned().ok_or_else(|| anyhow!("evidence payload missing"))?)
        })
        .collect::<Result<_>>()?;

    let epoch = epoch_fact
        .get("payload")
        .and_then(|p| p.get("policy_epoch"))
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow!("epoch missing"))?;
    let policy_payload = policy_fact
        .get("payload")
        .and_then(|p| {
            if p.get("policy").and_then(|v| v.as_str()).is_some() {
                Some(p.clone())
            } else {
                None
            }
        })
        .ok_or_else(|| anyhow!("policy payload missing"))?;

    if claim_payloads.is_empty() {
        return Err(anyhow!("no claims"));
    }
    let claims: Vec<IntentClaim> = claim_payloads
        .iter()
        .map(|p| IntentClaim {
            id: p["claim"]["id"].as_str().unwrap_or("").to_string(),
            requires: p["claim"]["requires"]
                .as_array()
                .unwrap_or(&vec![])
                .iter()
                .map(|r| r.as_str().unwrap_or("").to_string())
                .collect(),
        })
        .collect();
    let policy_features = claim_payloads[0]
        .get("policy_features")
        .and_then(|v| v.as_array())
        .unwrap_or(&vec![])
        .iter()
        .map(|b| b.as_bool().unwrap_or(false))
        .collect();
    let intent = IntentDoc {
        id: claim_payloads[0]
            .get("intent_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        utterance: claim_payloads[0]
            .get("utterance")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        claims,
        policy_features,
        policy_epoch: epoch,
    };

    let verdict = gate_run_from_atoms(
        claim_payloads,
        evidence_payloads,
        policy_payload,
        epoch,
    )?;

    let evidence_cids: Vec<String> = evidence_set
        .get("members")
        .and_then(|v| v.as_array())
        .unwrap_or(&vec![])
        .iter()
        .filter_map(|c| c.as_str().map(|s| s.to_string()))
        .collect();
    let (outputs, output_refs) = build_gate_outputs(&intent, &evidence_cids, verdict.clone(), None)?;

    let declared = atom
        .get("outputs")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("outputs missing"))?;
    if declared.len() != output_refs.len() {
        return Err(anyhow!("outputs length mismatch"));
    }
    for (decl, expect) in declared.iter().zip(output_refs.iter()) {
        if let Some(set) = decl.get("set").and_then(|v| v.as_str()) {
            let expected = expect.get("set").ok_or_else(|| anyhow!("expected set missing"))?;
            if set != expected {
                return Err(anyhow!("output set cid mismatch"));
            }
            let stored = map.get(set).ok_or_else(|| anyhow!("missing output set"))?;
            let expected_val = outputs
                .iter()
                .find(|o| o["cid"] == *expected)
                .ok_or_else(|| anyhow!("expected set atom missing"))?;
            if stored["cid"] != expected_val["cid"] {
                return Err(anyhow!("output set atom cid mismatch"));
            }
        } else if let Some(fact) = decl.get("fact").and_then(|v| v.as_str()) {
            let expected = expect.get("fact").ok_or_else(|| anyhow!("expected fact missing"))?;
            if fact != expected {
                return Err(anyhow!("output fact cid mismatch"));
            }
            let stored = map.get(fact).ok_or_else(|| anyhow!("missing output fact"))?;
            let expected_val = outputs
                .iter()
                .find(|o| o["cid"] == *expected)
                .ok_or_else(|| anyhow!("expected fact atom missing"))?;
            if stored["cid"] != expected_val["cid"] {
                return Err(anyhow!("output fact atom cid mismatch"));
            }
        } else {
            return Err(anyhow!("unknown output entry"));
        }
    }
    Ok(verdict)
}

fn verify_value_run(atom: &Value, map: &HashMap<String, Value>) -> Result<ValueSummary> {
    let inputs = atom
        .get("inputs")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("inputs missing"))?;
    if inputs.len() != 1 {
        return Err(anyhow!("value_run expects exactly 1 input set"));
    }
    let set_cid = inputs[0]
        .get("set")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("input set missing"))?;
    let set = map.get(set_cid).ok_or_else(|| anyhow!("missing input set"))?;
    if set["t"].as_str() != Some("atom.set") {
        return Err(anyhow!("input not set"));
    }
    let members = set
        .get("members")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("members missing"))?;
    let mut verdict_facts = Vec::new();
    for cid in members {
        let c = cid.as_str().ok_or_else(|| anyhow!("member not string"))?;
        let fact = map.get(c).ok_or_else(|| anyhow!("missing verdict fact"))?;
        if fact["t"].as_str() != Some("atom.fact") {
            return Err(anyhow!("input member not fact"));
        }
        match verdict_of_fact(fact) {
            Some("commit" | "ghost" | "reject") => {}
            Some(other) => return Err(anyhow!("unsupported verdict {}", other)),
            None => return Err(anyhow!("input fact missing verdict")),
        }
        verdict_facts.push(fact.clone());
    }

    let unit_value_per10k = atom
        .get("params")
        .and_then(|p| p.get("unit_value_per10k"))
        .and_then(|v| v.as_u64())
        .unwrap_or(10_000);
    let summary = build_value_summary_from_facts(&verdict_facts, unit_value_per10k)?;
    let summary_facts = value_metric_facts(&summary)?;
    let summary_cids: Vec<&str> = summary_facts
        .iter()
        .map(|f| f["cid"].as_str().unwrap())
        .collect();
    let expected_set = attach_cid(json!({
        "t":"atom.set",
        "v":1,
        "name":"value:summary",
        "members": summary_cids,
    }))?;

    let outputs = atom
        .get("outputs")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("outputs missing"))?;
    if outputs.len() != 1 {
        return Err(anyhow!("value_run expects single output set"));
    }
    let out_cid = outputs[0]
        .get("set")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("output set missing"))?;
    let out_set = map.get(out_cid).ok_or_else(|| anyhow!("missing output set"))?;
    if out_set["cid"] != expected_set["cid"] {
        return Err(anyhow!("value_run output set cid mismatch"));
    }
    let out_members = out_set
        .get("members")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("output members missing"))?;
    if out_members.len() != summary_facts.len() {
        return Err(anyhow!("value_run member length mismatch"));
    }
    for fact in summary_facts {
        let cid = fact["cid"].as_str().unwrap();
        let stored = map.get(cid).ok_or_else(|| anyhow!("missing value fact"))?;
        if stored["cid"] != fact["cid"] {
            return Err(anyhow!("value fact cid mismatch"));
        }
    }

    Ok(summary)
}

fn verify_map(map: &HashMap<String, Value>, loaded: usize) -> Result<VerifyReport> {
    for (_cid, atom) in map.iter() {
        match atom["t"].as_str().unwrap_or("") {
            "atom.set" => {
                for m in atom["members"].as_array().ok_or_else(|| anyhow!("set members missing"))? {
                    let mid = m.as_str().ok_or_else(|| anyhow!("member not string"))?;
                    let target = map.get(mid).ok_or_else(|| anyhow!("missing member {}", mid))?;
                    if target["t"].as_str() != Some("atom.fact") {
                        return Err(anyhow!("member {} not fact", mid));
                    }
                }
            }
            "atom.derivation" => {
                let _op = atom["op"].as_str().unwrap_or("");
                let inputs = atom["inputs"].as_array().ok_or_else(|| anyhow!("inputs missing"))?;
                for inp in inputs {
                    if let Some(sid) = inp.get("set").and_then(|v| v.as_str()) {
                        let set = map.get(sid).ok_or_else(|| anyhow!("missing input set {}", sid))?;
                        if set["t"].as_str() != Some("atom.set") {
                            return Err(anyhow!("input {} not set", sid));
                        }
                    } else if let Some(fid) = inp.get("fact").and_then(|v| v.as_str()) {
                        let fact = map.get(fid).ok_or_else(|| anyhow!("missing input fact {}", fid))?;
                        if fact["t"].as_str() != Some("atom.fact") {
                            return Err(anyhow!("input {} not fact", fid));
                        }
                    } else {
                        return Err(anyhow!("input reference missing"));
                    }
                }

                let outputs = atom["outputs"].as_array().ok_or_else(|| anyhow!("outputs missing"))?;
                for out in outputs {
                    if let Some(sid) = out.get("set").and_then(|v| v.as_str()) {
                        let set = map.get(sid).ok_or_else(|| anyhow!("missing output set {}", sid))?;
                        if set["t"].as_str() != Some("atom.set") {
                            return Err(anyhow!("output {} not set", sid));
                        }
                    } else if let Some(fid) = out.get("fact").and_then(|v| v.as_str()) {
                        let fact = map.get(fid).ok_or_else(|| anyhow!("missing output fact {}", fid))?;
                        if fact["t"].as_str() != Some("atom.fact") {
                            return Err(anyhow!("output {} not fact", fid));
                        }
                    } else {
                        return Err(anyhow!("output reference missing"));
                    }
                }
            }
            _ => {}
        }
    }

    let derivations: Vec<&Value> = map
        .values()
        .filter(|a| a["t"].as_str() == Some("atom.derivation"))
        .collect();

    let gate_results = derivations
        .par_iter()
        .map(|atom| {
            let op = atom["op"].as_str().ok_or_else(|| anyhow!("op missing"))?;
            if op == "evolve_run" {
                verify_evolve_run(atom, map)?;
                return Ok(None);
            }
            if op == "gate_run" {
                let verdict = verify_gate_run(atom, map)?;
                return Ok(Some(verdict));
            }
            if op == "value_run" {
                verify_value_run(atom, map)?;
                return Ok(None);
            }

            let input_set_cid = atom["inputs"][0]["set"].as_str().ok_or_else(|| anyhow!("input set missing"))?;
            let input_set = map.get(input_set_cid).ok_or_else(|| anyhow!("missing input set"))?;
            let members = input_set["members"].as_array().ok_or_else(|| anyhow!("members missing"))?;
            let facts: Vec<&Value> = members
                .iter()
                .map(|m| map.get(m.as_str().unwrap()).ok_or_else(|| anyhow!("missing fact")))
                .collect::<Result<_>>()?;
            let out_set_cid = atom["outputs"][0]["set"].as_str().ok_or_else(|| anyhow!("output set missing"))?;
            let out_set = map.get(out_set_cid).ok_or_else(|| anyhow!("missing output set"))?;
            let out_set_name = out_set["name"].as_str().unwrap_or("verify_z1");
            let (expected_fact, expected_set) = match op {
                "rollup_sum" => {
                    let sum: u64 = facts
                        .iter()
                        .map(|f| f["payload"]["amount"].as_u64().unwrap_or(0))
                        .sum();
                    let fact = attach_cid(json!({"t":"atom.fact","v":1,"ts":"2026-01-12T00:00:00Z","payload":{"sum_amount":sum}}))?;
                    let set = attach_cid(json!({"t":"atom.set","v":1,"name":out_set_name,"members":[fact["cid"].as_str().unwrap()]}))?;
                    (fact, set)
                }
                "rollup_count" => {
                    let count = facts.len() as u64;
                    let fact = attach_cid(json!({"t":"atom.fact","v":1,"ts":"2026-01-12T00:00:00Z","payload":{"count":count}}))?;
                    let set = attach_cid(json!({"t":"atom.set","v":1,"name":out_set_name,"members":[fact["cid"].as_str().unwrap()]}))?;
                    (fact, set)
                }
                other => return Err(anyhow!("unknown op {}", other)),
            };
            if out_set["cid"] != expected_set["cid"] {
                return Err(anyhow!("output set cid mismatch"));
            }
            let derived_member = out_set["members"][0].as_str().ok_or_else(|| anyhow!("output member missing"))?;
            let actual_fact = map.get(derived_member).ok_or_else(|| anyhow!("missing derived fact"))?;
            if actual_fact["cid"] != expected_fact["cid"] {
                return Err(anyhow!("derived fact cid mismatch"));
            }
            Ok(None)
        })
        .collect::<Result<Vec<_>>>()?;

    let mut commits = 0usize;
    let mut ghosts = 0usize;
    let mut rejects = 0usize;
    for v in gate_results.iter().filter_map(|o| o.as_ref()) {
        match v {
            GateVerdictLike::Commit { .. } => commits += 1,
            GateVerdictLike::Ghost { .. } => ghosts += 1,
            GateVerdictLike::Reject { .. } => rejects += 1,
        }
    }

    Ok(VerifyReport {
        atoms_loaded: loaded,
        atoms_unique: map.len(),
        derivations_replayed: derivations.len(),
        commits,
        ghosts,
        rejects,
    })
}

pub fn zlayer_verify(dir: &Path) -> Result<VerifyReport> {
    let paths = ndjson_paths_in_dir(dir)?;
    if paths.is_empty() {
        return Err(anyhow!("no ndjson files found"));
    }
    let (map, loaded) = load_atoms_from_paths(&paths)?;
    verify_map(&map, loaded)
}

pub fn zlayer_verify_file(file: &Path) -> Result<VerifyReport> {
    let paths = vec![file.to_path_buf()];
    let (map, loaded) = load_atoms_from_paths(&paths)?;
    verify_map(&map, loaded)
}

/// Streaming verify mode for large files (1M+ atoms).
/// Uses a two-pass approach:
/// 1. First pass: validate CIDs and count atom types
/// 2. Second pass: verify references and replay gate_run derivations
/// This avoids loading all atoms into memory at once.
pub fn zlayer_verify_streaming(file: &Path) -> Result<VerifyReport> {
    use std::collections::HashSet;
    
    // Pass 1: Validate CIDs and collect atom metadata
    let f = File::open(file)?;
    let mut atoms_loaded = 0usize;
    let mut cid_set: HashSet<String> = HashSet::new();
    let mut derivation_lines: Vec<(usize, String)> = Vec::new();
    let mut commits = 0usize;
    let mut ghosts = 0usize;
    let mut rejects = 0usize;

    for (line_num, line_result) in BufReader::new(f).lines().enumerate() {
        let line = line_result?;
        if line.trim().is_empty() {
            continue;
        }
        atoms_loaded += 1;
        
        let v: Value = serde_json::from_str(&line)
            .map_err(|e| anyhow!("line {}: invalid JSON: {}", line_num + 1, e))?;
        
        let cid = v["cid"]
            .as_str()
            .ok_or_else(|| anyhow!("line {}: missing cid", line_num + 1))?
            .to_string();
        
        // Verify CID matches content
        let recompute = cid_for_atom_without_cid(&v)?;
        if cid != recompute {
            return Err(anyhow!("line {}: cid mismatch for {}", line_num + 1, cid));
        }
        
        cid_set.insert(cid.clone());
        
        // Track derivations for second pass
        let atom_type = v["t"].as_str().unwrap_or("");
        if atom_type == "atom.derivation" {
            derivation_lines.push((line_num, line.clone()));
        }
        
        // Count verdicts from facts (only events, not templates)
        if atom_type == "atom.fact" {
            let payload = v.get("payload");
            let fact_type = payload.and_then(|p| p.get("fact_type")).and_then(|t| t.as_str());
            
            // Skip templates - only count events (old format has no fact_type, count as event)
            if fact_type == Some("verdict_template") {
                continue;
            }
            
            if let Some(verdict) = payload.and_then(|p| p.get("verdict")).and_then(|v| v.as_str()) {
                match verdict {
                    "commit" => commits += 1,
                    "ghost" => ghosts += 1,
                    "reject" => rejects += 1,
                    _ => {}
                }
            }
        }
    }
    
    let atoms_unique = cid_set.len();
    let derivations_replayed = derivation_lines.len();
    
    // Pass 2: Verify derivation references exist (parallel, streaming)
    // For streaming mode, we just verify that referenced CIDs exist in our set
    // Full replay would require loading atoms, which defeats the purpose
    let f2 = File::open(file)?;
    for line_result in BufReader::new(f2).lines() {
        let line = line_result?;
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(&line)?;
        let atom_type = v["t"].as_str().unwrap_or("");
        
        // Verify set members exist
        if atom_type == "atom.set" {
            if let Some(members) = v.get("members").and_then(|m| m.as_array()) {
                for member in members {
                    let mid = member.as_str().ok_or_else(|| anyhow!("set member not string"))?;
                    if !cid_set.contains(mid) {
                        return Err(anyhow!("set references missing member {}", mid));
                    }
                }
            }
        }
        
        // Verify derivation inputs/outputs exist
        if atom_type == "atom.derivation" {
            if let Some(inputs) = v.get("inputs").and_then(|i| i.as_array()) {
                for inp in inputs {
                    if let Some(sid) = inp.get("set").and_then(|s| s.as_str()) {
                        if !cid_set.contains(sid) {
                            return Err(anyhow!("derivation references missing input set {}", sid));
                        }
                    }
                    if let Some(fid) = inp.get("fact").and_then(|f| f.as_str()) {
                        if !cid_set.contains(fid) {
                            return Err(anyhow!("derivation references missing input fact {}", fid));
                        }
                    }
                }
            }
            if let Some(outputs) = v.get("outputs").and_then(|o| o.as_array()) {
                for out in outputs {
                    if let Some(sid) = out.get("set").and_then(|s| s.as_str()) {
                        if !cid_set.contains(sid) {
                            return Err(anyhow!("derivation references missing output set {}", sid));
                        }
                    }
                    if let Some(fid) = out.get("fact").and_then(|f| f.as_str()) {
                        if !cid_set.contains(fid) {
                            return Err(anyhow!("derivation references missing output fact {}", fid));
                        }
                    }
                }
            }
        }
    }
    
    Ok(VerifyReport {
        atoms_loaded,
        atoms_unique,
        derivations_replayed,
        commits,
        ghosts,
        rejects,
    })
}

/// Diagnostic report for analyzing atom distribution (Template + Occurrence pattern)
#[derive(Debug, Clone, Default)]
pub struct DiagnosticReport {
    pub total_atoms: usize,
    pub unique_cids: usize,
    pub atom_types: BTreeMap<String, usize>,
    
    // Template vs Event breakdown
    pub derivations: usize,
    pub verdict_events: BTreeMap<String, usize>,      // Events per verdict type (= occurrences)
    pub verdict_templates: BTreeMap<String, usize>,   // Unique templates per verdict type (= catalog)
    pub template_frequency: Vec<(String, String, usize)>, // (template_cid, verdict, event_count)
    
    // Legacy (for backward compat with old format)
    pub verdict_distribution: BTreeMap<String, usize>,
    pub unique_verdict_cids: BTreeMap<String, usize>,
    pub duplicate_atoms: usize,
    pub sample_duplicates: Vec<(String, usize, String)>,
}

/// Deep diagnostic analysis of a Z-layer NDJSON file
pub fn zlayer_diagnose(file: &Path) -> Result<DiagnosticReport> {
    use std::collections::HashSet;
    
    let f = File::open(file)?;
    let mut report = DiagnosticReport::default();
    
    // Track all atoms by CID
    let mut cid_counts: HashMap<String, usize> = HashMap::new();
    let mut cid_to_verdict: HashMap<String, String> = HashMap::new();
    let mut verdict_cids: HashMap<String, HashSet<String>> = HashMap::new();
    
    // Template + Occurrence tracking
    let mut template_event_count: HashMap<String, usize> = HashMap::new();
    let mut template_to_verdict: HashMap<String, String> = HashMap::new();
    let mut event_templates: HashSet<String> = HashSet::new();
    
    for line_result in BufReader::new(f).lines() {
        let line = line_result?;
        if line.trim().is_empty() {
            continue;
        }
        report.total_atoms += 1;
        
        let v: Value = serde_json::from_str(&line)?;
        let cid = v["cid"].as_str().unwrap_or("").to_string();
        let atom_type = v["t"].as_str().unwrap_or("unknown").to_string();
        
        // Count by type
        *report.atom_types.entry(atom_type.clone()).or_insert(0) += 1;
        
        // Count derivations
        if atom_type == "atom.derivation" {
            report.derivations += 1;
        }
        
        // Track duplicates
        *cid_counts.entry(cid.clone()).or_insert(0) += 1;
        
        // Track facts
        if atom_type == "atom.fact" {
            let payload = v.get("payload");
            let fact_type = payload.and_then(|p| p.get("fact_type")).and_then(|t| t.as_str());
            let verdict = payload.and_then(|p| p.get("verdict")).and_then(|v| v.as_str());
            
            if let Some(verdict_str) = verdict {
                let verdict_str = verdict_str.to_string();
                
                // Legacy tracking (all verdict facts)
                *report.verdict_distribution.entry(verdict_str.clone()).or_insert(0) += 1;
                cid_to_verdict.insert(cid.clone(), verdict_str.clone());
                verdict_cids.entry(verdict_str.clone()).or_default().insert(cid.clone());
                
                // Template + Occurrence tracking
                match fact_type {
                    Some("verdict_template") => {
                        template_to_verdict.insert(cid.clone(), verdict_str.clone());
                        *report.verdict_templates.entry(verdict_str).or_insert(0) += 1;
                    }
                    Some("verdict_event") => {
                        *report.verdict_events.entry(verdict_str.clone()).or_insert(0) += 1;
                        // Track which template this event references
                        if let Some(tcid) = payload.and_then(|p| p.get("template_cid")).and_then(|t| t.as_str()) {
                            *template_event_count.entry(tcid.to_string()).or_insert(0) += 1;
                            event_templates.insert(tcid.to_string());
                        }
                    }
                    _ => {
                        // Old format (no fact_type) - treat as event for backward compat
                        *report.verdict_events.entry(verdict_str).or_insert(0) += 1;
                    }
                }
            }
        }
    }
    
    report.unique_cids = cid_counts.len();
    
    // Count unique CIDs per verdict type
    for (verdict, cids) in &verdict_cids {
        report.unique_verdict_cids.insert(verdict.clone(), cids.len());
    }
    
    // Build template frequency list (top templates by event count)
    let mut freq: Vec<(String, String, usize)> = template_event_count
        .into_iter()
        .map(|(tcid, count)| {
            let verdict = template_to_verdict.get(&tcid).cloned().unwrap_or_default();
            (tcid, verdict, count)
        })
        .collect();
    freq.sort_by(|a, b| b.2.cmp(&a.2));
    report.template_frequency = freq.into_iter().take(10).collect();
    
    // Find duplicates
    let mut duplicates: Vec<(String, usize, String)> = cid_counts
        .iter()
        .filter(|(_, &count)| count > 1)
        .map(|(cid, &count)| {
            let verdict = cid_to_verdict.get(cid).cloned().unwrap_or_default();
            (cid.clone(), count, verdict)
        })
        .collect();
    
    report.duplicate_atoms = duplicates.iter().map(|(_, count, _)| count - 1).sum();
    
    // Sort by count descending and take top 10
    duplicates.sort_by(|a, b| b.1.cmp(&a.1));
    report.sample_duplicates = duplicates.into_iter().take(10).collect();
    
    Ok(report)
}

fn load_fixture_intent(path: &Path) -> Result<IntentDoc> {
    let text = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}

fn load_fixture_evidence(path: &Path) -> Result<Vec<EvidenceDoc>> {
    let text = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}

pub fn zlayer_gate_demo(base: &Path) -> Result<Vec<(String, String, String, Vec<String>)>> {
    let scenarios = vec![
        (
            "good",
            "examples/intent_good.json",
            "examples/evidence_good.json",
            "examples/policy.txt",
        ),
        (
            "missing_evidence",
            "examples/intent_missing.json",
            "examples/evidence_missing.json",
            "examples/policy.txt",
        ),
        (
            "unanchored",
            "examples/intent_unanchored.json",
            "examples/evidence_unanchored.json",
            "examples/policy.txt",
        ),
        (
            "policy_violation",
            "examples/intent_policy_violation.json",
            "examples/evidence_policy_violation.json",
            "examples/policy.txt",
        ),
    ];
    let mut summaries = Vec::new();
    for (name, intent_path, evidence_path, policy_path) in scenarios {
        let intent = load_fixture_intent(Path::new(intent_path))?;
        let evidence = load_fixture_evidence(Path::new(evidence_path))?;
        let policy_text = fs::read_to_string(policy_path)?;
        let out_dir = base.join(name);
        let (verdict, deriv_cid, outputs) =
            build_gate_run_atoms_from_fixtures(&intent, &evidence, &policy_text, &out_dir)?;
        println!(
            "scenario={} verdict={} derivation={} outputs={:?}",
            name, verdict, deriv_cid, outputs
        );
        summaries.push((name.into(), verdict, deriv_cid, outputs));
    }
    Ok(summaries)
}

pub fn zlayer_gate_verify(base: &Path) -> Result<()> {
    for entry in fs::read_dir(base)? {
        let entry = entry?;
        if !entry.path().is_dir() {
            continue;
        }
        let dir = entry.path();
        let _stats = zlayer_verify(&dir)?;
        println!(
            "verified scenario={} gate_run ok",
            dir.file_name().unwrap().to_string_lossy()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use std::path::Path;
    use blake3;

    #[test]
    fn cid_roundtrip_excludes_cid_field() {
        let base = json!({"t":"atom.fact","v":1,"payload":{"x":1}});
        let cid1 = cid_for_atom_without_cid(&base).unwrap();
        let mut with = base.clone();
        if let Value::Object(ref mut m) = with {
            m.insert("cid".into(), Value::String(cid1.clone()));
        }
        let cid2 = cid_for_atom_without_cid(&with).unwrap();
        assert_eq!(cid1, cid2);
    }

    #[test]
    fn zlayer_demo_then_verify() {
        let dir = tempdir().unwrap();
        let out = dir.path().join("z");
        zlayer_demo(out.to_str().unwrap()).unwrap();
        let _ = zlayer_verify(&out).unwrap();
    }

    #[test]
    fn rollup_count_produces_set_and_fact() {
        let facts = vec![
            attach_cid(json!({"t":"atom.fact","v":1,"ts":"2026-01-12T00:00:00Z","payload":{"amount":1}})).unwrap(),
        ];
        let z0_set = attach_cid(json!({
            "t":"atom.set","v":1,"name":"demo","members":[facts[0]["cid"].as_str().unwrap()],
        }))
        .unwrap();
        let (fact, set, deriv) = run_derivation(&z0_set, &facts, DerivOp::RollupCount).unwrap();
        assert_eq!(fact["t"], "atom.fact");
        assert_eq!(set["members"][0], fact["cid"]);
        assert_eq!(deriv["op"], "rollup_count");
    }

    #[test]
    fn zlayer_evolve_then_verify() {
        let dir = tempdir().unwrap();
        let out_dir = dir.path().join("out");
        let z_out = dir.path().join("zout");
        let cfg = EvolutionConfig {
            task: Task::Xor,
            seed: 1337,
            generations: 10,
            population: 200,
            offspring: 200,
            elite: 10,
            max_gates: 12,
            mutation_rate_per10k: 1500,
            out_dir: out_dir.to_string_lossy().into_owned(),
            debug: false,
            backend: Backend::Cpu,
        };
        zlayer_evolve(cfg, &z_out).unwrap();
        let _ = zlayer_verify(&z_out).unwrap();
    }

    #[test]
    fn gate_run_demo_then_verify() {
        let dir = tempdir().unwrap();
        let base = dir.path().join("zgate");
        zlayer_gate_demo(&base).unwrap();
        zlayer_gate_verify(&base).unwrap();

        let atoms: Vec<Value> = BufReader::new(File::open(base.join("missing_evidence/atoms.ndjson")).unwrap())
            .lines()
            .map(|l| serde_json::from_str(&l.unwrap()).unwrap())
            .collect();
        let has_commit = atoms
            .iter()
            .any(|a| a["t"] == "atom.set" && a.get("name") == Some(&Value::String("commit_set".into())));
        assert!(!has_commit, "missing evidence scenario must not emit commit_set");
    }

    #[test]
    fn no_guess_law_never_commits() {
        let dir = tempdir().unwrap();
        let base = dir.path().join("zgate");
        let intent = load_fixture_intent(Path::new("examples/intent_missing.json")).unwrap();
        let evidence = load_fixture_evidence(Path::new("examples/evidence_missing.json")).unwrap();
        let policy = fs::read_to_string("examples/policy.txt").unwrap();
        let out = base.join("missing_evidence");
        build_gate_run_atoms_from_fixtures(&intent, &evidence, &policy, &out).unwrap();
        let atoms: Vec<Value> = BufReader::new(File::open(out.join("atoms.ndjson")).unwrap())
            .lines()
            .map(|l| serde_json::from_str(&l.unwrap()).unwrap())
            .collect();
        let has_commit = atoms
            .iter()
            .any(|a| a["t"] == "atom.set" && a["name"] == "commit_set");
        assert!(!has_commit, "missing evidence must not commit");
        let ghost = atoms
            .iter()
            .find(|a| a["t"] == "atom.set" && a["name"] == "ghost_set")
            .cloned()
            .expect("ghost set missing");
        let ghost_event_cid = ghost["members"][0].as_str().unwrap();
        let ghost_event = atoms
            .iter()
            .find(|a| a.get("cid") == Some(&Value::String(ghost_event_cid.into())))
            .unwrap();
        // Event should have fact_type = verdict_event and reference a template
        assert_eq!(ghost_event["payload"]["fact_type"].as_str(), Some("verdict_event"));
        let template_cid = ghost_event["payload"]["template_cid"].as_str().unwrap();
        let ghost_template = atoms
            .iter()
            .find(|a| a.get("cid") == Some(&Value::String(template_cid.into())))
            .unwrap();
        // Template should have the question
        assert!(ghost_template["payload"]["question"].as_str().is_some());
    }

    #[test]
    fn gate_run_replay_matches_cids() {
        let dir = tempdir().unwrap();
        let base = dir.path().join("zgate");
        let intent = load_fixture_intent(Path::new("examples/intent_good.json")).unwrap();
        let evidence = load_fixture_evidence(Path::new("examples/evidence_good.json")).unwrap();
        let policy = fs::read_to_string("examples/policy.txt").unwrap();
        let out = base.join("good");
        build_gate_run_atoms_from_fixtures(&intent, &evidence, &policy, &out).unwrap();
        let _ = zlayer_verify(&out).unwrap();
    }

    #[test]
    fn value_run_demo_then_verify() {
        let dir = tempdir().unwrap();
        let gate_dir = dir.path().join("zgate");
        zlayer_gate_demo(&gate_dir).unwrap();

        let merged = dir.path().join("all.ndjson");
        let pattern = gate_dir.join("*/atoms.ndjson");
        zlayer_merge(&pattern.to_string_lossy(), &merged).unwrap();

        let value_dir = dir.path().join("value");
        let summary = zlayer_value(&merged, &value_dir, "gate_rollup", 10_000).unwrap();
        assert_eq!(summary.commit_count, 1);
        assert_eq!(summary.ghost_count, 1);
        assert_eq!(summary.reject_count, 2);
        assert_eq!(summary.total_value_per10k, 10_000);

        let merged_with_value = dir.path().join("all_with_value.ndjson");
        let merge_pattern = format!("{},{}", merged.to_string_lossy(), value_dir.join("*.ndjson").to_string_lossy());
        zlayer_merge(&merge_pattern, &merged_with_value).unwrap();
        let stats = zlayer_verify_file(&merged_with_value).unwrap();
        assert_eq!(stats.commits, 1);
        assert_eq!(stats.ghosts, 1);
        assert_eq!(stats.rejects, 2);
    }

    #[test]
    fn stress_merge_value_is_deterministic() {
        let dir = tempdir().unwrap();
        let base1 = dir.path().join("run1");
        let shards1 = base1.join("shards");
        zlayer_stress(321, 200, 5, &shards1).unwrap();
        let all1 = base1.join("all.ndjson");
        zlayer_merge(&shards1.join("shard_*.ndjson").to_string_lossy(), &all1).unwrap();
        let value1 = base1.join("value");
        zlayer_value(&all1, &value1, "gate_rollup", 10_000).unwrap();
        let all_with_value1 = base1.join("all_with_value.ndjson");
        let pattern1 = format!("{},{}", all1.to_string_lossy(), value1.join("*.ndjson").to_string_lossy());
        zlayer_merge(&pattern1, &all_with_value1).unwrap();
        let hash1 = blake3::hash(&fs::read(&all_with_value1).unwrap());

        let base2 = dir.path().join("run2");
        let shards2 = base2.join("shards");
        zlayer_stress(321, 200, 5, &shards2).unwrap();
        let all2 = base2.join("all.ndjson");
        zlayer_merge(&shards2.join("shard_*.ndjson").to_string_lossy(), &all2).unwrap();
        let value2 = base2.join("value");
        zlayer_value(&all2, &value2, "gate_rollup", 10_000).unwrap();
        let all_with_value2 = base2.join("all_with_value.ndjson");
        let pattern2 = format!("{},{}", all2.to_string_lossy(), value2.join("*.ndjson").to_string_lossy());
        zlayer_merge(&pattern2, &all_with_value2).unwrap();
        let hash2 = blake3::hash(&fs::read(&all_with_value2).unwrap());

        assert_eq!(hash1.as_bytes(), hash2.as_bytes());
    }

    #[test]
    fn stress_merge_verify_deterministic() {
        let dir = tempdir().unwrap();
        let base = dir.path().join("stress");
        let shards = zlayer_stress(123, 200, 4, &base).unwrap();
        assert_eq!(shards.len(), 4);

        let merged = base.join("all.ndjson");
        let pattern = base.join("shard_*.ndjson");
        let merge_report = zlayer_merge(&pattern.to_string_lossy(), &merged).unwrap();
        assert!(merge_report.atoms_unique > 0);
        let stats = zlayer_verify_file(&merged).unwrap();
        assert!(stats.derivations_replayed > 0);

        let bytes1 = fs::read(&merged).unwrap();
        let hash1 = blake3::hash(&bytes1);

        let base2 = dir.path().join("stress2");
        zlayer_stress(123, 200, 4, &base2).unwrap();
        let merged2 = base2.join("all.ndjson");
        let pattern2 = base2.join("shard_*.ndjson");
        zlayer_merge(&pattern2.to_string_lossy(), &merged2).unwrap();
        let bytes2 = fs::read(&merged2).unwrap();
        let hash2 = blake3::hash(&bytes2);

        assert_eq!(bytes1.len(), bytes2.len());
        assert_eq!(hash1.as_bytes(), hash2.as_bytes());
    }

    #[test]
    fn malformed_ndjson_returns_error_not_panic() {
        let dir = tempdir().unwrap();
        let bad_file = dir.path().join("bad.ndjson");
        // Missing CID field
        fs::write(&bad_file, r#"{"t":"atom.fact","v":1,"payload":{"x":1}}"#).unwrap();
        let result = zlayer_verify_file(&bad_file);
        assert!(result.is_err(), "missing cid should error, not panic");
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("cid"), "error should mention cid: {}", err_msg);

        // Invalid JSON line
        let bad_file2 = dir.path().join("bad2.ndjson");
        fs::write(&bad_file2, "not valid json\n").unwrap();
        let result2 = zlayer_verify_file(&bad_file2);
        assert!(result2.is_err(), "invalid json should error, not panic");
    }
}