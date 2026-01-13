use crate::chip_ir::Chip;
use anyhow::{anyhow, Result};
use logline::compiler::{compile, CompileCtx};
use logline::gate::{decide, Consent, Decision, GateOutput, PolicyCtx};
use logline::json_atomic;
use serde_json::json;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::path::Path;
use ubl_ledger::{LedgerEntry, SimpleLedgerReader, SimpleLedgerWriter};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentClaim {
    pub id: String,
    pub requires: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentDoc {
    pub id: String,
    pub utterance: String,
    pub claims: Vec<IntentClaim>,
    pub policy_features: Vec<bool>,
    pub policy_epoch: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceDoc {
    pub id: String,
    pub anchored: bool,
    pub supports: Vec<String>,
    #[serde(default)]
    pub payload: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GateVerdict {
    Commit {
        cid_hex: String,
        chip_hash: String,
        proof: String,
    },
    Ghost {
        question: String,
    },
    Reject {
        reason: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum GateVerdictLike {
    Commit { proof: String },
    Ghost { missing_claim: String, question: String },
    Reject { reason: String },
}

pub fn run_gate(
    intent_path: &str,
    evidence_path: &str,
    policy_path: &str,
    ledger_path: &str,
) -> Result<GateVerdict> {
    let intent_text = fs::read_to_string(intent_path)?;
    let evidence_text = fs::read_to_string(evidence_path)?;
    let policy_text = fs::read_to_string(policy_path)?;
    let intent: IntentDoc = serde_json::from_str(&intent_text)?;
    let evidence: Vec<EvidenceDoc> = serde_json::from_str(&evidence_text)?;
    let policy_chip = Chip::parse(&policy_text)?;

    if !evidence.iter().all(|e| e.anchored) {
        return Ok(GateVerdict::Reject {
            reason: "evidence not anchored".into(),
        });
    }

    if let Some(question) = missing_evidence_question(&intent, &evidence) {
        let payload = ledger_payload(
            "ghost",
            &intent,
            &policy_chip,
            &[],
            &question,
            None,
        )?;
        append_ledger(payload, ledger_path)?;
        return Ok(GateVerdict::Ghost { question });
    }

    if intent.policy_features.len() != policy_chip.features {
        return Ok(GateVerdict::Reject {
            reason: "policy feature mismatch".into(),
        });
    }

    let policy_allow = policy_chip.eval(&intent.policy_features)?;
    if !policy_allow {
        return Ok(GateVerdict::Reject {
            reason: "policy denied".into(),
        });
    }

    let gate = run_tdln_gate(&intent, policy_allow)?;

    let evidence_cids: Vec<String> = evidence
        .iter()
        .map(|e| cid_for_value(&serde_json::to_value(e).unwrap()))
        .collect();

    let payload = ledger_payload(
        "commit",
        &intent,
        &policy_chip,
        &evidence_cids,
        "",
        Some(&gate),
    )?;
    let cid_hex = append_ledger(payload, ledger_path)?;
    Ok(GateVerdict::Commit {
        cid_hex,
        chip_hash: policy_chip.hash(),
        proof: hex::encode(gate.proof_ref),
    })
}

fn run_tdln_gate(intent: &IntentDoc, allow: bool) -> Result<GateOutput> {
    let ctx = CompileCtx {
        rule_set: format!("epoch-{}", intent.policy_epoch),
    };
    let compiled = compile(&intent.utterance, &ctx)?;
    let gctx = PolicyCtx {
        allow_freeform: allow,
    };
    let consent = Consent { accepted: allow };
    let out = decide(&compiled, &consent, &gctx)?;
    if out.decision == Decision::Deny {
        return Err(anyhow!("gate denied"));
    }
    Ok(out)
}

fn missing_evidence_question(intent: &IntentDoc, evidence: &[EvidenceDoc]) -> Option<String> {
    for claim in &intent.claims {
        for req in &claim.requires {
            let found = evidence
                .iter()
                .any(|e| e.id == *req && e.supports.contains(&claim.id));
            if !found {
                return Some(format!("Provide anchored evidence for claim {}", claim.id));
            }
        }
    }
    None
}

fn ledger_payload(
    kind: &str,
    intent: &IntentDoc,
    chip: &Chip,
    evidence_cids: &[String],
    question: &str,
    gate: Option<&GateOutput>,
) -> Result<Value> {
    let mut payload = serde_json::json!({
        "kind": kind,
        "chip_hash": chip.hash(),
        "policy_epoch": intent.policy_epoch,
        "evidence_cids": evidence_cids,
        "intent_id": intent.id,
    });
    if let Some(g) = gate {
        payload["gate_proof"] = serde_json::json!(hex::encode(g.proof_ref));
        payload["decision"] = serde_json::json!(format!("{:?}", g.decision));
    }
    if !question.is_empty() {
        payload["question"] = serde_json::json!(question);
    }
    Ok(payload)
}

pub fn gate_run_from_atoms(
    intent_claims: Vec<Value>,
    evidence: Vec<Value>,
    policy: Value,
    epoch: u64,
) -> Result<GateVerdictLike> {
    if intent_claims.is_empty() {
        return Err(anyhow!("no intent claims"));
    }
    let base_intent = intent_claims[0]
        .get("intent_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("intent_id missing"))?
        .to_string();
    let utterance = intent_claims[0]
        .get("utterance")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("utterance missing"))?
        .to_string();
    let policy_features = intent_claims[0]
        .get("policy_features")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("policy_features missing"))?
        .iter()
        .map(|b| b.as_bool().unwrap_or(false))
        .collect::<Vec<_>>();

    let claims: Vec<IntentClaim> = intent_claims
        .iter()
        .map(|c| {
            let claim = c
                .get("claim")
                .ok_or_else(|| anyhow!("claim missing"))?;
            let id = claim
                .get("id")
                .and_then(|v| v.as_str())
                .ok_or_else(|| anyhow!("claim id missing"))?
                .to_string();
            let requires = claim
                .get("requires")
                .and_then(|v| v.as_array())
                .ok_or_else(|| anyhow!("claim requires missing"))?
                .iter()
                .map(|r| r.as_str().unwrap_or_default().to_string())
                .collect();
            Ok(IntentClaim { id, requires })
        })
        .collect::<Result<_>>()?;

    let intent = IntentDoc {
        id: base_intent,
        utterance,
        claims,
        policy_features,
        policy_epoch: epoch,
    };

    let evidence_docs: Vec<EvidenceDoc> = evidence
        .iter()
        .map(|e| {
            Ok(EvidenceDoc {
                id: e
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                anchored: e.get("anchored").and_then(|v| v.as_bool()).unwrap_or(false),
                supports: e
                    .get("supports")
                    .and_then(|v| v.as_array())
                    .unwrap_or(&vec![])
                    .iter()
                    .map(|s| s.as_str().unwrap_or_default().to_string())
                    .collect(),
                payload: e.get("payload").cloned().unwrap_or_else(|| json!({})),
            })
        })
        .collect::<Result<_>>()?;

    let policy_text = policy
        .get("policy")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("policy missing"))?;
    let policy_chip = Chip::parse(policy_text)?;

    if !evidence_docs.iter().all(|e| e.anchored) {
        return Ok(GateVerdictLike::Reject {
            reason: "evidence not anchored".into(),
        });
    }

    if let Some(question) = missing_evidence_question(&intent, &evidence_docs) {
        return Ok(GateVerdictLike::Ghost {
            missing_claim: question.clone(),
            question,
        });
    }

    if intent.policy_features.len() != policy_chip.features {
        return Ok(GateVerdictLike::Reject {
            reason: "policy feature mismatch".into(),
        });
    }
    let policy_allow = policy_chip.eval(&intent.policy_features)?;
    if !policy_allow {
        return Ok(GateVerdictLike::Reject {
            reason: "policy denied".into(),
        });
    }
    let gate = run_tdln_gate(&intent, policy_allow)?;
    Ok(GateVerdictLike::Commit {
        proof: hex::encode(gate.proof_ref),
    })
}

fn append_ledger(payload: Value, ledger_path: &str) -> Result<String> {
    let canon = json_atomic::canonize(&payload).map_err(|e| anyhow!("canon {e:?}"))?;
    let cid = blake3::hash(&canon);
    let entry = LedgerEntry::unsigned(&payload, None, &[])?;
    let path = Path::new(ledger_path);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut w = SimpleLedgerWriter::open_append(path)?;
    w.append(&entry)?;
    w.sync()?;
    Ok(hex::encode(cid.as_bytes()))
}

fn cid_for_value(v: &Value) -> String {
    let canon = json_atomic::canonize(v).unwrap_or_default();
    let h = blake3::hash(&canon);
    hex::encode(h.as_bytes())
}

pub fn verify_ledger(path: &str) -> Result<()> {
    let entries: Vec<_> = SimpleLedgerReader::from_path(path)?.iter().collect();
    let entries: Vec<_> = entries.into_iter().collect::<Result<_, _>>()?;
    entries.par_iter().try_for_each(|entry| verify_entry(entry))?;
    Ok(())
}

fn verify_entry(entry: &LedgerEntry) -> Result<()> {
    let value: Value = serde_json::from_slice(&entry.intent)?;
    let canon = json_atomic::canonize(&value).map_err(|e| anyhow!("canon {e:?}"))?;
    let cid = blake3::hash(&canon);
    let stored: [u8; 32] = entry.cid.0;
    if stored != *cid.as_bytes() {
        return Err(anyhow!("cid mismatch"));
    }
    entry.verify()?;
    Ok(())
}

pub fn demo_paths(base: &Path) -> (String, String, String) {
    (
        base.join("examples/intent.json").to_string_lossy().to_string(),
        base.join("examples/evidence.json").to_string_lossy().to_string(),
        base.join("examples/policy.txt").to_string_lossy().to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn missing_evidence_results_ghost() {
        let intent = IntentDoc {
            id: "i1".into(),
            utterance: "hello".into(),
            claims: vec![IntentClaim {
                id: "c1".into(),
                requires: vec!["e1".into()],
            }],
            policy_features: vec![true, true, true],
            policy_epoch: 1,
        };
        let evidence = vec![EvidenceDoc {
            id: "e0".into(),
            anchored: true,
            supports: vec!["c1".into()],
            payload: serde_json::json!({"foo":"bar"}),
        }];
        let q = missing_evidence_question(&intent, &evidence);
        assert!(q.is_some());
    }

    #[test]
    fn ledger_commit_verifies() {
        let dir = tempdir().unwrap();
        let intent_path = dir.path().join("intent.json");
        let evidence_path = dir.path().join("evidence.json");
        let policy_path = dir.path().join("policy.txt");
        let ledger_path = dir.path().join("ledger.ndjson");

        fs::write(
            &intent_path,
            serde_json::to_string(&IntentDoc {
                id: "intent-test".into(),
                utterance: "approve".into(),
                claims: vec![IntentClaim {
                    id: "c1".into(),
                    requires: vec!["e1".into()],
                }],
                policy_features: vec![true, true, true],
                policy_epoch: 1,
            })
            .unwrap(),
        )
        .unwrap();
        fs::write(
            &evidence_path,
            serde_json::to_string(&vec![EvidenceDoc {
                id: "e1".into(),
                anchored: true,
                supports: vec!["c1".into()],
                payload: serde_json::json!({"ok":true}),
            }])
            .unwrap(),
        )
        .unwrap();
        fs::write(
            &policy_path,
            "CHIP v0\nFEATURES n=3\nGATES m=1\ng0 = AND(f0,f1,f2)\nOUTPUT = g0\n",
        )
        .unwrap();

        let verdict = run_gate(
            intent_path.to_string_lossy().as_ref(),
            evidence_path.to_string_lossy().as_ref(),
            policy_path.to_string_lossy().as_ref(),
            ledger_path.to_string_lossy().as_ref(),
        )
        .unwrap();
        match verdict {
            GateVerdict::Commit { .. } => {}
            _ => panic!("expected commit"),
        }
        verify_ledger(ledger_path.to_string_lossy().as_ref()).unwrap();
    }
}
