use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use chip_as_code::chip_ir::Chip;
use chip_as_code::evolve::{evolve, replay, stats, Backend, EvolutionConfig, Task};
use chip_as_code::gatebox::{run_gate, verify_ledger, GateVerdict};
use chip_as_code::zlayer::{zlayer_demo, zlayer_diagnose, zlayer_evolve, zlayer_gate_demo, zlayer_gate_verify, zlayer_merge, zlayer_stress, zlayer_value, zlayer_verify, zlayer_verify_file, zlayer_verify_streaming};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "chip", version, about = "Deterministic chip trainer + gatebox")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Demo,
    Zlayer {
        #[command(subcommand)]
        command: ZlayerCommands,
    },
    Gate {
        #[command(subcommand)]
        command: GateCommands,
    },
    GateLegacy {
        #[arg(long)]
        intent: PathBuf,
        #[arg(long)]
        evidence: PathBuf,
        #[arg(long)]
        policy: PathBuf,
        #[arg(long, default_value = "out/ledger.ndjson")]
        ledger: PathBuf,
    },
    Verify {
        #[arg(long)]
        ledger: PathBuf,
    },
    Evolve {
        #[arg(long)]
        task: String,
        #[arg(long)]
        seed: u64,
        #[arg(long, default_value_t = 200)]
        generations: usize,
        #[arg(long, default_value_t = 2000)]
        population: usize,
        #[arg(long, default_value_t = 2000)]
        offspring: usize,
        #[arg(long, default_value_t = 50)]
        elite: usize,
        #[arg(long, default_value_t = 32)]
        max_gates: usize,
        #[arg(long, default_value_t = 1500)]
        mutation_rate_per10k: u32,
        #[arg(long, default_value_t = false)]
        debug: bool,
        #[arg(long, default_value = "cpu")]
        backend: String,
    },
    Replay {
        #[arg(long)]
        task: String,
        #[arg(long)]
        seed: u64,
        #[arg(long)]
        chip: PathBuf,
        #[arg(long, default_value = "cpu")]
        backend: String,
    },
    Stats {
        #[arg(long)]
        task: String,
        #[arg(long)]
        seed: u64,
        #[arg(long, default_value = "cpu")]
        backend: String,
    },
}

#[derive(Subcommand)]
enum ZlayerCommands {
    Demo,
    Verify {
        #[arg(long, default_value = "out/zlayer")]
        dir: Option<PathBuf>,
        #[arg(long)]
        file: Option<PathBuf>,
        /// Use streaming mode for large files (1M+ atoms). Faster and uses less memory.
        #[arg(long, default_value = "full")]
        mode: String,
    },
    /// Deep diagnostic analysis of atom distribution and deduplication
    Diagnose {
        #[arg(long)]
        file: PathBuf,
    },
    Evolve {
        #[arg(long)]
        task: String,
        #[arg(long)]
        seed: u64,
        #[arg(long, default_value_t = 50)]
        generations: usize,
        #[arg(long, default_value_t = 2000)]
        population: usize,
        #[arg(long, default_value_t = 2000)]
        offspring: usize,
        #[arg(long, default_value_t = 50)]
        elite: usize,
        #[arg(long, default_value_t = 32)]
        max_gates: usize,
        #[arg(long, default_value_t = 1500)]
        mutation_rate_per10k: u32,
        #[arg(long, default_value = "cpu")]
        backend: String,
    },
    Merge {
        #[arg(long)]
        glob: String,
        #[arg(long)]
        out: PathBuf,
    },
    Value {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value = "gate_rollup")]
        mode: String,
        #[arg(long, default_value_t = 10_000)]
        unit_value_per10k: u64,
    },
    Stress {
        #[arg(long)]
        seed: u64,
        #[arg(long)]
        n: usize,
        #[arg(long)]
        shards: usize,
        #[arg(long, default_value = "out/stress")]
        out: PathBuf,
    },
    StressValue {
        #[arg(long)]
        seed: u64,
        #[arg(long)]
        n: usize,
        #[arg(long)]
        shards: usize,
        #[arg(long, default_value = "out/stress_value")]
        out: PathBuf,
        #[arg(long, default_value_t = 10_000)]
        unit_value_per10k: u64,
    },
}

#[derive(Subcommand)]
enum GateCommands {
    Demo,
    Verify {
        #[arg(long, default_value = "out/zgate")]
        dir: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Demo => demo()?,
        Commands::Zlayer { command } => match command {
            ZlayerCommands::Demo => {
                zlayer_demo("out/zlayer")?;
                println!("zlayer demo written to out/zlayer");
            }
            ZlayerCommands::Verify { dir, file, mode } => {
                let stats = if let Some(file) = file {
                    if mode == "streaming" {
                        zlayer_verify_streaming(&file)?
                    } else {
                        zlayer_verify_file(&file)?
                    }
                } else {
                    let dir_path = dir.unwrap_or_else(|| PathBuf::from("out/zlayer"));
                    zlayer_verify(&dir_path)?
                };
                println!(
                    "zlayer verified atoms_loaded={} atoms_unique={} derivations={} commit={} ghost={} reject={} mode={}",
                    stats.atoms_loaded,
                    stats.atoms_unique,
                    stats.derivations_replayed,
                    stats.commits,
                    stats.ghosts,
                    stats.rejects,
                    mode
                );
            }
            ZlayerCommands::Diagnose { file } => {
                let report = zlayer_diagnose(&file)?;
                println!("╔══════════════════════════════════════════════════════════╗");
                println!("║           Z-Layer Diagnostic Report                      ║");
                println!("╚══════════════════════════════════════════════════════════╝");
                println!();
                println!("📊 SUMMARY");
                println!("  Total atoms: {}", report.total_atoms);
                println!("  Unique CIDs: {}", report.unique_cids);
                println!("  Derivations: {}", report.derivations);
                println!("  Duplicates:  {}", report.duplicate_atoms);
                println!();
                
                println!("📦 ATOM TYPES");
                for (t, count) in &report.atom_types {
                    println!("  {}: {}", t, count);
                }
                println!();
                
                // Template + Occurrence breakdown (the important part!)
                println!("═══════════════════════════════════════════════════════════");
                println!("🎯 VERDICT ANALYSIS (Template + Occurrence Pattern)");
                println!("═══════════════════════════════════════════════════════════");
                println!();
                
                println!("📋 EVENTS (occurrences = decisions executed)");
                if report.verdict_events.is_empty() {
                    println!("  (old format - using verdict_distribution)");
                    for (v, count) in &report.verdict_distribution {
                        println!("  {}: {}", v, count);
                    }
                } else {
                    let total_events: usize = report.verdict_events.values().sum();
                    println!("  Total: {}", total_events);
                    for (v, count) in &report.verdict_events {
                        println!("  {}: {}", v, count);
                    }
                }
                println!();
                
                println!("📚 TEMPLATES (catalog = unique verdict reasons)");
                if report.verdict_templates.is_empty() {
                    println!("  (old format - using unique_verdict_cids)");
                    for (v, count) in &report.unique_verdict_cids {
                        println!("  {}: {}", v, count);
                    }
                } else {
                    let total_templates: usize = report.verdict_templates.values().sum();
                    println!("  Total: {}", total_templates);
                    for (v, count) in &report.verdict_templates {
                        println!("  {}: {}", v, count);
                    }
                }
                println!();
                
                if !report.template_frequency.is_empty() {
                    println!("🔥 TOP TEMPLATES BY USAGE");
                    for (tcid, verdict, count) in &report.template_frequency {
                        println!("  {}... [{}] → {} events", &tcid[..16], verdict, count);
                    }
                    println!();
                }
                
                // Invariant check
                println!("═══════════════════════════════════════════════════════════");
                println!("✅ INVARIANT CHECK");
                let total_events: usize = report.verdict_events.values().sum();
                let events_ok = total_events == report.derivations || report.verdict_events.is_empty();
                if events_ok {
                    println!("  derivations == verdict_events: ✓ ({} == {})", 
                             report.derivations, total_events);
                } else {
                    println!("  ⚠️  derivations != verdict_events: {} != {}", 
                             report.derivations, total_events);
                }
            }
            ZlayerCommands::Evolve {
                task,
                seed,
                generations,
                population,
                offspring,
                elite,
                max_gates,
                mutation_rate_per10k,
                backend,
            } => {
                let task_enum = Task::from_str(&task).ok_or_else(|| anyhow!("unknown task"))?;
                let backend_enum = Backend::from_str(&backend).ok_or_else(|| anyhow!("unknown backend"))?;
                let cfg = EvolutionConfig {
                    task: task_enum,
                    seed,
                    generations,
                    population,
                    offspring,
                    elite,
                    max_gates,
                    mutation_rate_per10k,
                    out_dir: "out".into(),
                    debug: false,
                    backend: backend_enum,
                };
                zlayer_evolve(cfg, Path::new("out/zlayer_evo"))?;
                println!("zlayer evolve written to out/zlayer_evo");
            }
            ZlayerCommands::Merge { glob, out } => {
                let report = zlayer_merge(&glob, &out)?;
                println!(
                    "merged atoms_loaded={} atoms_unique={} into {}",
                    report.atoms_loaded,
                    report.atoms_unique,
                    out.to_string_lossy()
                );
            }
            ZlayerCommands::Value {
                input,
                out,
                mode,
                unit_value_per10k,
            } => {
                let summary = zlayer_value(&input, &out, &mode, unit_value_per10k)?;
                println!(
                    "value_run summary commit={} ghost={} reject={} total={} total_value_per10k={} reasons={}",
                    summary.commit_count,
                    summary.ghost_count,
                    summary.reject_count,
                    summary.total_events,
                    summary.total_value_per10k,
                    summary.reason_counts.len()
                );
            }
            ZlayerCommands::Stress {
                seed,
                n,
                shards,
                out,
            } => {
                let paths = zlayer_stress(seed, n, shards, &out)?;
                println!(
                    "stress generated {} shards at {} (first shard={})",
                    paths.len(),
                    out.to_string_lossy(),
                    paths.get(0).map(|p| p.to_string_lossy()).unwrap_or_default()
                );
            }
            ZlayerCommands::StressValue {
                seed,
                n,
                shards,
                out,
                unit_value_per10k,
            } => {
                let shards_dir = out.join("shards");
                let merge_path = out.join("all.ndjson");
                let value_dir = out.join("value");
                let merged_with_value = out.join("all_with_value.ndjson");

                let shard_paths = zlayer_stress(seed, n, shards, &shards_dir)?;
                println!("stress-value shards={} dir={}", shard_paths.len(), shards_dir.to_string_lossy());

                let merge_report = zlayer_merge(&shards_dir.join("shard_*.ndjson").to_string_lossy(), &merge_path)?;
                println!(
                    "merge atoms_loaded={} atoms_unique={} -> {}",
                    merge_report.atoms_loaded,
                    merge_report.atoms_unique,
                    merge_path.to_string_lossy()
                );

                let _summary = zlayer_value(&merge_path, &value_dir, "gate_rollup", unit_value_per10k)?;

                let merge_pattern = format!("{},{}", merge_path.to_string_lossy(), value_dir.join("*.ndjson").to_string_lossy());
                let merge_report2 = zlayer_merge(&merge_pattern, &merged_with_value)?;
                println!(
                    "merge+value atoms_loaded={} atoms_unique={} -> {}",
                    merge_report2.atoms_loaded,
                    merge_report2.atoms_unique,
                    merged_with_value.to_string_lossy()
                );

                let stats = zlayer_verify_file(&merged_with_value)?;
                println!(
                    "verify derivations={} commit={} ghost={} reject={}",
                    stats.derivations_replayed,
                    stats.commits,
                    stats.ghosts,
                    stats.rejects
                );
            }
        },
        Commands::Gate { command } => match command {
            GateCommands::Demo => {
                zlayer_gate_demo(Path::new("out/zgate"))?;
            }
            GateCommands::Verify { dir } => {
                zlayer_gate_verify(&dir)?;
            }
        },
        Commands::GateLegacy {
            intent,
            evidence,
            policy,
            ledger,
        } => {
            let verdict = run_gate_paths(&intent, &evidence, &policy, &ledger)?;
            print_verdict(&verdict);
        }
        Commands::Verify { ledger } => {
            verify_ledger(ledger.to_string_lossy().as_ref())?;
            println!("ledger verified");
        }
        Commands::Evolve {
            task,
            seed,
            generations,
            population,
            offspring,
            elite,
            max_gates,
            mutation_rate_per10k,
            debug,
            backend,
        } => {
            let task_enum = Task::from_str(&task).ok_or_else(|| anyhow!("unknown task"))?;
            let backend_enum = Backend::from_str(&backend).ok_or_else(|| anyhow!("unknown backend"))?;
            fs::create_dir_all("out")?;
            let cfg = EvolutionConfig {
                task: task_enum,
                seed,
                generations,
                population,
                offspring,
                elite,
                max_gates,
                mutation_rate_per10k,
                out_dir: "out".into(),
                debug,
                backend: backend_enum,
            };
            let res = evolve(cfg)?;
            #[cfg(feature = "gpu")]
            if let Some(meta) = res.backend_info.gpu.clone() {
                println!(
                    "BACKEND {} adapter={} shader_hash={}",
                    res.config.backend.as_str(),
                    meta.adapter,
                    meta.shader_hash
                );
            }
            println!(
                "best chip hash={} train={:.3} test={:.3} backend={}",
                res.best.chip.hash(),
                res.best.train_acc_per10k as f64 / 10000.0,
                res.best.test_acc_per10k as f64 / 10000.0,
                res.config.backend.as_str()
            );
        }
        Commands::Replay { task, seed, chip, backend } => {
            let task_enum = Task::from_str(&task).ok_or_else(|| anyhow!("unknown task"))?;
            let backend_enum = Backend::from_str(&backend).ok_or_else(|| anyhow!("unknown backend"))?;
            let text = fs::read_to_string(&chip)?;
            let chip = Chip::parse(&text)?;
            let report = replay(task_enum, seed, &chip, "out", backend_enum)?;
            println!(
                "replay_ok={} train={:.3} test={:.3} chip_hash={} backend={}",
                report.replay_ok,
                report.train_acc_per10k as f64 / 10000.0,
                report.test_acc_per10k as f64 / 10000.0,
                report.chip_hash,
                backend
            );
        }
        Commands::Stats { task, seed, backend } => {
            let task_enum = Task::from_str(&task).ok_or_else(|| anyhow!("unknown task"))?;
            Backend::from_str(&backend).ok_or_else(|| anyhow!("unknown backend"))?;
            let st = stats(task_enum, seed)?;
            println!(
                "train_size={} test_size={} train_pos={} train_neg={} test_pos={} test_neg={}",
                st.train_size, st.test_size, st.train_pos, st.train_neg, st.test_pos, st.test_neg
            );
            for (i, s) in st.samples.iter().enumerate() {
                println!("sample{}: x0={} x1={} y={}", i, s.x0, s.x1, s.y);
            }
        }
    }
    Ok(())
}

fn run_gate_paths(intent: &Path, evidence: &Path, policy: &Path, ledger: &Path) -> Result<GateVerdict> {
    run_gate(
        intent.to_string_lossy().as_ref(),
        evidence.to_string_lossy().as_ref(),
        policy.to_string_lossy().as_ref(),
        ledger.to_string_lossy().as_ref(),
    )
}

fn print_verdict(v: &GateVerdict) {
    match v {
        GateVerdict::Commit {
            cid_hex,
            chip_hash,
            proof,
        } => println!("COMMIT cid={} chip_hash={} proof={} ", cid_hex, chip_hash, proof),
        GateVerdict::Ghost { question } => println!("GHOST question={}", question),
        GateVerdict::Reject { reason } => println!("REJECT reason={}", reason),
    }
}

fn demo() -> Result<()> {
    let base = Path::new(".");
    let demo_dir = base.join("examples");
    fs::create_dir_all(&demo_dir)?;
    let ledger = Path::new("out/ledger.ndjson");
    if ledger.exists() {
        fs::remove_file(ledger)?;
    }

    let scenarios = write_demo_fixtures(&demo_dir)?;
    for (name, intent, evidence, policy) in scenarios {
        println!("scenario: {}", name);
        let verdict = run_gate_paths(&intent, &evidence, &policy, ledger)?;
        print_verdict(&verdict);
    }
    Ok(())
}

fn write_demo_fixtures(demo_dir: &Path) -> Result<Vec<(String, PathBuf, PathBuf, PathBuf)>> {
    let policy = demo_dir.join("policy.txt");
    let policy_text = "CHIP v0\nFEATURES n=3\nGATES m=1\ng0 = AND(f0,f1,f2)\nOUTPUT = g0\n";
    fs::write(&policy, policy_text)?;

    // Scenario 1: good evidence, policy ok
    let intent_good = demo_dir.join("intent_good.json");
    let evidence_good = demo_dir.join("evidence_good.json");
    fs::write(
        &intent_good,
        serde_json::to_string_pretty(&serde_json::json!({
            "id":"intent-good",
            "utterance":"approve budget",
            "policy_features":[true,true,true],
            "policy_epoch":1,
            "claims":[{"id":"c1","requires":["e1"]}]
        }))?,
    )?;
    fs::write(
        &evidence_good,
        serde_json::to_string_pretty(&serde_json::json!([
            {"id":"e1","anchored":true,"supports":["c1"],"payload":{"kind":"pdf"}}
        ]))?,
    )?;

    // Scenario 2: missing evidence -> ghost
    let intent_missing = demo_dir.join("intent_missing.json");
    let evidence_missing = demo_dir.join("evidence_missing.json");
    fs::write(
        &intent_missing,
        fs::read_to_string(&intent_good)?,
    )?;
    fs::write(
        &evidence_missing,
        serde_json::to_string_pretty(&serde_json::json!([
            {"id":"e2","anchored":true,"supports":["c2"],"payload":{"kind":"note"}}
        ]))?,
    )?;

    // Scenario 3: unanchored evidence -> reject
    let intent_unanchored = demo_dir.join("intent_unanchored.json");
    let evidence_unanchored = demo_dir.join("evidence_unanchored.json");
    fs::write(&intent_unanchored, fs::read_to_string(&intent_good)?)?;
    fs::write(
        &evidence_unanchored,
        serde_json::to_string_pretty(&serde_json::json!([
            {"id":"e1","anchored":false,"supports":["c1"],"payload":{"kind":"pdf"}}
        ]))?,
    )?;

    // Scenario 4: policy violation -> reject
    let intent_policy = demo_dir.join("intent_policy_violation.json");
    let evidence_policy = demo_dir.join("evidence_policy_violation.json");
    fs::write(
        &intent_policy,
        serde_json::to_string_pretty(&serde_json::json!({
            "id":"intent-policy-deny",
            "utterance":"approve budget",
            "policy_features":[false,true,true],
            "policy_epoch":1,
            "claims":[{"id":"c1","requires":["e1"]}]
        }))?,
    )?;
    fs::write(
        &evidence_policy,
        serde_json::to_string_pretty(&serde_json::json!([
            {"id":"e1","anchored":true,"supports":["c1"],"payload":{"kind":"pdf"}}
        ]))?,
    )?;

    Ok(vec![
        (
            "good evidence + policy ok".into(),
            intent_good,
            evidence_good,
            policy.clone(),
        ),
        (
            "missing evidence".into(),
            intent_missing,
            evidence_missing,
            policy.clone(),
        ),
        (
            "unanchored evidence".into(),
            intent_unanchored,
            evidence_unanchored,
            policy.clone(),
        ),
        (
            "policy violation".into(),
            intent_policy,
            evidence_policy,
            policy,
        ),
    ])
}
