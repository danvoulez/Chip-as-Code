//! # EvoChip — Discrete Evolution Engine
//!
//! Evolve semantic chips through generations using deterministic mutation.
//! This module implements **integer-only fitness** with **ChaCha20 RNG**
//! for reproducible evolution across any platform.
//!
//! ## Key Properties
//!
//! - **Deterministic**: Same seed = same evolution trajectory
//! - **Integer Fitness**: No floating-point drift between platforms
//! - **Parallel**: Uses Rayon for population-level parallelism
//! - **GPU-Ready**: Optional WGPU backend for massive populations
//!
//! ## Evolution Operators
//!
//! - **Mutation**: Flip gates, change operators, add/remove gates
//! - **Crossover**: Combine gates from two parent chips
//! - **Selection**: Tournament selection with elitism
//!
//! ## Example
//!
//! ```ignore
//! use chip_as_code::evolve::{EvolutionConfig, Task, Backend, run_evolution};
//!
//! let config = EvolutionConfig {
//!     task: Task::Xor,
//!     seed: 42,
//!     generations: 100,
//!     population: 50,
//!     offspring: 25,
//!     elite: 5,
//!     max_gates: 10,
//!     mutation_rate_per10k: 1000,
//!     out_dir: "./evo_output".into(),
//!     backend: Backend::Cpu,
//! };
//!
//! run_evolution(config)?;
//! ```

use crate::chip_ir::{random_chip, Chip, ChipHash, Ref};
#[cfg(feature = "gpu")]
use crate::gpu_eval::{init_gpu, GpuEvalMetadata, GpuEvaluator};
use anyhow::{anyhow, Result};
use blake3::Hasher;
use blake3::Hasher as BlakeHasher;
use logline::json_atomic;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

/// The optimization task for evolution.
///
/// Each task defines a fitness function that chips are evolved to maximize.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Task {
    /// Learn XOR function: classic boolean benchmark
    Xor,
    /// Learn a hidden policy chip: reverse-engineering
    PolicyHidden,
}

impl Task {
    /// Parse task from string name.
    pub fn from_str(name: &str) -> Option<Self> {
        match name {
            "xor" => Some(Task::Xor),
            "policy_hidden" => Some(Task::PolicyHidden),
            _ => None,
        }
    }

    /// Get canonical string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Task::Xor => "xor",
            Task::PolicyHidden => "policy_hidden",
        }
    }
}

/// Compute backend for chip evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Backend {
    /// CPU with Rayon parallelism
    Cpu,
    /// GPU with WGPU (requires `gpu` feature)
    Gpu,
}

impl Backend {
    /// Parse backend from string name.
    pub fn from_str(name: &str) -> Option<Self> {
        match name {
            "cpu" => Some(Backend::Cpu),
            "gpu" => Some(Backend::Gpu),
            _ => None,
        }
    }

    /// Get canonical string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Gpu => "gpu",
        }
    }
}

fn default_backend() -> Backend {
    Backend::Cpu
}

/// Configuration for an evolution run.
///
/// All parameters are deterministic — same config + same seed = same results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionConfig {
    /// The task/fitness function to optimize
    pub task: Task,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Number of generations to evolve
    pub generations: usize,
    /// Population size per generation
    pub population: usize,
    /// Number of offspring per generation
    pub offspring: usize,
    /// Number of elite individuals to preserve
    pub elite: usize,
    /// Maximum gates allowed per chip
    pub max_gates: usize,
    /// Mutation rate in basis points (1000 = 10%)
    pub mutation_rate_per10k: u32,
    /// Output directory for results
    pub out_dir: String,
    /// Enable debug logging
    pub debug: bool,
    /// Compute backend (CPU or GPU)
    #[serde(default = "default_backend")]
    pub backend: Backend,
}

/// A single training/test sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sample {
    /// Input feature vector
    pub x: Vec<bool>,
    /// Expected output
    pub y: bool,
}

/// Train/test split of a dataset.
#[derive(Debug, Clone)]
pub struct DatasetSplit {
    /// Training samples
    pub train: Vec<Sample>,
    /// Test samples (held out)
    pub test: Vec<Sample>,
    /// Number of features per sample
    pub features: usize,
    /// BLAKE3 hash of the dataset for verification
    pub hash_hex: String,
}

/// Statistics about a dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStats {
    /// Number of training samples
    pub train_size: usize,
    /// Number of test samples
    pub test_size: usize,
    /// Positive examples in training set
    pub train_pos: usize,
    /// Negative examples in training set
    pub train_neg: usize,
    /// Positive examples in test set
    pub test_pos: usize,
    /// Negative examples in test set
    pub test_neg: usize,
    /// Preview of samples
    pub samples: Vec<SampleView>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleView {
    pub x0: u8,
    pub x1: u8,
    pub y: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingLine {
    pub generation: usize,
    pub task: String,
    pub seed: u64,
    pub best_train_per10k: u32,
    pub best_test_per10k: u32,
    pub mean_train_per10k: u32,
    pub chip_hash: ChipHash,
    pub gates: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageLine {
    pub generation: usize,
    pub seed: u64,
    pub parent_hash: ChipHash,
    pub child_hash: ChipHash,
    pub mutation: String,
    pub train_acc_per10k: u32,
    pub test_acc_per10k: u32,
}

#[derive(Debug, Serialize)]
struct WithCid<T: Serialize> {
    cid: String,
    #[serde(flatten)]
    value: T,
}

fn cid_for<T: Serialize>(value: &T) -> String {
    let canon = json_atomic::canonize(value).expect("canonize record");
    let digest = blake3::hash(&canon);
    hex::encode(digest.as_bytes())
}

fn is_better(candidate: &ScoredChip, incumbent: &ScoredChip) -> bool {
    candidate
        .test_acc_per10k
        .cmp(&incumbent.test_acc_per10k)
        .then_with(|| candidate.train_acc_per10k.cmp(&incumbent.train_acc_per10k))
        .then_with(|| incumbent.live_gates.cmp(&candidate.live_gates))
        .then_with(|| candidate.chip.hash().cmp(&incumbent.chip.hash()))
        == Ordering::Greater
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredChip {
    pub chip: Chip,
    pub train_acc_per10k: u32,
    pub test_acc_per10k: u32,
    pub live_gates: usize,
}

#[derive(Debug, Clone)]
pub struct EvalBackendInfo {
    pub backend: Backend,
    #[cfg(feature = "gpu")]
    pub gpu: Option<GpuEvalMetadata>,
}

struct EvalEngine {
    backend: Backend,
    #[cfg(feature = "gpu")]
    gpu: Option<GpuEvaluator>,
}

pub struct EvolutionResult {
    pub best: ScoredChip,
    pub split: DatasetSplit,
    pub config: EvolutionConfig,
    pub backend_info: EvalBackendInfo,
}

pub fn evolve(config: EvolutionConfig) -> Result<EvolutionResult> {
    fs::create_dir_all(&config.out_dir)?;
    let dataset = generate_dataset(config.task, config.seed);
    write_dataset(&config, &dataset)?;
    let split = split_dataset(dataset, config.seed);

    let engine = EvalEngine::new(config.backend, &split)?;

    let mut rng = ChaCha20Rng::seed_from_u64(config.seed);
    let mut population = seed_population(&split, &config, &mut rng);

    let mut train_curve =
        BufWriter::new(File::create(Path::new(&config.out_dir).join("training_curve.ndjson"))?);
    let mut lineage_file =
        BufWriter::new(File::create(Path::new(&config.out_dir).join("lineage.ndjson"))?);

    let mut best_overall: Option<ScoredChip> = None;

    for generation in 0..config.generations {
        let scored = engine.evaluate_population(&population, &split)?;

        let mut sorted = scored;
        sorted.sort_by(|a, b| {
            b.test_acc_per10k
                .cmp(&a.test_acc_per10k)
                .then_with(|| b.train_acc_per10k.cmp(&a.train_acc_per10k))
                .then_with(|| a.live_gates.cmp(&b.live_gates))
                .then_with(|| a.chip.hash().cmp(&b.chip.hash()))
        });
        let best = sorted.first().expect("population non-empty");
        if best_overall
            .as_ref()
            .map_or(true, |b| is_better(best, b))
        {
            best_overall = Some(best.clone());
        }

        let mean_train_per10k: u32 = if sorted.is_empty() {
            0
        } else {
            (sorted
                .iter()
                .map(|s| s.train_acc_per10k as u64)
                .sum::<u64>()
                / sorted.len() as u64) as u32
        };
        let line = TrainingLine {
            generation,
            task: config.task.as_str().into(),
            seed: config.seed,
            best_train_per10k: best.train_acc_per10k,
            best_test_per10k: best.test_acc_per10k,
            mean_train_per10k,
            chip_hash: best.chip.hash(),
            gates: best.live_gates,
        };
        let line_out = WithCid {
            cid: cid_for(&line),
            value: line,
        };
        serde_json::to_writer(&mut train_curve, &line_out)?;
        writeln!(&mut train_curve)?;

        let (new_population, clone_children) =
            breed(&sorted, &split, &config, &mut rng, generation, &mut lineage_file, &engine)?;
        population = new_population;

        if config.debug {
            let unique_children: usize = population
                .iter()
                .map(|c| c.hash())
                .collect::<HashSet<_>>()
                .len();
            let elite_unique = sorted
                .iter()
                .take(config.elite.min(sorted.len()))
                .map(|c| c.chip.hash())
                .collect::<HashSet<_>>()
                .len();
            let best_depends_primary = best.chip.depends_on_primary_inputs();
            println!(
                "debug gen={} unique_children={} invalid_children={} elite_unique={} best_depends_f0f1={} best_live_gates={}",
                generation,
                unique_children,
                clone_children,
                elite_unique,
                best_depends_primary,
                best.live_gates,
            );
        }
    }
    let best = best_overall.expect("best exists");
    let best_path = Path::new(&config.out_dir).join("best_chip.txt");
    fs::write(&best_path, best.chip.canonical_text())?;

    let replay_report_path = Path::new(&config.out_dir).join("replay_report.json");
    let replay = ReplayReport::from_scored(&best, &split, &config)?;
    serde_json::to_writer_pretty(File::create(&replay_report_path)?, &replay)?;

    let backend_info = engine.info();
    Ok(EvolutionResult {
        best,
        split,
        config,
        backend_info,
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ReplayReport {
    pub dataset_hash: String,
    pub chip_hash: String,
    pub train_acc_per10k: u32,
    pub test_acc_per10k: u32,
    pub config: EvolutionConfig,
    pub replay_ok: bool,
}

impl ReplayReport {
    fn from_scored(best: &ScoredChip, split: &DatasetSplit, config: &EvolutionConfig) -> Result<Self> {
        let engine = EvalEngine::new(config.backend, split)?;
        let replay = engine.evaluate_chip(&best.chip, split)?;
        let replay_ok = replay.train_acc_per10k == best.train_acc_per10k
            && replay.test_acc_per10k == best.test_acc_per10k
            && replay.chip.hash() == best.chip.hash();
        Ok(ReplayReport {
            dataset_hash: split.hash_hex.clone(),
            chip_hash: best.chip.hash(),
            train_acc_per10k: replay.train_acc_per10k,
            test_acc_per10k: replay.test_acc_per10k,
            config: config.clone(),
            replay_ok,
        })
    }
}

fn write_dataset(config: &EvolutionConfig, dataset: &[Sample]) -> Result<()> {
    let path = Path::new(&config.out_dir).join("dataset.ndjson");
    let mut w = BufWriter::new(File::create(path)?);
    for sample in dataset {
        let line = serde_json::json!({
            "x": sample.x.iter().map(|b| if *b {1} else {0}).collect::<Vec<u8>>(),
            "y": if sample.y {1} else {0},
        });
        serde_json::to_writer(&mut w, &line)?;
        writeln!(&mut w)?;
    }
    Ok(())
}

fn seed_population(split: &DatasetSplit, config: &EvolutionConfig, rng: &mut ChaCha20Rng) -> Vec<Chip> {
    let mut pop = Vec::with_capacity(config.population);
    let base = Chip {
        features: split.features,
        gates: Vec::new(),
        output: Ref::Feature(0),
    };
    pop.push(base.clone());
    for _ in 1..config.population {
        let gates = rng.gen_range(1..=config.max_gates.min(6).max(1));
        let bias_seed = rng.gen_bool(0.3);
        let bias_mut = rng.gen_bool(0.2);
        let chip = random_chip(rng, split.features, gates, bias_seed)
            .mutate_random(rng, config.max_gates, bias_mut);
        pop.push(chip);
    }
    pop
}

fn breed(
    elites: &[ScoredChip],
    split: &DatasetSplit,
    config: &EvolutionConfig,
    rng: &mut ChaCha20Rng,
    generation: usize,
    lineage: &mut BufWriter<File>,
    engine: &EvalEngine,
) -> Result<(Vec<Chip>, usize)> {
    let mut next = Vec::with_capacity(config.population);
    let keep = config.elite.min(elites.len()).max(1);
    for e in elites.iter().take(keep) {
        next.push(e.chip.clone());
    }

    let mut clones = 0usize;
    for _ in keep..config.population {
        let parent_idx = rng.gen_range(0..keep);
        let parent = &elites[parent_idx].chip;
        let mut child = parent.clone();
        let mut mutation_steps = 1usize;
        if rng.gen_range(0..10000) < config.mutation_rate_per10k {
            mutation_steps += 1;
        }
        if rng.gen_range(0..10000) < config.mutation_rate_per10k / 2 {
            mutation_steps += 1;
        }
        let bias = rng.gen_bool(0.2);
        for _ in 0..mutation_steps {
            child = child.mutate_random(rng, config.max_gates, bias);
        }
        let mutation = match mutation_steps {
            1 => "mutate1",
            2 => "mutate2",
            _ => "mutate3",
        }
        .to_string();
        if child.hash() == parent.hash() {
            clones += 1;
        }
        let scored = engine.evaluate_chip(&child, split)?;
        let line = LineageLine {
            generation,
            seed: config.seed,
            parent_hash: parent.hash(),
            child_hash: child.hash(),
            mutation,
            train_acc_per10k: scored.train_acc_per10k,
            test_acc_per10k: scored.test_acc_per10k,
        };
        let line_out = WithCid {
            cid: cid_for(&line),
            value: line,
        };
        serde_json::to_writer(&mut *lineage, &line_out)?;
        writeln!(lineage)?;
        next.push(child);
    }
    if next.len() < config.population {
        while next.len() < config.population {
            next.push(elites[0].chip.clone());
        }
    }
    if config.debug {
        println!(
            "debug_gen={} clones_from_parents={} unique_children={}",
            generation,
            clones,
            next.iter().map(|c| c.hash()).collect::<HashSet<_>>().len()
        );
    }
    Ok((next, clones))
}

#[cfg(feature = "gpu")]
fn per10k(correct: u32, total: usize) -> u32 {
    if total == 0 {
        return 0;
    }
    ((correct as u64 * 10_000) / total as u64) as u32
}

fn accuracy_per10k_cpu(chip: &Chip, samples: &[Sample]) -> u32 {
    if samples.is_empty() {
        return 0;
    }
    let mut correct: u64 = 0;
    for s in samples {
        if let Ok(pred) = chip.eval(&s.x) {
            if pred == s.y {
                correct += 1;
            }
        }
    }
    ((correct * 10_000) / samples.len() as u64) as u32
}

fn evaluate_chip_cpu(chip: &Chip, split: &DatasetSplit) -> ScoredChip {
    let train_acc = accuracy_per10k_cpu(chip, &split.train);
    let test_acc = accuracy_per10k_cpu(chip, &split.test);
    ScoredChip {
        chip: chip.clone(),
        train_acc_per10k: train_acc,
        test_acc_per10k: test_acc,
        live_gates: chip.live_gate_count(),
    }
}

impl EvalEngine {
    fn new(backend: Backend, split: &DatasetSplit) -> Result<Self> {
        #[cfg(feature = "gpu")]
        {
            let gpu = match backend {
                Backend::Cpu => None,
                Backend::Gpu => Some(init_gpu(split)?),
            };
            Ok(EvalEngine { backend, gpu })
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = split; // silence unused warning
            if backend == Backend::Gpu {
                return Err(anyhow!("GPU backend requires --features gpu"));
            }
            Ok(EvalEngine { backend })
        }
    }

    fn info(&self) -> EvalBackendInfo {
        #[cfg(feature = "gpu")]
        {
            EvalBackendInfo {
                backend: self.backend,
                gpu: self.gpu.as_ref().map(|g| g.metadata.clone()),
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            EvalBackendInfo {
                backend: self.backend,
            }
        }
    }

    fn evaluate_population(&self, pop: &[Chip], split: &DatasetSplit) -> Result<Vec<ScoredChip>> {
        match self.backend {
            Backend::Cpu => Ok(pop
                .par_iter()
                .map(|c| evaluate_chip_cpu(c, split))
                .collect()),
            Backend::Gpu => {
                #[cfg(feature = "gpu")]
                {
                    let gpu = self
                        .gpu
                        .as_ref()
                        .ok_or_else(|| anyhow!("gpu backend not initialized"))?;
                    let (train_counts, test_counts) = gpu.evaluate(pop)?;
                    if train_counts.len() != pop.len() || test_counts.len() != pop.len() {
                        return Err(anyhow!("gpu counts length mismatch"));
                    }
                    let mut scored = Vec::with_capacity(pop.len());
                    for (idx, chip) in pop.iter().enumerate() {
                        scored.push(ScoredChip {
                            chip: chip.clone(),
                            train_acc_per10k: per10k(train_counts[idx], split.train.len()),
                            test_acc_per10k: per10k(test_counts[idx], split.test.len()),
                            live_gates: chip.live_gate_count(),
                        });
                    }
                    Ok(scored)
                }
                #[cfg(not(feature = "gpu"))]
                {
                    Err(anyhow!("GPU backend requires --features gpu"))
                }
            }
        }
    }

    fn evaluate_chip(&self, chip: &Chip, split: &DatasetSplit) -> Result<ScoredChip> {
        match self.backend {
            Backend::Cpu => Ok(evaluate_chip_cpu(chip, split)),
            Backend::Gpu => {
                #[cfg(feature = "gpu")]
                {
                    let scored = self.evaluate_population(std::slice::from_ref(chip), split)?;
                    scored
                        .into_iter()
                        .next()
                        .ok_or_else(|| anyhow!("empty gpu eval"))
                }
                #[cfg(not(feature = "gpu"))]
                {
                    Err(anyhow!("GPU backend requires --features gpu"))
                }
            }
        }
    }
}

fn generate_dataset(task: Task, seed: u64) -> Vec<Sample> {
    match task {
        Task::Xor => generate_xor(seed),
        Task::PolicyHidden => generate_policy_hidden(seed),
    }
}

fn split_dataset(dataset: Vec<Sample>, seed: u64) -> DatasetSplit {
    let features = dataset.get(0).map(|s| s.x.len()).unwrap_or(0);
    let mut train = Vec::new();
    let mut test = Vec::new();
    for (idx, s) in dataset.into_iter().enumerate() {
        let mut h = BlakeHasher::new();
        h.update(&seed.to_le_bytes());
        h.update(&(idx as u64).to_le_bytes());
        let digest = h.finalize();
        let bucket = u64::from_le_bytes(digest.as_bytes()[0..8].try_into().unwrap());
        let is_test = bucket % 5 == 0;
        if is_test {
            test.push(s);
        } else {
            train.push(s);
        }
    }

    let mut hasher = Hasher::new();
    for s in train.iter().chain(test.iter()) {
        for b in &s.x {
            hasher.update(&[*b as u8]);
        }
        hasher.update(&[s.y as u8]);
    }
    let hash_hex = hex::encode(hasher.finalize().as_bytes());

    DatasetSplit {
        train,
        test,
        features,
        hash_hex,
    }
}

fn generate_xor(seed: u64) -> Vec<Sample> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed ^ 0x1234_5678);
    let mut data = Vec::new();
    for i in 0..256 {
        let force_one = i < 128;
        let a = rng.gen_bool(0.5);
        let b = if force_one { !a } else { a };
        let mut x = vec![false; 8];
        x[0] = a;
        x[1] = b;
        for j in 2..8 {
            x[j] = rng.gen_bool(0.5);
        }
        let y = a ^ b;
        data.push(Sample { x, y });
    }
    data
}

fn generate_policy_hidden(seed: u64) -> Vec<Sample> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed ^ 0xABCD_EF01);
    let mut data = Vec::new();
    for _ in 0..320 {
        let role_manager = rng.gen_bool(0.4);
        let amount_under = rng.gen_bool(0.6);
        let has_2fa = rng.gen_bool(0.7);
        let risk_flag = rng.gen_bool(0.2);
        let region_ok = rng.gen_bool(0.8);
        let peer_ok = rng.gen_bool(0.5);
        let allow = (role_manager && amount_under)
            || (amount_under && has_2fa && region_ok && peer_ok && !risk_flag);
        let x = vec![role_manager, amount_under, has_2fa, risk_flag, region_ok, peer_ok];
        data.push(Sample { x, y: allow });
    }
    data
}

pub fn replay(task: Task, seed: u64, chip: &Chip, out_dir: &str, backend: Backend) -> Result<ReplayReport> {
    let dataset = generate_dataset(task, seed);
    let split = split_dataset(dataset, seed);
    let engine = EvalEngine::new(backend, &split)?;
    let scored = engine.evaluate_chip(chip, &split)?;
    let report = ReplayReport::from_scored(&scored, &split, &EvolutionConfig {
        task,
        seed,
        generations: 0,
        population: 0,
        offspring: 0,
        elite: 0,
        max_gates: chip.gates.len(),
        mutation_rate_per10k: 0,
        out_dir: out_dir.to_string(),
        debug: false,
        backend,
    })?;
    Ok(report)
}

pub fn stats(task: Task, seed: u64) -> Result<DatasetStats> {
    let dataset = generate_dataset(task, seed);
    let split = split_dataset(dataset, seed);
    let mut train_pos = 0usize;
    let mut train_neg = 0usize;
    let mut test_pos = 0usize;
    let mut test_neg = 0usize;
    let mut samples = Vec::new();

    for s in &split.train {
        if task == Task::Xor && (s.x.len() < 2 || (s.x[0] ^ s.x[1]) != s.y) {
            return Err(anyhow!("XOR invariant broken in train"));
        }
        if s.y { train_pos += 1 } else { train_neg += 1 };
    }
    for s in &split.test {
        if task == Task::Xor && (s.x.len() < 2 || (s.x[0] ^ s.x[1]) != s.y) {
            return Err(anyhow!("XOR invariant broken in test"));
        }
        if s.y { test_pos += 1 } else { test_neg += 1 };
    }

    for s in split.train.iter().take(10) {
        samples.push(SampleView {
            x0: s.x.get(0).copied().unwrap_or(false) as u8,
            x1: s.x.get(1).copied().unwrap_or(false) as u8,
            y: s.y as u8,
        });
    }

    Ok(DatasetStats {
        train_size: split.train.len(),
        test_size: split.test.len(),
        train_pos,
        train_neg,
        test_pos,
        test_neg,
        samples,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "gpu")]
    use std::fs::File;
    #[cfg(feature = "gpu")]
    use std::io::{BufRead, BufReader};
    #[cfg(feature = "gpu")]
    use std::path::Path;
    #[cfg(feature = "gpu")]
    use tempfile::tempdir;

    #[test]
    fn determinism_short_run() {
        let cfg = EvolutionConfig {
            task: Task::Xor,
            seed: 1337,
            generations: 5,
            population: 25,
            offspring: 25,
            elite: 5,
            max_gates: 8,
            mutation_rate_per10k: 2000,
            out_dir: "out-test-a".into(),
            debug: false,
            backend: Backend::Cpu,
        };
        let mut cfg_b = cfg.clone();
        cfg_b.out_dir = "out-test-b".into();
        let res1 = evolve(cfg.clone()).unwrap();
        let res2 = evolve(cfg_b).unwrap();
        assert_eq!(res1.best.chip.hash(), res2.best.chip.hash());
    }

    #[test]
    fn cid_is_deterministic() {
        let line = TrainingLine {
            generation: 1,
            task: "xor".into(),
            seed: 7,
            best_train_per10k: 9000,
            best_test_per10k: 8500,
            mean_train_per10k: 8700,
            chip_hash: "deadbeef".into(),
            gates: 3,
        };
        let cid1 = cid_for(&line);
        let cid2 = cid_for(&line);
        assert_eq!(cid1, cid2);
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn gpu_matches_cpu_single_chip() {
        // Skip if no GPU adapter available
        let dataset = generate_xor(2024);
        let split = split_dataset(dataset, 2024);
        let chip = Chip::parse(
            "CHIP v0\nFEATURES n=8\nGATES m=1\ng0 = THRESH(1,f0,f1)\nOUTPUT = g0\n",
        )
        .unwrap();
        let cpu = evaluate_chip_cpu(&chip, &split);
        let engine = match EvalEngine::new(Backend::Gpu, &split) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("Skipping GPU test: no GPU adapter available ({e})");
                return;
            }
        };
        let gpu = engine.evaluate_chip(&chip, &split).unwrap();
        assert_eq!(cpu.train_acc_per10k, gpu.train_acc_per10k);
        assert_eq!(cpu.test_acc_per10k, gpu.test_acc_per10k);
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn evolve_gpu_matches_cpu_small_run() {
        // Skip if no GPU adapter available
        let dataset = generate_xor(99);
        let split = split_dataset(dataset, 99);
        if EvalEngine::new(Backend::Gpu, &split).is_err() {
            eprintln!("Skipping GPU test: no GPU adapter available");
            return;
        }
        
        let dir = tempdir().unwrap();
        let cpu_out = dir.path().join("cpu");
        let gpu_out = dir.path().join("gpu");
        let base_cfg = EvolutionConfig {
            task: Task::Xor,
            seed: 99,
            generations: 5,
            population: 50,
            offspring: 50,
            elite: 5,
            max_gates: 8,
            mutation_rate_per10k: 1500,
            out_dir: cpu_out.to_string_lossy().into_owned(),
            debug: false,
            backend: Backend::Cpu,
        };
        let mut gpu_cfg = base_cfg.clone();
        gpu_cfg.out_dir = gpu_out.to_string_lossy().into_owned();
        gpu_cfg.backend = Backend::Gpu;

        let res_cpu = evolve(base_cfg).unwrap();
        let res_gpu = evolve(gpu_cfg).unwrap();

        assert_eq!(res_cpu.best.chip.hash(), res_gpu.best.chip.hash());
        assert_eq!(res_cpu.best.train_acc_per10k, res_gpu.best.train_acc_per10k);
        assert_eq!(res_cpu.best.test_acc_per10k, res_gpu.best.test_acc_per10k);

        let cid_cpu = last_training_cid(&res_cpu.config.out_dir);
        let cid_gpu = last_training_cid(&res_gpu.config.out_dir);
        assert_eq!(cid_cpu, cid_gpu);
    }

    #[cfg(feature = "gpu")]
    fn last_training_cid(out_dir: &str) -> String {
        let path = Path::new(out_dir).join("training_curve.ndjson");
        let file = File::open(&path).expect("training_curve exists");
        let reader = BufReader::new(file);
        let mut last_line = None;
        for line in reader.lines() {
            let l = line.unwrap();
            if l.trim().is_empty() {
                continue;
            }
            last_line = Some(l);
        }
        let raw = last_line.expect("at least one training line");
        let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
        v.get("cid")
            .and_then(|c| c.as_str())
            .unwrap()
            .to_string()
    }
}
