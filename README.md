# chip_as_code

Deterministic Rust proof that (1) the TDLN Gate is a No-Guess constitutional checkpoint and (2) we can “train” an IA chip using discrete evolution over a text DNA. The build is fully offline (no LLM calls), uses danvoulez crates (`logline`, `ubl-ledger`, `json_atomic`, `tdln-gate` via `logline`), and keeps the learning core integer-only.

## What’s here
- **GateBox**: Intent + evidence + policy chip → Commit / Ghost / Reject. Missing evidence is always demoted to **Ghost** with exactly one question; unanchored evidence or policy failure **Rejects** with no ledger append. Commits/Ghosts land in `out/ledger.ndjson` as canonical NDJSON using `ubl-ledger` + `json_atomic` CIDs.
- **EvoChip**: A boolean/threshold semantic transistor circuit defined in `chip.txt` format. Evolution uses integer fitness (per10k accuracy), seeded ChaCha20 RNG, and deterministic datasets to “learn” XOR or a hidden policy gate—no gradients involved. Artifacts land in `out/`.

## 30-second demo (repeatable)
```bash
cargo run --release -- demo
cargo run --release -- verify --ledger out/ledger.ndjson
cargo run --release -- evolve --task xor --seed 1337 --generations 100
cargo run --release -- replay --task xor --seed 1337 --chip out/best_chip.txt
cargo run --release -- stats --task xor --seed 1337
cargo run --release -- replay --task xor --seed 1337 --chip examples/xor_golden.chip
```

## Commands
- `chip demo` — runs 4 scenarios from `examples/` (commit, ghost, unanchored reject, policy reject) and writes `out/ledger.ndjson`.
- `chip gate --intent ... --evidence ... --policy ... --ledger ...` — single GateBox decision.
- `chip verify --ledger ...` — recomputes canonical bytes + CID for every NDJSON line (parallel) and fails on mismatch.
- `chip evolve --task xor|policy_hidden --seed <u64> [--generations 200 --population 2000 --offspring 2000 --elite 50 --max-gates 32 --mutation_rate_per10k 1500]` — discrete evolution. Produces:
  - `out/best_chip.txt` (canonical DNA + BLAKE3 hash)
  - `out/training_curve.ndjson` (per-gen metrics)
  - `out/lineage.ndjson` (offspring lineage)
  - `out/replay_report.json` (deterministic replay check)
  - `out/dataset.ndjson` (transparency)
  - optional `--debug` prints per-gen uniqueness and whether best chip depends on f0/f1.
- `chip replay --task ... --seed ... --chip out/best_chip.txt` — re-evaluates a chip against the deterministic dataset split to prove replayability.
- `chip stats --task ... --seed ...` — shows dataset balance, XOR invariant check, and sample rows.

## DNA format (chip.txt)
```
CHIP v0
FEATURES n=<N>
GATES m=<M>
g0 = AND(f0,f1,...)
g1 = THRESH(2,g0,f2,...)
...
OUTPUT = gK|fK
```
Ops: `AND`, `OR`, `NOT`, `THRESH(k, ...)`. References only point backward (DAG). Canonical text is stable; identity is `blake3(canonical_bytes)`.

## GateBox rules (No-Guess Law)
1) Missing supporting evidence → `Ghost` with exactly one deterministic question; line written to ledger as `kind="ghost"`.
2) Unanchored evidence → `Reject` (no ledger append).
3) Policy check fail (chip output false) → `Reject`.
4) Otherwise → `Commit` with gate proof + evidence CIDs appended to ledger.

## EvoChip datasets
- **xor**: 8-bit inputs with distractors, label = `x0 XOR x1`. Deterministic hash split (~80/20) with balanced labels.
- **policy_hidden**: Synthetic policy tuples (role/amount/2FA/region/risk). Fitness = agreement between candidate policy chip and hidden rule. Train/test split is deterministic (hash), evaluation is integer accuracy per10k.

Identity: chip hashes are BLAKE3(JSON✯Atomic canon of the parsed AST). Training artifacts (`training_curve.ndjson`, `lineage.ndjson`) include a CID field computed the same way for each record. Whitespace in `.chip` files does not affect hashes.

`examples/xor_golden.chip` is a hand-crafted XOR circuit; `chip replay --task xor --seed 1337 --chip examples/xor_golden.chip` should score ~1.0 train/test, proving parser and evaluator correctness.

## Determinism
- RNG: `rand_chacha::ChaCha20Rng` with explicit `--seed`.
- No floats in the learning core; fitness uses integer per10k accuracy, floats only for reporting.
- Same seed + same command → identical artifacts (`best_chip.txt` hash, metrics, ledger CIDs).

## Running tests
```
cargo test
```
Includes unit tests for DNA parse/eval, evolution determinism, and gate ghost handling.
