# Handoff — metabo-sllm, 2026-09-07

Graph-free mass-spectrum prediction: Qwen3-0.6B + LoRA encodes a molecule as text,
a 64-slot decoder emits fragment **formulas** (not graph substructures), and the
spectrum is rendered from those formulas. No molecular graph encoder, no MAGMa
labels, no hard candidate selection, no full fine-tuning of the backbone, and the
`test` fold has never been read.

## 1. Where things stand

Two training runs exist, both on `scaffold_sub_10` (train 88,743 / valid 10,988
spectra). All numbers are on the full valid fold.

| valid cos@100 | pilot V0 (ep 9) | **V1 Base (ep 27)** | change |
|---|---|---|---|
| `canonical_sqrt` (primary) | 0.2182 | **0.2505** | +14.8 % |
| `canonical_sqrt` cos@20 | 0.1976 | **0.2271** | +14.9 % |
| `legacy_raw` (secondary) | 0.1720 | **0.2028** | +17.9 % |
| fragment bag_hit | 0.1980 | **0.2194** | +10.8 % |
| wall clock, 4x B200 | 0.65 h | 3.82 h | 5.9x |

**V1 has converged on this subset.** The four-epoch gain decayed 0.0160 → 0.0102
→ 0.0036 and then went negative: epoch 27 = 0.2505, epoch 30 = 0.2463, epoch 31 =
0.2471. More epochs on this data will not help.

**A third run is in progress:** `bmscaffold_1/qwen_formula_slots_v1_full_seed0` —
the V1 configuration on the full split (train 891,038 spectra, 10x), 8 epochs,
~27,800 optimizer steps, `fragment_supervision_v2` bags, launched 2026-09-07
06:34 UTC, ~9.4 h. Log: `$CLAUDE_JOB_DIR/tmp/full_train.txt`; stop with
`pkill -f qwen_formula_slots_v1_full`.

**One experiment did not work and should not be repeated as designed.** A
set-level identity term — SCARF's prefix-tree objective in marginal form,
conditioned on a pooled molecule vector and sharing the slot decoder's
parameters (`losses/prefix_loss.py`, `configs/train/qwen_formula_slots_v1_prefix_subset.yaml`)
— was behind V1 on valid at every epoch and behind on *train* by epoch 7
(valid 0.1351 vs 0.1680; train bag NLL 3.69 vs 3.39). The molecule-conditioned
marginal (broad, "every fragment of this molecule") and the slot-conditioned
decoding (peaked, "this slot's one fragment") interfere on one decoder body and
head. Withdrawing the weight (0.9/epoch) did not recover it. The code stays,
defaults to off, and is byte-for-byte the old objective at weight 0. What the
failure supports: bolting set-level supervision onto per-slot generation does
not transfer; the per-slot generation itself is what to replace.

Run directories (under
`/NHNHOME/26moe001_B/BASE/metabo_data/results/metabo_sllm/nist23/scaffold_sub_10/`):

- `qwen_formula_slots_v0_seed0/` — pilot, 10 epochs, `diagnostics/` holds the
  error decomposition and the evaluation contract audit.
- `qwen_formula_slots_v1_base_seed0/` — V1, 32 epochs, best at epoch 27 /
  step 9,884, `predictions/valid_best.parquet`.

## 2. What has been established by measurement

Do not re-litigate these; they are measured, not assumed.

**The metric.** The cosine is computed in two named spaces and both are always
reported. `canonical_sqrt` is primary: the intensity head is trained on
`sqrt(y)/||sqrt(y)||`, and ms-pred applies its square root once at preprocessing
(`common/misc_utils.py:2207-2210`) and none at evaluation (`norm_spectrum` is dead
code; every call site is commented out). The old single number, 0.1723, compared a
square-root prediction against raw observations and is kept as `legacy_raw`.
`summarise_scores` emits no bare `cos@K` key, so a pre-change metrics file is
distinguishable from a post-change one at a glance.

**The ceiling.** What the formula supervision can express is `formula_rendered` =
**0.9725** (`canonical_sqrt`); adding the 64-slot budget gives 0.9675, so the slot
cap costs only ~0.005. `peak_copy` = 0.9911 is a sanity check, not a ceiling — the
0.9 % shortfall is the metric's own top-100 trim against an untrimmed observation.

**Identity is the bottleneck, not intensity, and not ion state.** Counterfactuals
in `canonical_sqrt`, pilot checkpoint:

| condition | cos@100 |
|---|---|
| `all_predicted` | 0.2182 |
| `oracle_identity_predicted_weights` | **0.4260** |
| `predicted_identity_oracle_weights` | 0.3023 |
| `predicted_formula_oracle_ion` | 0.2182 (ion state changes nothing) |
| `oracle_identity_sqrt_weights` | 0.9686 |

**Search is not the bottleneck — the scoring function is.** Beam decoding gave
`legacy_raw` 0.1720 (beam 1) → 0.1744 (beam 4) → 0.1744 (beam 16). Widening the
search buys 0.0024. The model's distribution over formulas is what is wrong.

**Low bag mass is not an ambiguity problem.** Unique bags (one member, no ambiguity
possible) are 268,355 of 342,433 valid pairs and still sit at 0.0159 geometric-mean
mass with 24.9 % argmax-in-bag. Ambiguous bags are worse (0.0027 / 5.5 %) and decay
monotonically with cardinality, but they are not the origin.

**Difficulty tracks molecule size, not peak count.** Argmax-in-bag by precursor m/z:
41 % below 200 Da, 30 % at 200-300, 18 % at 300-400, 16 % at 400-600, 15 % above
600. Stratified by target count it is nearly flat (0.19-0.23).

**Inference is reproducible.** Predicted formulas, ion states and active masks are
identical across batch sizes 1/8/16/32, neighbour swaps and order reversal under
both ragged and canonical padding (n=1024), and identical across a 4-GPU sharded
rerun of all 10,988 valid spectra versus the original single-GPU run. Only the
continuous intensity head moves, ~2e-05, from batched GEMM reduction order.

## 3. Data

Canonical source, never modified; SHA-256 of each file is in every manifest:
`/NHNHOME/26moe001_B/BASE/metabo_data/data/spec_datasets/nist23/{labels.tsv,spec_files.hdf5,splits/}`.

Processed, under `metabo_data/processed/metabo_sllm/nist23/<split>/`:

| split | artifact | train | valid | note |
|---|---|---|---|---|
| `scaffold_sub_10` | `spectra_v1`, `spectra_v2`, `formula_support_audit_v0`, `fragment_supervision_v1`, `fragment_supervision_v2` | 88,743 | 10,988 | V0/V1 used v1 bags |
| `bmscaffold_1` | `spectra_v1`, `spectra_v2`, `fragment_supervision_v1`, `fragment_supervision_v2` | **891,038** | 109,817 | the full run uses v2 |

`fragment_supervision_v2` = v1 minus candidates that fail the two hard valence
bounds (RDBE ≥ 0; monovalent atoms ≤ 2(C+Si)+2+(N+P)). Same columns; the reader
accepts both. On the subset it drops 346,794 of 5,029,102 candidates (6.9 %) and
costs the V1 checkpoint 5 of 91,236 bag hits; on the full split 4,055,150 of
52,906,460 (7.7 %). Softer rules (fragment RDBE ≤ precursor + 1) were measured
and *not* applied: they also exclude 4.6 % of formulas real hits take. The
`test` fold is deliberately not built for supervision.

Backbones, both local-only (`local_files_only=True`, never downloads):
`models/qwen3-0.6b` (post-trained, used by V0) and `models/qwen3-0.6b-base`
(revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, used by V1). Each has a
manifest with a SHA-256 per file.

## 4. Code map

- `src/metabo_sllm/data/` — `ms_parser` (contract-first `.ms` parsing), `sharding`
  (SHA-256 of `parent_spec`, the single source of truth), `supervision`
  (64 slots, deterministic ordering), `fragment_dataset`, `fragment_collator`
  (`transform_intensities` = `sqrt(y)/||sqrt(y)||`).
- `src/metabo_sllm/chem/` — `formula` (Hill, `SubformulaTable`), `candidates`
  (ion channels, precision-aware tolerance, `generate_candidates`).
- `src/metabo_sllm/model/` — `qwen_encoder`, `slot_decoder` (`query_init_std=1.0`
  is load-bearing; 0.02 collapses all 64 slots), `formula_decoder` (**dropout must
  stay 0.0** or the two scoring passes disagree), `heads`, `fragment_latent_model`.
- `src/metabo_sllm/losses/` — `candidate_scoring` (two-pass: no-grad matching cost,
  then gradients only on matched pairs; `pair_index` is **global across the batch**),
  `matching` (rectangular Hungarian on a detached cost), `fragment_losses`.
- `src/metabo_sllm/evaluation/` — `spectrum_metrics` (`PRIMARY_SPACE`,
  `EVALUATION_SPACES`, nested `cosine[space][k]`), `inference` (leakage guards),
  `error_decomposition`, `contract_audit`, `prediction_writer`.
- `scripts/` — `build_nist23_spectra.py`, `build_fragment_supervision.py`
  (`--workers`, shard-parallel), `audit_formula_support.py`, `correctness_pass.py`,
  `smoke_fragment_model.py`, `train_fragment_model.py`, `run_pilot.py`,
  `evaluate_fragment_model.py`, `decompose_errors.py`,
  `audit_evaluation_contract.py`.
- `configs/model/qwen_formula_slots_v{0,1}.yaml`,
  `configs/train/qwen_formula_slots_v0_{smoke,pilot}.yaml`,
  `configs/train/qwen_formula_slots_v1_subset.yaml`.

Tests: 338 passing (`PYTHONPATH=src python3 -m pytest tests/ -q`).

## 5. Reference harnesses

- **ms-pred** (`https://github.com/coleygroup/ms-pred`), cloned at
  `$CLAUDE_JOB_DIR/tmp/ms-pred`. Contains ICEBERG, SCARF, GLACIER, MassFormer,
  GrAFF-MS, MARASON. Consult it for undetermined design decisions rather than
  guessing. Two designs matter here:
  - **GrAFF-MS** (`graff_ms/graff_ms_model.py:118-275`): a fixed vocabulary of
    formulas, each flagged fragment or neutral loss (`is_loss`); one score per
    vocabulary entry from a pooled molecule embedding; loss entries materialised as
    `precursor - formula`; then masked to what is valid for this precursor. No
    generation at all.
  - **SCARF** (`scarf_pred/scarf_model.py:254-595`): a prefix tree over the
    subformula lattice, expanded one element at a time under the remaining-supply
    constraint, keeping the top-k prefixes alive per level. Candidates compete
    within a molecule.
- **`docs/papers/`** — 11 PDFs (ICEBERG, GLACIER, SCARF, MARASON, MassFormer,
  GrAFF-MS, CFM-ID, 3DMolMS, NEIMS, DeepMet, LLaMA-vs-GPT). ICEBERG p.17 states the
  square-root intensity convention in prose. Untracked in git (30 MB).

## 6. Open work, in the order proposed

1. **Full-data training on `bmscaffold_1`** — running (see §1). Read its result
   against the subset's 0.2505: it says what ten times the data buys at fixed
   architecture.
2. **Replace per-slot formula generation with scoring over an enumerated candidate
   set.** This is the change that attacks the measured bottleneck, and the prefix
   experiment's failure points here too. The precursor formula is a *model input*,
   so enumerating its subformulas at inference is not leakage, and
   `chem/candidates.generate_candidates` already does the enumeration. The
   evidence: GrAFF-MS scores a fixed vocabulary (10k products + losses, 98 % ion
   coverage on NIST20) with one softmax and reaches 0.658 scaffold; SCARF's
   FixedVocab baseline (5k) reaches 0.704/0.658 against SCARF's own 0.726/0.669;
   and in our own diagnostic 80 % of V1's miss formulas occur ≥20 times in train,
   so a 20-30k vocabulary covers 98.4 % of hits. Shape it as: candidates =
   (vocab ∩ subformulas(precursor)) ∪ (precursor − loss-vocab), valence-filtered;
   slots attend over candidates with a softmax so they compete; positions are
   always candidate masses; intensity is weighted per candidate. Keep the two-pass
   scorer's memory discipline.
3. **Cross-slot competition.** 9 % of active slots currently decode to a duplicate
   identity; slots do not compete.
4. **Joint modelling of a molecule's collision-energy siblings.** They already share
   a shard, so the plumbing is cheap; today each (molecule, CE) is independent.
5. **Molecular representation.** A SMILES string through a frozen general-purpose LM
   is weak. A chemistry-pretrained LM is in scope; Morgan fingerprints are
   graph-derived and would need an explicit decision against the graph-free rule.

## 7. Constraints that must not be violated

- The `test` fold is never read, in any form. `evaluate_fragment_model.py` refuses
  it; `prediction_writer` refuses it; the supervision builder does not build it.
- Graph-free: no molecular graph encoder, no MAGMa labels.
- No full fine-tuning of the Qwen backbone; LoRA only, backbone frozen.
- No candidate hard cap, bag truncation, or subsampling in the scorer.
- `formula_decoder` dropout stays 0.0.
- Predictions are produced from conditioning fields alone, before any label is
  touched; `assert_no_leakage` runs on every batch.
- Original data files are never modified. Artifacts are written to a tmp path and
  renamed; an existing output path fails unless `--overwrite`.

## 8. Gotchas that have already cost time

- **Pin BLAS threads.** On this 72-core box an unpinned thread pool made a single
  15,000-bin cosine take 221 ms instead of ~0.5 ms — a 450x penalty that would have
  turned a 2-minute audit into 43 hours. Set `OMP_NUM_THREADS=1` (and MKL /
  OPENBLAS / NUMEXPR) *before* importing numpy, and get parallelism from processes.
- **Never point a smoke run at a real run's `output_dir`.** Metrics are appended, so
  smoke rows land in `train_metrics.jsonl` / `valid_metrics.jsonl`. The V1 files are
  contaminated this way; filter on `spectra == 10988` to recover the real rows.
- `torchrun` picks up the system Python; use `python3 -m torch.distributed.run`.
- cuDNN SDPA fails under bf16 autocast here; `configure_attention_backends()`
  disables it.
- peft's torchao probe fails against the installed torchao;
  `neutralise_torchao_probe()` works around it without changing the environment.
- The git stash stack is shared across worktrees. Use a WIP commit instead.

## 9. Repository state

Branch `worktree-training-pipeline-v0` (worktree at
`.claude/worktrees/training-pipeline-v0`), eleven commits ahead of `main`:

```
696ac88 add the V1 full-split training config
ff3e0c2 supervise formula identity at the prefix-tree level, on cleaner bags
9112615 add an identity-error diagnostic: near or far, valence, vocabulary
528a6a2 add a handoff covering what is measured, what is open, and what not to redo
6b53bb1 build supervision shards in parallel, and add the V1 training setup
5978c9e bring PROJECT_STATE up to date through the evaluation contract audit
5810ac1 report the cosine in both intensity spaces, and name them
2558dbc audit the evaluation contract: intensity space, oracles, reproducibility
d36ed2a add candidate-free error decomposition for the pilot checkpoint
f412238 add distributed training and validation pipeline
```

**`main` and `origin/main` are both still at `53f013e`.** The fast-forward has not
happened: `git push origin HEAD:main` was refused by this session's permission
classifier and needs a human to run it. `docs/` (30 MB of reference PDFs) is
untracked and deliberately not committed.
