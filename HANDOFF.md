# Handoff — metabo-sllm, 2026-09-07

Graph-free mass-spectrum prediction: Qwen3-0.6B + LoRA encodes a molecule as text,
a 64-slot decoder emits fragment **formulas** (not graph substructures), and the
spectrum is rendered from those formulas. No molecular graph encoder, no MAGMa
labels, no hard candidate selection, no full fine-tuning of the backbone, and the
`test` fold has never been read.

## 0. Read this first: two training defects, fixed in commit 876aa41

Every multi-GPU run before 2026-09-07 ~10:30 UTC had two silent defects.
**Every in-run validation number logged by those runs is a mean over four
different models**, and every checkpoint is one of the four (rank 0's). Numbers
in this file from those runs are labelled; the reliable number for a saved
model is the standalone one (`scripts/evaluate_fragment_model.py`).

1. **Ranks never synchronised.** `Trainer._optimizer_step` called the bare
   module, not the `DistributedDataParallel` wrapper, so gradients were never
   all-reduced: each rank trained on its own quarter of the data. Found by
   noticing a checkpoint did not reproduce its own logged validation; proven by
   splitting the in-run prediction parquet by rank (rows are gathered in rank
   order): rank 0's quarter agreed with the checkpoint 100 %, the other three
   0–2 %. On the V1 subset run the four ranks scored 0.350 / 0.098 / 0.345 /
   0.210; the logged 0.2505 was their mean; **the saved rank-0 model scores
   0.3481 on the whole fold.** Now the forward goes through the wrapper and
   `save()` refuses to write until every rank's trainable parameters equal
   rank 0's.
2. **The formula decoder's linear weights never received a gradient.** The
   two-pass scorer runs the decoder under `no_grad` (matching cost) and then
   with gradients (matched pairs) inside one bf16 autocast context; autocast
   caches the bf16 copy of each fp32 weight, and a copy first made under
   `no_grad` carries no autograd history, so the gradient pass reused it.
   LayerNorms and embeddings stay in fp32 and did train, which kept the loss
   moving. Confirmed in isolation and pinned by `tests/test_autocast_two_pass.py`;
   the trainer's autocast now runs with `cache_enabled=False`. DDP itself
   caught this the moment ranks were synchronised ("parameters that were not
   used in producing loss": every decoder linear and the count head).

Consequence: every conclusion below about *formula identity being the
bottleneck* was measured on models whose decoder body and count head sat at
initialisation. The measurements are real; their cause was this, not the
architecture. Re-measure on a post-fix checkpoint before redesigning anything.

## 1. Where things stand

Two subset runs exist on `scaffold_sub_10` (train 88,743 / valid 10,988
spectra), both pre-fix. In-run numbers are four-model means (see §0).

| valid cos@100 | pilot V0 (ep 9) | V1 Base (ep 27), in-run mean | **V1 rank-0 checkpoint, standalone** |
|---|---|---|---|
| `canonical_sqrt` (primary) | 0.2182 (standalone ≈ in-run) | 0.2505 | **0.3481** |
| `legacy_raw` (secondary) | 0.1720 | 0.2028 | **0.2957** |
| fragment bag_hit | 0.1980 | 0.2194 | **0.2842** |
| wall clock, 4x B200 | 0.65 h | 3.82 h | — |

The V0 pilot's four ranks happened to land close together (standalone 0.17201
vs in-run 0.17229 in `legacy_raw`), which is why the defect went unnoticed then.

**V1 has converged on this subset.** The four-epoch gain decayed 0.0160 → 0.0102
→ 0.0036 and then went negative: epoch 27 = 0.2505, epoch 30 = 0.2463, epoch 31 =
0.2471. More epochs on this data will not help.

**The first full-split run was stopped after four epochs and continued under a
different regime.** `bmscaffold_1/qwen_formula_slots_v1_full_seed0` (V1 config,
lr 1.5e-4, 8 epochs planned, `fragment_supervision_v2` bags) peaked at epoch 0 —
canonical cos@100 **0.1364** on the full 109,817-spectrum valid fold — then fell
to 0.1062 / 0.1111 / 0.1193 at epochs 1-3 while train kept improving and
gradients stayed calm (gnorm 0.7). Between epochs 0 and 1 the presence head
went from 1,052 zero-prediction spectra to none, predicted peaks per spectrum
rose 35.9 → 41.1, and active-slot precision collapsed 0.094 → 0.062. Ruled out:
fold difficulty (same precursor-mass distribution as the subset; 42.6 % of valid
formulas appear in train vs 11.4 % on the subset — the subset's 0.2505 was
therefore mostly on *unseen* formulas), supervision density (28-31 candidates
per spectrum, same as the subset), slot duplicates (lower than the subset). The
reading: ten times the subset's steps at peak lr drove the presence head open.
Throughput was 348 spectra/s, 58 % above the subset's.

A warm-start continuation of that run (`..._full_warm_seed0`, lr 8e-5, presence
0.5) was started and then stopped once the defects were found: its "presence
collapse" diagnosis was itself an artefact of four diverging ranks.

**The first post-fix run — the current best model:**
`bmscaffold_1/qwen_formula_slots_v1_full_fixed_seed0`, config
`configs/train/qwen_formula_slots_v1_full_fixed.yaml`. Warm-started from the
V1 rank-0 checkpoint (0.3418 on the full valid fold at step 0), lr 1.5e-4 at a
real effective batch of 256, presence 0.2. Planned for 8 epochs; stopped by
request after epoch 6 (2026-09-08 ~16:25 UTC). `best` = `last` = epoch 6,
step 24,919. Valid canonical cos@100 by epoch: 0.443, 0.464, 0.483, 0.502,
0.508, 0.510, **0.515** — 53 % of the 0.9725 ceiling (the full valid fold's
oracles equal the subset's: formula_rendered 0.9725, slot_capacity 0.9684).
Train bag NLL 1.79 → 0.83 and train bag_hit 0.52 → 0.70 over the same epochs
while valid bag_hit went 0.33 → 0.385 → 0.365 → 0.388: the train/valid gap on
identity is the live question, with the caveat that lr was already decaying
when valid flattened. Predictions for every epoch are in
`predictions/valid_epochNN.parquet` (presence values included); raising the
presence threshold above 0.5 only hurts (0.6 → 0.501, 0.9 → 0.423).

Post-fix identity errors (valid, 2.75 M active slots, epoch 6): hit 54.0 %
(pre-fix 28.3 %); of the misses, one heavy atom off 22.9 %, two 9.9 %,
hydrogen-only 5.0 %, far 8.3 % (pre-fix 29.1 %); median miss sits 2.0 Da from
the nearest observed peak. Valence violations are 0 % on both sides (v2 bags);
97.5 % of miss formulas occur ≥20 times in train. Per-peak bag sizes among
supervised peaks: 80.7 % singletons (83.5 % of intensity), 14.2 % of size 2–3,
5.1 % larger — so bag-ambiguity objectives (hard-EM, RC/PRODEN) can touch at
most ~5 % of peaks. The residual is discrimination among near formulas on
singleton bags, not ambiguity resolution.

Stratified (epoch 6, canonical cos@100): precursor formula **seen in train
0.599 (42.6 % of valid) vs unseen 0.453 (57.4 %)** — formula novelty alone
accounts for a 0.146 gap, the generalization signature RQ1 asked for. By
precursor m/z: 0.638 (<200), 0.578 (200–300), 0.501 (300–400), 0.460
(400–600), 0.463 (≥600). Distribution is wide: median 0.549, p10 0.068,
p90 0.889.

Free-running prediction on a 4,000-spectrum **train** sample with the same
checkpoint: canonical cos@100 **0.808** (83 % of the ceiling), bag_hit 0.613,
active-slot precision 0.635, unique-bag hit 0.671 — against 0.515 / 0.388 /
0.47 / 0.451 on valid. The dominant residual is generalization, not capacity
or the objective.

Counterfactuals on the post-fix best (canonical cos@100): all_predicted 0.5151;
oracle ion 0.5152 (no effect); presence forced on matched 0.5398; oracle
identity with model weights 0.5784; model identity with oracle weights
**0.6229**; both 0.9688. Unlike the pre-fix pilot, the weight component
(presence × intensity) is now the larger single lever, and the two errors are
entangled (fixing one leaves ~0.4 on the table). The full research report with
the ranked experiment plan is
`docs/research/performance_improvement_report_2026-09-08.md`.

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

- **A checkpoint must reproduce its logged validation.** Check it on every new
  run's first checkpoint (`evaluate_fragment_model.py` on `checkpoints/last`
  vs the `[valid]` line). Two defects hid behind that gap for the whole
  project (§0). `save()` now asserts rank agreement; the autocast cache is off.
- **Under bf16 autocast, never run a module under `no_grad` before its
  gradient pass in the same context** unless the cache is disabled — the
  cached casts carry no history.
- **Call the DDP wrapper, not `.module`,** for any forward whose backward must
  synchronise.
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
`.claude/worktrees/training-pipeline-v0`), twenty commits ahead of `main` (this handoff's own commit will make it twenty):

```
272db1e Add performance-improvement research report; record post-fix counterfactuals in HANDOFF
48156ca handoff: post-fix run outcome, identity residuals, bag sizes, stratified cosine
b5ce59f add the Phase 2 literature packs for the performance research question
882cf31 handoff: the two training defects, corrected numbers, and the post-fix run
c3b8a4d add the first full-split config to run with the training defects fixed
876aa41 fix two silent training defects: unsynchronised ranks, and a decoder that never learned
a704707 record the first full-split run's collapse and the warm-start continuation
0a68bad warm-start a run from another's weights, and keep every epoch's predictions
0624abe record the prefix-tree result, the v2 bags, and the running full-split job
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
