## Current working design

- Encoder: Qwen3-0.6B, frozen
- Adaptation: LoRA / RSLoRA, r=16, alpha=32
- Fragment latent representation: molecular formula, 64 fixed slots
- Dataset sharding: 64 shards

## Completed: NIST23 spectrum preprocessing (`spectra_v1`, `spectra_v2`)

Canonical source (never modified; SHA-256 of each recorded in every manifest):

- `metabo_data/data/spec_datasets/nist23/labels.tsv`
- `metabo_data/data/spec_datasets/nist23/spec_files.hdf5`

Built with `scripts/build_nist23_spectra.py`, output under
`metabo_data/processed/metabo_sllm/nist23/<split>/spectra_v1/`:

| split | parents | spectra | peaks |
|---|---:|---:|---:|
| `scaffold_sub_10` | 17,671 | 111,090 | 4,337,110 |
| `bmscaffold_1` (full) | 176,697 | 1,113,475 | 44,168,247 |

- One row per `parent_spec` x collision block; `spectrum_uid = {parent_spec}:{collision_index:02d}`.
- 64 Parquet shards per fold (train / valid / test), assigned by SHA-256(`parent_spec`)
  so all collision-energy siblings of a molecule stay in one shard.
- All verify gates pass on both splits: manifest/Parquet row counts agree, 0 duplicate
  `spectrum_uid`, 0 parents spanning folds, 64 shards per fold, 0 mzs/intensities length
  mismatches, 0 parse failures, 0 empty spectra, shard assignment and intra-shard ordering
  consistent. Diagnostic: 0 InChIKeys cross folds on the full split.

`spectra_v2` adds `mz_decimal_places`, which the candidate matcher needs to pick a mass
tolerance per peak.

## Completed: graph-free candidate generation and `fragment_supervision_v1`

Subformula candidates are enumerated from the precursor formula alone -- no molecular
graph, no MAGMa labels. One row per spectrum carries the model input, the full-spectrum
target, deduplicated candidates, peak-candidate edges, and a supervision mask
(a peak is supervised when it has at least one candidate and its m/z carries at least
two decimals). 64 slots, filled in a deterministic order: intensity descending, then m/z
ascending, then index.

## Completed: fragment latent model V0 and the training pipeline

Qwen3-0.6B + LoRA encoder, 64-slot transformer decoder, structured per-element formula
decoder, and ion / presence / intensity heads. Training uses a candidate-bag likelihood
(`-log sum_{z in B} p(z)`) with rectangular Hungarian matching on a detached cost, scored
in two passes so that no candidate is truncated. BF16, 4-GPU DDP, dynamic batching,
adapter-only checkpoints.

## Completed: pilot on `scaffold_sub_10` and the evaluation contract audit

A 10-epoch pilot (seed 0) is the current baseline. Its evaluation contract has been
audited and three things pinned down:

- **Intensity space.** The intensity head is trained on `sqrt(y)/||sqrt(y)||`, so its
  output is a square-root-space quantity; the reported cosine had compared it against raw
  observed intensities. Both spaces are now computed and named. On the pilot's best
  checkpoint over 10,988 valid spectra: `canonical_sqrt` cos@100 = **0.2182**,
  `legacy_raw` cos@100 = 0.1720. `canonical_sqrt` is primary and matches ms-pred.
- **Oracles.** The ceiling the formula supervision can express is **0.9725**
  (`formula_rendered`, `canonical_sqrt`); respecting the 64-slot budget as well gives
  0.9675. The earlier 0.9067 was the same kind of quantity read in the mismatched space.
- **Reproducibility.** Predicted formulas, ion states and active masks are identical
  across batch sizes, neighbours, order and a 4-GPU sharded rerun of the whole fold. Only
  the continuous intensity head moves, at ~2e-05.

Diagnosis: the model reaches 0.0502 bag mass and 40.7 % argmax-in-bag on *training* data,
so it is underfitting rather than overfitting, and low bag mass is not confined to
ambiguous bags -- unique bags are 78 % of pairs and still sit at 0.0159 / 24.9 %.
Difficulty tracks precursor mass (41 % argmax-in-bag below 200 Da, 15 % above 600 Da).

**Test metrics are sealed.** Nothing has been measured on any test fold; the test shards
exist only as held-out data.

### Next

A longer, larger-budget run on the same `scaffold_sub_10` subset, to establish how far
this configuration converges before spending the full `bmscaffold_1` split on it.
