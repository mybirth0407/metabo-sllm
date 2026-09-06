## Current working design

- Encoder: Qwen3-0.6B
- Adaptation: LoRA / RSLoRA
- Fragment latent representation: molecular formula
- Dataset sharding: 64 shards

## Completed: NIST23 spectrum preprocessing (`spectra_v1`)

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

**Test metrics are sealed.** Nothing has been measured on any test fold; the test shards
exist only as held-out data.

### Not yet implemented

Fragment candidate generation, model code, and training.
