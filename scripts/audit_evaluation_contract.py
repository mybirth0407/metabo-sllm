#!/usr/bin/env python3
"""Evaluation Contract Audit v1. Diagnostic only; nothing is trained or changed.

Sections:
  oracles        four separately-named ceilings, in every intensity space
  canonical      reproducibility of inference under different batching
  counterfactual the error decomposition, recomputed per intensity space
  bags           bag-mass statistics on train and valid through one code path

The test fold is never opened.

Every sweep over the fold is sharded: the model-free oracle sweep across CPU
processes, the model sections across the visible GPUs.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(_ROOT / "src"), str(_ROOT / "scripts")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

# One BLAS thread per process, set before numpy is imported.  A 15,000-bin
# cosine is a few microseconds of arithmetic, but on a 72-core box each call
# was spinning up a thread pool and taking 221 ms -- 450x longer than the work
# itself, which would have turned this sweep into a 43-hour run.  Parallelism
# here comes from processes, which the machine can actually schedule.
for _threads in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_threads, "1")

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from metabo_sllm.data.fragment_collator import FragmentCollator  # noqa: E402
from metabo_sllm.data.fragment_dataset import FragmentSupervisionDataset  # noqa: E402
from metabo_sllm.evaluation.contract_audit import (  # noqa: E402
    INTENSITY_SPACES,
    build_oracles,
    canonical_model_inputs,
    cosine_in_space,
)
from metabo_sllm.evaluation.error_decomposition import CONDITIONS, analyse_batch  # noqa: E402
from metabo_sllm.evaluation.inference import (  # noqa: E402
    MODEL_INPUT_FIELDS,
    assert_no_leakage,
    build_model_inputs,
    predict_spectrum,
)
from metabo_sllm.evaluation.spectrum_metrics import BinningConfig  # noqa: E402
from metabo_sllm.losses.candidate_scoring import candidate_pairs_for_assignment  # noqa: E402
from metabo_sllm.losses.matching import hungarian_assign  # noqa: E402
from metabo_sllm.model.qwen_encoder import load_tokenizer  # noqa: E402
from metabo_sllm.training.checkpoint import load_checkpoint  # noqa: E402
from metabo_sllm.training.trainer import configure_attention_backends  # noqa: E402
from smoke_fragment_model import build as build_model_parts, seed_everything  # noqa: E402

FORBIDDEN_FOLD = "test"
TOP_K = (20, 100)


def stats(values) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {}
    return {
        "mean": float(array.mean()),
        "median": float(np.percentile(array, 50)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
    }


def load_model(config, device):
    model, _, _ = build_model_parts(config.model)
    model.to(device)
    return model


# ------------------------------------------------------------------ sharding


def chunks(size: int, parts: int) -> list[tuple[int, int]]:
    """Contiguous index ranges, so a worker's rows stay in dataset order."""
    if size <= 0:
        return []
    parts = max(1, min(parts, size))
    edges = np.linspace(0, size, parts + 1).round().astype(int).tolist()
    return [(a, b) for a, b in zip(edges[:-1], edges[1:], strict=True) if b > a]


def run_shards(function, payloads, workers: int):
    """Run ``function`` over ``payloads``, in processes unless asked not to.

    Spawned rather than forked: the parent may already hold a CUDA context, and
    forking one is undefined.
    """
    payloads = list(payloads)
    if workers <= 1 or len(payloads) <= 1:
        return [function(payload) for payload in payloads]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(payloads), mp_context=context) as pool:
        return list(pool.map(function, payloads))


def merge_cosines(parts) -> dict:
    merged: dict = {}
    for part in parts:
        for key, per_k in part.items():
            target = merged.setdefault(key, {})
            for k, values in per_k.items():
                target.setdefault(k, []).extend(values)
    return merged


def summarise_cosines(cosine: dict) -> dict:
    return {
        key: {f"cos@{k}": stats(values) for k, values in per_k.items()}
        for key, per_k in cosine.items()
    }


# ------------------------------------------------------------------- oracles


ORACLE_SHAPE_KEYS = (
    "total_peaks",
    "representable_peaks",
    "supervised_peaks",
    "rendered_offset_da",
)


def collect_oracles(payload) -> tuple[dict, dict]:
    """One worker's slice of the oracle sweep. Model-free, so it needs no GPU."""
    root, fold, binning, start, stop = payload
    dataset = FragmentSupervisionDataset(root, fold)
    cosine: dict = {}
    shape: dict = {key: [] for key in ORACLE_SHAPE_KEYS}
    for index in range(start, stop):
        oracle = build_oracles(dataset[index])
        observed_mz, observed_intensity = oracle.observed
        for name, (mz, intensity) in (
            ("peak_copy_oracle", oracle.peak_copy),
            ("support_mask_oracle", oracle.support_mask),
            ("formula_rendered_oracle", oracle.formula_rendered),
            ("slot_capacity_oracle", oracle.slot_capacity),
        ):
            for space in INTENSITY_SPACES:
                for k in TOP_K:
                    cosine.setdefault(f"{name}|{space}", {}).setdefault(k, []).append(
                        cosine_in_space(
                            mz, intensity, observed_mz, observed_intensity,
                            space=space, k=k, config=binning,
                        )
                    )
        shape["total_peaks"].append(oracle.total_peaks)
        shape["representable_peaks"].append(oracle.representable_peaks)
        shape["supervised_peaks"].append(oracle.supervised_peaks)
        shape["rendered_offset_da"].append(oracle.rendered_offset_da)
    return cosine, shape


def run_oracles(root, fold, binning, *, size: int, workers: int) -> dict:
    """Four ceilings, model-free. Only ``peak_copy`` reaches 1.0, by construction."""
    payloads = [(root, fold, binning, start, stop) for start, stop in chunks(size, workers)]
    parts = run_shards(collect_oracles, payloads, workers)

    shape: dict = {key: [] for key in ORACLE_SHAPE_KEYS}
    for _, chunk_shape in parts:
        for key, values in chunk_shape.items():
            shape[key].extend(values)

    return {
        "cosine": summarise_cosines(merge_cosines(cosine for cosine, _ in parts)),
        "shape": {key: stats(values) for key, values in shape.items()},
        "workers": len(payloads),
        "note": (
            "Oracle peak lists carry sqrt(observed intensity), the space the intensity "
            "head is trained to emit, so an oracle is read in the same space as a "
            "prediction. peak_copy therefore reaches 1.0 in canonical_sqrt and in "
            "raw_via_squared_prediction, and falls short in legacy_raw and double_sqrt "
            "-- the two spaces that compare a square-root quantity against a raw one. "
            "peak_copy copies the observation and is a sanity check, not a ceiling. "
            "formula_rendered is the ceiling the formula supervision can express."
        ),
    }


# ----------------------------------------------------------------- canonical


def _fingerprint(prediction) -> dict:
    return {
        "formulas": tuple(f.formula for f in prediction.fragments),
        "ions": tuple(f.ion_state for f in prediction.fragments),
        "presence": np.asarray([f.presence for f in prediction.fragments]),
        "intensity": np.asarray([f.weighted_intensity for f in prediction.fragments]),
        "active": prediction.active_slots,
    }


def _predict(model, rows, tokenizer, device, *, canonical: bool, sequence_length: int,
             batch_size: int):
    out = {}
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        inputs = (
            canonical_model_inputs(chunk, tokenizer, sequence_length=sequence_length)
            if canonical
            else build_model_inputs(chunk, tokenizer, max_text_length=512)
        )
        assert_no_leakage(inputs)
        inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        for prediction in predict_spectrum(model, inputs, presence_threshold=0.5):
            out[prediction.spectrum_uid] = _fingerprint(prediction)
    return out


def _compare(reference: dict, other: dict) -> dict:
    shared = sorted(set(reference) & set(other))
    formula_same = sum(reference[u]["formulas"] == other[u]["formulas"] for u in shared)
    ion_same = sum(reference[u]["ions"] == other[u]["ions"] for u in shared)
    active_same = sum(reference[u]["active"] == other[u]["active"] for u in shared)
    intensity_gap = 0.0
    presence_gap = 0.0
    for uid in shared:
        a, b = reference[uid], other[uid]
        if a["formulas"] == b["formulas"] and a["intensity"].size == b["intensity"].size:
            if a["intensity"].size:
                intensity_gap = max(
                    intensity_gap, float(np.abs(a["intensity"] - b["intensity"]).max())
                )
                presence_gap = max(
                    presence_gap, float(np.abs(a["presence"] - b["presence"]).max())
                )
    return {
        "spectra": len(shared),
        "formula_identical": formula_same / len(shared),
        "ion_identical": ion_same / len(shared),
        "active_mask_identical": active_same / len(shared),
        "intensity_max_abs_diff": intensity_gap,
        "presence_max_abs_diff": presence_gap,
    }


def run_canonical(model, dataset, tokenizer, device, *, count: int, sequence_length: int) -> dict:
    rows = [dataset[i] for i in range(min(count, len(dataset)))]
    lengths = [len(tokenizer(r["smiles"])["input_ids"]) for r in rows]
    longest = rows[int(np.argmax(lengths))]
    shortest = rows[int(np.argmin(lengths))]

    report = {}
    for label, canonical in (("ragged", False), ("canonical", True)):
        alone = _predict(model, rows, tokenizer, device, canonical=canonical,
                         sequence_length=sequence_length, batch_size=1)
        variants = {
            "batch8": _predict(model, rows, tokenizer, device, canonical=canonical,
                               sequence_length=sequence_length, batch_size=8),
            "batch16": _predict(model, rows, tokenizer, device, canonical=canonical,
                                sequence_length=sequence_length, batch_size=16),
            "batch32": _predict(model, rows, tokenizer, device, canonical=canonical,
                                sequence_length=sequence_length, batch_size=32),
            "with_long_neighbour": _predict(model, [longest, *rows], tokenizer, device,
                                            canonical=canonical,
                                            sequence_length=sequence_length, batch_size=8),
            "with_short_neighbour": _predict(model, [shortest, *rows], tokenizer, device,
                                             canonical=canonical,
                                             sequence_length=sequence_length, batch_size=8),
            "reversed_order": _predict(model, rows[::-1], tokenizer, device, canonical=canonical,
                                       sequence_length=sequence_length, batch_size=16),
        }
        report[label] = {name: _compare(alone, values) for name, values in variants.items()}
    return report


# ------------------------------------------------------------ counterfactual


def collect_counterfactual(model, dataset, collator, device, binning, indices,
                           *, batch_size: int) -> tuple[dict, list]:
    names = list(CONDITIONS)
    cosine = {f"{n}|{s}": {k: [] for k in TOP_K} for n in names for s in INTENSITY_SPACES}
    rows_out = []
    indices = list(indices)
    for start in range(0, len(indices), batch_size):
        window = indices[start : start + batch_size]
        rows = [dataset[i] for i in window]
        batch = collator(rows)
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        model_inputs = {k: v for k, v in batch.items() if k in MODEL_INPUT_FIELDS}
        assert_no_leakage(model_inputs)
        predictions = predict_spectrum(model, model_inputs, presence_threshold=0.5)
        records = analyse_batch(model, batch, rows, predictions, presence_threshold=0.5)
        for row, record in zip(rows, records, strict=True):
            observed_mz = record.meta["observed_mz"]
            observed_intensity = record.meta["observed_intensity"]
            entry = {"spectrum_uid": record.spectrum_uid}
            for name in names:
                mz, intensity = record.conditions[name]
                for space in INTENSITY_SPACES:
                    for k in TOP_K:
                        value = cosine_in_space(
                            mz, intensity, observed_mz, observed_intensity,
                            space=space, k=k, config=binning,
                        )
                        cosine[f"{name}|{space}"][k].append(value)
                        if k == 100:
                            entry[f"cos100_{name}_{space}"] = value
            rows_out.append(entry)
    return cosine, rows_out


# ---------------------------------------------------------------------- bags


def collect_bags(model, dataset, collator, device, indices, *, batch_size: int) -> list:
    per_pair = []
    indices = list(indices)
    for start in range(0, len(indices), batch_size):
        window = indices[start : start + batch_size]
        rows = [dataset[i] for i in window]
        batch = collator(rows)
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        outputs = model(batch)
        cost = model.compute_matching_cost_no_grad(
            outputs.slots, outputs.presence, outputs.intensity, batch
        )
        assignment = hungarian_assign(cost, batch["target_peak_mask"])
        if len(assignment) == 0:
            continue
        matched = model.compute_matched_bag_nll(outputs.slots, batch, assignment)
        ion_log_prob = model.ion_head(outputs.slots, batch["admissible_ion_mask"])
        pb, pc, ps, pi = candidate_pairs_for_assignment(batch, assignment)
        scores = (
            model.formula_decoder.log_prob(
                outputs.slots[pb, ps],
                batch["precursor_element_ids"][pb],
                batch["precursor_element_counts"][pb],
                batch["candidate_formula_counts"][pb, pc],
                batch["precursor_element_mask"][pb],
            )
            + ion_log_prob[pb, ps, batch["candidate_ion_states"][pb, pc]]
        ).cpu().numpy()

        elements = batch["precursor_element_ids"].shape[1]
        shape = (outputs.slots.shape[0], outputs.slots.shape[1], elements)
        greedy = model.formula_decoder.greedy_decode(
            outputs.slots.reshape(-1, outputs.slots.shape[-1]),
            batch["precursor_element_ids"].unsqueeze(1).expand(shape).reshape(-1, elements),
            batch["precursor_element_counts"].unsqueeze(1).expand(shape).reshape(-1, elements),
            batch["precursor_element_mask"].unsqueeze(1).expand(shape).reshape(-1, elements),
        ).view(*shape).cpu().numpy()
        greedy_ion = outputs.ion_log_prob.argmax(dim=-1).cpu().numpy()
        candidate_counts = batch["candidate_formula_counts"].cpu().numpy()
        candidate_ions = batch["candidate_ion_states"].cpu().numpy()
        nll = matched.cpu().numpy()
        a_batch = assignment.batch_index.cpu().numpy()
        a_slot = assignment.slot_index.cpu().numpy()
        pi_np, pb_np, pc_np = pi.cpu().numpy(), pb.cpu().numpy(), pc.cpu().numpy()
        precursor_mz = [float(r["precursor_mz"]) for r in rows]
        targets = batch["target_peak_mask"].sum(dim=1).cpu().numpy()

        for pair in range(len(assignment)):
            members = np.flatnonzero(pi_np == pair)
            if members.size == 0:
                continue
            index, slot = int(a_batch[pair]), int(a_slot[pair])
            in_bag = any(
                bool((candidate_counts[index, int(pc_np[m])] == greedy[index, slot]).all())
                and int(candidate_ions[index, int(pc_np[m])]) == int(greedy_ion[index, slot])
                for m in members
            )
            per_pair.append(
                {
                    "cardinality": int(members.size),
                    "bag_nll": float(nll[pair]),
                    "bag_mass": float(np.exp(-nll[pair])),
                    "best_in_bag": float(scores[members].max()),
                    "in_bag": in_bag,
                    "precursor_mz": precursor_mz[index],
                    "targets": int(targets[index]),
                }
            )
    return per_pair


def summarise_bags(per_pair: list, label: str) -> dict:
    if not per_pair:
        return {"fold": label, "pairs": 0}

    nll = np.asarray([p["bag_nll"] for p in per_pair])
    mass = np.asarray([p["bag_mass"] for p in per_pair])
    card = np.asarray([p["cardinality"] for p in per_pair])
    hit = np.asarray([p["in_bag"] for p in per_pair], dtype=np.float64)

    def block(mask) -> dict:
        if not mask.any():
            return {}
        return {
            "pairs": int(mask.sum()),
            "bag_nll_mean": float(nll[mask].mean()),
            "geometric_mean_bag_mass": float(np.exp(-nll[mask].mean())),
            "arithmetic_mean_bag_mass": float(mass[mask].mean()),
            "median_bag_mass": float(np.median(mass[mask])),
            "argmax_in_bag": float(hit[mask].mean()),
        }

    groups = {
        "all": block(np.ones_like(card, dtype=bool)),
        "unique_bag": block(card == 1),
        "ambiguous_bag": block(card >= 2),
        "cardinality_1": block(card == 1),
        "cardinality_2_4": block((card >= 2) & (card <= 4)),
        "cardinality_5_8": block((card >= 5) & (card <= 8)),
        "cardinality_9_plus": block(card >= 9),
    }
    mz = np.asarray([p["precursor_mz"] for p in per_pair])
    for low, high in ((0, 200), (200, 300), (300, 400), (400, 600), (600, 1e9)):
        groups[f"precursor_mz_{low}_{high:g}"] = block((mz >= low) & (mz < high))
    targets = np.asarray([p["targets"] for p in per_pair])
    for low, high in ((0, 8), (8, 32), (32, 64), (64, 10**6)):
        groups[f"targets_{low}_{high:g}"] = block((targets >= low) & (targets < high))
    return {"fold": label, "groups": groups}


# ----------------------------------------------------------------- gpu shards


def build_worker_parts(config_path, checkpoint, rank):
    """A worker's own model, tokenizer and collator, pinned to its own GPU."""
    device = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    config = OmegaConf.load(config_path)
    config.model = OmegaConf.load(_ROOT / config.model_config)
    seed_everything(config.training.seed)
    configure_attention_backends()
    model = load_model(config, device)
    load_checkpoint(checkpoint, model=model, map_location=device)
    model.eval()
    tokenizer = load_tokenizer(config.model.encoder.model_name_or_path)
    collator = FragmentCollator(tokenizer, max_text_length=config.data.max_text_length)
    return config, model, collator, device


def open_dataset(root, fold, limit):
    if fold == "train":
        return FragmentSupervisionDataset(root, "train", exclude_zero_target=True, limit=limit)
    return FragmentSupervisionDataset(root, fold)


def model_shard(payload):
    """One GPU's share of a model section, run in its own spawned process."""
    (section, rank, config_path, checkpoint, root, fold, limit,
     start, stop, batch_size, binning) = payload
    _, model, collator, device = build_worker_parts(config_path, checkpoint, rank)
    dataset = open_dataset(root, fold, limit)
    with torch.no_grad():
        if section == "counterfactual":
            return collect_counterfactual(
                model, dataset, collator, device, binning, range(start, stop),
                batch_size=batch_size,
            )
        return collect_bags(
            model, dataset, collator, device, range(start, stop), batch_size=batch_size
        )


def shard_payloads(section, *, config_path, checkpoint, root, fold, limit, size,
                   batch_size, binning, gpus):
    return [
        (section, rank % max(gpus, 1), config_path, checkpoint, root, fold, limit,
         start, stop, batch_size, binning)
        for rank, (start, stop) in enumerate(chunks(size, gpus))
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/train/qwen_formula_slots_v0_pilot.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--canonical-count", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--train-sample", type=int, default=4000)
    parser.add_argument("--sections", default="oracles,canonical,counterfactual,bags")
    parser.add_argument("--cpu-workers", type=int, default=16,
                        help="processes for the model-free oracle sweep")
    parser.add_argument("--gpus", type=int, default=0,
                        help="GPU shards for the model sections; 0 uses every visible GPU")
    args = parser.parse_args(argv)

    sections = set(args.sections.split(","))
    config_path = str(Path(args.config).resolve())
    config = OmegaConf.load(config_path)
    config.model = OmegaConf.load(_ROOT / config.model_config)
    gpus = args.gpus or max(torch.cuda.device_count(), 1)

    binning = BinningConfig(
        upper_limit=config.evaluation.upper_limit,
        num_bins=config.evaluation.num_bins,
        min_pred_intensity=config.evaluation.min_pred_intensity,
        top_k=TOP_K,
    )
    root = config.data.root
    valid_size = len(FragmentSupervisionDataset(root, "valid"))

    audit = {
        "diagnostic_only": True,
        "fold": "valid",
        "test_fold_accessed": False,
        "checkpoint": str(args.checkpoint),
        "valid_spectra": valid_size,
        "intensity_spaces": list(INTENSITY_SPACES),
        "parallelism": {
            "cpu_workers": args.cpu_workers,
            "gpu_shards": gpus,
            "blas_threads_per_process": 1,
        },
        "ms_pred_reference": {
            "source": "https://github.com/coleygroup/ms-pred",
            "preprocessing": "common/misc_utils.py: spec[:,1] /= max; spec[:,1] = sqrt(spec[:,1])",
            "binning": "bin_spectra: floor(mz*(num_bins-1)/upper_limit)+1, pool_fn=max",
            "evaluation": "analysis/spec_pred_eval.py: no further transform, min_inten=1e-5, max_peaks=20",
            "norm_spectrum_is_dead_code": True,
        },
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if "oracles" in sections:
        started = time.perf_counter()
        audit["oracles"] = run_oracles(
            root, "valid", binning, size=valid_size, workers=args.cpu_workers
        )
        audit["oracles"]["seconds"] = round(time.perf_counter() - started, 1)
        print("[audit] oracles done", flush=True)

    if "canonical" in sections:
        started = time.perf_counter()
        seed_everything(config.training.seed)
        configure_attention_backends()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(config, device)
        load_checkpoint(args.checkpoint, model=model, map_location=device)
        model.eval()
        tokenizer = load_tokenizer(config.model.encoder.model_name_or_path)
        with torch.no_grad():
            audit["canonical_inference"] = run_canonical(
                model, FragmentSupervisionDataset(root, "valid"), tokenizer, device,
                count=args.canonical_count, sequence_length=args.sequence_length,
            )
        audit["canonical_inference"]["seconds"] = round(time.perf_counter() - started, 1)
        del model
        torch.cuda.empty_cache()
        print("[audit] canonical done", flush=True)

    if "counterfactual" in sections:
        started = time.perf_counter()
        payloads = shard_payloads(
            "counterfactual", config_path=config_path, checkpoint=args.checkpoint,
            root=root, fold="valid", limit=None, size=valid_size,
            batch_size=args.batch_size, binning=binning, gpus=gpus,
        )
        parts = run_shards(model_shard, payloads, gpus)
        rows_out = [entry for _, chunk_rows in parts for entry in chunk_rows]
        audit["counterfactual"] = {
            "cosine": summarise_cosines(merge_cosines(cosine for cosine, _ in parts)),
            "shards": len(payloads),
            "seconds": round(time.perf_counter() - started, 1),
        }
        frame = {key: [r.get(key) for r in rows_out] for key in rows_out[0]}
        pq.write_table(
            pa.table(frame), out / "canonical_valid_predictions.parquet", compression="zstd"
        )
        print("[audit] counterfactual done", flush=True)

    if "bags" in sections:
        started = time.perf_counter()
        train_size = len(
            FragmentSupervisionDataset(
                root, "train", exclude_zero_target=True, limit=args.train_sample
            )
        )
        bags = {}
        for label, fold, limit, size in (
            ("valid", "valid", None, valid_size),
            ("train_sample", "train", args.train_sample, train_size),
        ):
            payloads = shard_payloads(
                "bags", config_path=config_path, checkpoint=args.checkpoint, root=root,
                fold=fold, limit=limit, size=size, batch_size=args.batch_size,
                binning=binning, gpus=gpus,
            )
            per_pair = [pair for part in run_shards(model_shard, payloads, gpus) for pair in part]
            bags[label] = summarise_bags(per_pair, label)
        bags["train_sample_size"] = train_size
        bags["shards"] = gpus
        bags["seconds"] = round(time.perf_counter() - started, 1)
        audit["bags"] = bags
        print("[audit] bags done", flush=True)

    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"wrote {out / 'audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
