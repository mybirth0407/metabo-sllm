#!/usr/bin/env python3
"""What kind of identity errors does a checkpoint make? Diagnostic only.

Three questions about every active predicted slot on the valid fold, all
answerable from the prediction parquet and the supervision data alone:

  errors      Is the (formula, ion) inside some observed peak's candidate bag?
              If not, how far is it from the nearest true candidate -- a
              hydrogen or two, one heavy atom, or nowhere near -- and how far
              in Da is its rendered m/z from the nearest observed peak?
              Near-misses and far-misses call for different fixes.
  valence     Do misses violate valence/RDBE bounds that hits do not?  If so a
              decoder mask removes them for free.  Also reports how many true
              candidates violate them, which is a supervision-quality number.
  vocabulary  Would a frequency vocabulary built from the train bags have
              contained the miss formulas?  If misses are mostly never-seen
              formulas a vocabulary prior removes them; if they are common
              formulas on the wrong molecule it does not.

Sharded over processes per fold shard.  One BLAS thread per process, set
before numpy is imported.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _threads in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_threads, "1")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from metabo_sllm.chem.formula import formula_to_string, parse_formula  # noqa: E402

FORBIDDEN_FOLD = "test"
HALOGENS = ("F", "Cl", "Br", "I")
MONOVALENT = HALOGENS + ("Na", "K")
MASS_BUCKETS = ((0, 200), (200, 300), (300, 400), (400, 600), (600, float("inf")))

_PREDICTIONS: dict = {}


def _load_predictions(path: str) -> None:
    global _PREDICTIONS
    _PREDICTIONS = {r["spectrum_uid"]: r for r in pq.read_table(path).to_pylist()}


def _bucket(precursor_mz: float) -> str:
    for low, high in MASS_BUCKETS:
        if low <= precursor_mz < high:
            return f"{low:g}-{high:g}" if high != float("inf") else f">={low:g}"
    return "?"


# ------------------------------------------------------------------- errors


def _classify(pred: dict, truths: list[dict]) -> tuple[str, int, int, int]:
    """Nearest true candidate by L1 over element counts, and what differs."""
    best = None
    for tc in truths:
        keys = set(pred) | set(tc)
        diff = {k: pred.get(k, 0) - tc.get(k, 0) for k in keys}
        l1 = sum(abs(v) for v in diff.values())
        if best is None or l1 < best[0]:
            best = (l1, diff)
    l1, diff = best
    h = abs(diff.get("H", 0))
    heavy = sum(abs(v) for k, v in diff.items() if k != "H")
    if l1 == 0:
        return "hit", l1, h, heavy
    if heavy == 0:
        return ("H_only_le2" if h <= 2 else "H_only_gt2"), l1, h, heavy
    if heavy == 1:
        return "one_heavy_atom", l1, h, heavy
    if heavy == 2:
        return "two_heavy_atoms", l1, h, heavy
    return "far", l1, h, heavy


def errors_shard(payload):
    supervision, fold, shard = payload
    rows = pq.read_table(
        Path(supervision) / fold / f"part-{shard:05d}.parquet",
        columns=["spectrum_uid", "precursor_mz", "mzs", "intensities",
                 "candidate_neutral_formula", "candidate_ion_state",
                 "edge_peak_index", "edge_candidate_index"],
    ).to_pylist()
    category = Counter()
    by_mass = defaultdict(Counter)
    gaps = []
    covered_intensity = total_intensity = 0.0
    slots = spectra = 0
    for row in rows:
        pred = _PREDICTIONS.get(row["spectrum_uid"])
        if pred is None:
            continue
        spectra += 1
        observed_mz = np.asarray(row["mzs"], dtype=np.float64)
        observed_intensity = np.asarray(row["intensities"], dtype=np.float64)
        candidates = list(zip(row["candidate_neutral_formula"], row["candidate_ion_state"]))
        true_set = set(candidates)
        true_counts = [parse_formula(f) for f, _ in candidates]
        peaks_of = defaultdict(set)
        for peak, cand in zip(row["edge_peak_index"], row["edge_candidate_index"]):
            peaks_of[candidates[cand]].add(peak)
        covered = set()
        bucket = _bucket(float(row["precursor_mz"]))
        for f, ion, mz in zip(pred["predicted_formulas"], pred["predicted_ion_states"],
                              pred["predicted_mz"]):
            slots += 1
            if (f, ion) in true_set:
                category["hit"] += 1
                by_mass[bucket]["hit"] += 1
                covered |= peaks_of[(f, ion)]
                continue
            label = _classify(parse_formula(f), true_counts)[0] if true_counts else "no_candidates"
            category[label] += 1
            by_mass[bucket][label] += 1
            if observed_mz.size:
                gaps.append(float(np.min(np.abs(observed_mz - mz))))
        total_intensity += float(observed_intensity.sum())
        if covered:
            covered_intensity += float(observed_intensity[sorted(covered)].sum())
    return {
        "category": category,
        "by_mass": {k: dict(v) for k, v in by_mass.items()},
        "gaps": gaps,
        "covered_intensity": covered_intensity,
        "total_intensity": total_intensity,
        "slots": slots,
        "spectra": spectra,
    }


# ------------------------------------------------------------------ valence


def rdbe(counts: dict) -> float:
    tetra = counts.get("C", 0) + counts.get("Si", 0)
    mono = counts.get("H", 0) + sum(counts.get(x, 0) for x in MONOVALENT)
    tri = counts.get("N", 0) + counts.get("P", 0)
    return tetra - mono / 2 + tri / 2 + 1


def hydrogen_excess(counts: dict) -> int:
    """Monovalent atoms beyond the saturated maximum ``2C + 2 + N``."""
    tetra = counts.get("C", 0) + counts.get("Si", 0)
    tri = counts.get("N", 0) + counts.get("P", 0)
    mono = counts.get("H", 0) + sum(counts.get(x, 0) for x in MONOVALENT)
    return mono - (2 * tetra + 2 + tri)


def valence_flags(counts: dict, precursor_rdbe: float) -> dict:
    r = rdbe(counts)
    over = hydrogen_excess(counts) > 0
    return {
        "rdbe_negative": r < 0,
        "h_over_saturation": over,
        "rdbe_above_precursor_plus1": r > precursor_rdbe + 1,
        "any_hard": r < 0 or over,
        "any_hard_or_soft": r < 0 or over or r > precursor_rdbe + 1,
    }


def valence_shard(payload):
    supervision, fold, shard = payload
    rows = pq.read_table(
        Path(supervision) / fold / f"part-{shard:05d}.parquet",
        columns=["spectrum_uid", "formula", "candidate_neutral_formula", "candidate_ion_state"],
    ).to_pylist()
    tally = {g: Counter() for g in ("hit", "miss", "candidate")}
    n = Counter()
    for row in rows:
        pred = _PREDICTIONS.get(row["spectrum_uid"])
        if pred is None:
            continue
        precursor_rdbe = rdbe(parse_formula(row["formula"]))
        true_set = set(zip(row["candidate_neutral_formula"], row["candidate_ion_state"]))
        for f in row["candidate_neutral_formula"]:
            n["candidate"] += 1
            for k, v in valence_flags(parse_formula(f), precursor_rdbe).items():
                tally["candidate"][k] += int(v)
        for f, ion in zip(pred["predicted_formulas"], pred["predicted_ion_states"]):
            group = "hit" if (f, ion) in true_set else "miss"
            n[group] += 1
            for k, v in valence_flags(parse_formula(f), precursor_rdbe).items():
                tally[group][k] += int(v)
    return tally, n


# --------------------------------------------------------------- vocabulary


def neutral_loss(precursor: str, fragment: str) -> str | None:
    p, f = parse_formula(precursor), parse_formula(fragment)
    diff = {k: p.get(k, 0) - f.get(k, 0) for k in set(p) | set(f)}
    if any(v < 0 for v in diff.values()):
        return None
    diff = {k: v for k, v in diff.items() if v > 0}
    return formula_to_string(diff) if diff else "[M]"


def count_train_shard(payload):
    supervision, shard = payload
    rows = pq.read_table(
        Path(supervision) / "train" / f"part-{shard:05d}.parquet",
        columns=["formula", "candidate_neutral_formula", "edge_candidate_index"],
    ).to_pylist()
    fragments, losses = Counter(), Counter()
    for row in rows:
        # each candidate counted once per spectrum in which it explains a peak
        for cand in set(row["edge_candidate_index"]):
            f = row["candidate_neutral_formula"][cand]
            fragments[f] += 1
            loss = neutral_loss(row["formula"], f)
            if loss is not None:
                losses[loss] += 1
    return fragments, losses


def vocabulary_shard(payload):
    supervision, fold, shard, fragments, losses, thresholds = payload
    rows = pq.read_table(
        Path(supervision) / fold / f"part-{shard:05d}.parquet",
        columns=["spectrum_uid", "formula", "candidate_neutral_formula", "candidate_ion_state"],
    ).to_pylist()
    seen = {g: {k: Counter() for k in thresholds} for g in ("hit", "miss")}
    n = Counter()
    for row in rows:
        pred = _PREDICTIONS.get(row["spectrum_uid"])
        if pred is None:
            continue
        true_set = set(zip(row["candidate_neutral_formula"], row["candidate_ion_state"]))
        for f, ion in zip(pred["predicted_formulas"], pred["predicted_ion_states"]):
            group = "hit" if (f, ion) in true_set else "miss"
            n[group] += 1
            loss = neutral_loss(row["formula"], f)
            for k in thresholds:
                as_fragment = fragments.get(f, 0) >= k
                as_loss = loss is not None and losses.get(loss, 0) >= k
                seen[group][k]["fragment"] += int(as_fragment)
                seen[group][k]["loss"] += int(as_loss)
                seen[group][k]["either"] += int(as_fragment or as_loss)
    return seen, n


# --------------------------------------------------------------------- main


def run_pool(function, payloads, workers, predictions):
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(workers, len(payloads)), mp_context=context,
        initializer=_load_predictions, initargs=(predictions,),
    ) as pool:
        return list(pool.map(function, payloads))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="training run directory")
    parser.add_argument("--supervision", required=True, help="fragment_supervision_v1 directory")
    parser.add_argument("--fold", default="valid")
    parser.add_argument("--predictions", default="predictions/valid_best.parquet",
                        help="prediction parquet, relative to --run")
    parser.add_argument("--out", default=None, help="default: <run>/diagnostics/identity_errors.json")
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--sections", default="errors,valence,vocabulary")
    args = parser.parse_args(argv)

    if args.fold == FORBIDDEN_FOLD:
        raise SystemExit("refusing to read the test fold")
    run = Path(args.run)
    predictions = str(run / args.predictions)
    out = Path(args.out) if args.out else run / "diagnostics" / "identity_errors.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    sections = set(args.sections.split(","))
    shards = range(args.num_shards)
    report: dict = {"diagnostic_only": True, "fold": args.fold, "predictions": predictions,
                    "test_fold_accessed": False}

    if "errors" in sections:
        parts = run_pool(errors_shard, [(args.supervision, args.fold, s) for s in shards],
                         args.workers, predictions)
        category, by_mass, gaps = Counter(), defaultdict(Counter), []
        covered = total = 0.0
        slots = spectra = 0
        for p in parts:
            category.update(p["category"])
            for k, v in p["by_mass"].items():
                by_mass[k].update(v)
            gaps.extend(p["gaps"])
            covered += p["covered_intensity"]
            total += p["total_intensity"]
            slots += p["slots"]
            spectra += p["spectra"]
        gaps_arr = np.asarray(gaps)
        order = ["hit", "H_only_le2", "H_only_gt2", "one_heavy_atom", "two_heavy_atoms", "far",
                 "no_candidates"]
        report["errors"] = {
            "spectra": spectra,
            "active_slots": slots,
            "observed_intensity_covered_by_hits": covered / total if total else 0.0,
            "category_fraction": {k: category[k] / slots for k in order if slots},
            "category_by_precursor_mz": {
                b: {k: v / sum(c.values()) for k, v in c.items()} for b, c in by_mass.items()
            },
            "miss_mz_gap_da_percentiles": {
                str(q): float(np.percentile(gaps_arr, q)) for q in (10, 25, 50, 75, 90)
            } if gaps_arr.size else {},
            "miss_within_da": {
                str(t): float((gaps_arr <= t).mean()) for t in (0.5, 2, 15)
            } if gaps_arr.size else {},
        }
        print(f"[errors] spectra={spectra} active_slots={slots} "
              f"intensity_covered={report['errors']['observed_intensity_covered_by_hits']:.4f}")
        for k in order:
            if category[k]:
                print(f"  {k:18s} {category[k]:8d}  {category[k] / slots:6.1%}")
        if gaps_arr.size:
            print("  miss |Δm/z| p50=%.3f p75=%.3f p90=%.3f Da; within 0.5 Da %.1f%%" % (
                np.percentile(gaps_arr, 50), np.percentile(gaps_arr, 75),
                np.percentile(gaps_arr, 90), 100 * (gaps_arr <= 0.5).mean()))

    if "valence" in sections:
        parts = run_pool(valence_shard, [(args.supervision, args.fold, s) for s in shards],
                         args.workers, predictions)
        tally = {g: Counter() for g in ("hit", "miss", "candidate")}
        n = Counter()
        for t, c in parts:
            for g in tally:
                tally[g].update(t[g])
            n.update(c)
        report["valence"] = {
            g: {k: v / n[g] for k, v in tally[g].items()} if n[g] else {} for g in tally
        }
        report["valence"]["counts"] = dict(n)
        print("[valence] fraction violating, hit / miss / true candidates:")
        for k in ("rdbe_negative", "h_over_saturation", "rdbe_above_precursor_plus1",
                  "any_hard", "any_hard_or_soft"):
            print(f"  {k:30s} " + "  ".join(
                f"{tally[g][k] / n[g]:6.1%}" if n[g] else "   n/a" for g in ("hit", "miss", "candidate")))

    if "vocabulary" in sections:
        counted = run_pool(count_train_shard, [(args.supervision, s) for s in shards],
                           args.workers, predictions)
        fragments, losses = Counter(), Counter()
        for f, l in counted:
            fragments.update(f)
            losses.update(l)
        thresholds = (1, 5, 20, 100)
        parts = run_pool(
            vocabulary_shard,
            [(args.supervision, args.fold, s, fragments, losses, thresholds) for s in shards],
            args.workers, predictions,
        )
        seen = {g: {k: Counter() for k in thresholds} for g in ("hit", "miss")}
        n = Counter()
        for s, c in parts:
            for g in seen:
                for k in thresholds:
                    seen[g][k].update(s[g][k])
            n.update(c)
        report["vocabulary"] = {
            "train_distinct_fragments": len(fragments),
            "train_distinct_losses": len(losses),
            "train_fragments_seen_at_least": {
                str(k): sum(1 for v in fragments.values() if v >= k) for k in thresholds},
            "train_losses_seen_at_least": {
                str(k): sum(1 for v in losses.values() if v >= k) for k in thresholds},
            "prediction_seen_in_train": {
                g: {str(k): {kind: v / n[g] for kind, v in seen[g][k].items()} for k in thresholds}
                for g in seen if n[g]
            },
        }
        print(f"[vocabulary] train distinct fragments={len(fragments):,} losses={len(losses):,}")
        for k in thresholds:
            h, m = seen["hit"][k], seen["miss"][k]
            print(f"  seen>={k:3d}: hit fragment {h['fragment'] / n['hit']:6.1%} either "
                  f"{h['either'] / n['hit']:6.1%} | miss fragment {m['fragment'] / n['miss']:6.1%} "
                  f"either {m['either'] / n['miss']:6.1%}")

    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
