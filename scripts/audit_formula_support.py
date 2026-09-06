#!/usr/bin/env python3
"""Measure how much of a spectrum graph-free formula candidates can explain.

For every train/valid spectrum this enumerates the subformulas of the
precursor formula, matches them against each peak inside a precision-aware
mass window, and reports coverage.  Nothing is written per candidate: the
policy is not settled yet, so materialising a candidate table would bake in
choices that are still under review.  Only the aggregate report is produced.

The test fold is never read.

    build   audit --spectra <spectra_v2 dir> --out <report dir>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from metabo_sllm.chem.candidates import (  # noqa: E402
    BOUNDARY_EPS,
    DEFAULT_PPM,
    HEAVY_SUBFORMULA_CAP,
    PEAK_CANDIDATE_CAP,
    UnsupportedChargeError,
    channels_for_adduct,
    match_spectrum,
)
from metabo_sllm.chem.formula import (  # noqa: E402
    EnumerationCapExceeded,
    FormulaError,
    SubformulaTable,
    parse_formula,
)

REPORT_VERSION = "formula_support_audit_v0"
REPORT_NAME = "report.json"
# The test fold is intentionally absent and must stay that way.
AUDIT_FOLDS = ("train", "valid")
REQUIRED_SCHEMA_VERSION = "spectra_v2"
UNIQUE_CANDIDATE_THRESHOLDS = (64, 128, 256, 512)
TABLE_CACHE_SIZE = 512
MAX_FAILURE_EXAMPLES = 20


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


class TableCache:
    """LRU of :class:`SubformulaTable` keyed by precursor formula string.

    Rows arrive sorted by ``parent_spec``, so the collision-energy siblings of
    one molecule hit the same entry back to back.
    """

    def __init__(self, maxsize: int = TABLE_CACHE_SIZE) -> None:
        self._entries: OrderedDict[str, SubformulaTable] = OrderedDict()
        self._maxsize = maxsize
        self.hits = 0
        self.misses = 0
        self.max_heavy_size = 0
        self.max_total_size = 0

    def get(self, formula_text: str, *, heavy_cap: int) -> SubformulaTable:
        entry = self._entries.get(formula_text)
        if entry is not None:
            self.hits += 1
            self._entries.move_to_end(formula_text)
            return entry
        self.misses += 1
        table = SubformulaTable(parse_formula(formula_text), heavy_cap=heavy_cap)
        self.max_heavy_size = max(self.max_heavy_size, table.heavy_size)
        self.max_total_size = max(self.max_total_size, table.total_size)
        self._entries[formula_text] = table
        if len(self._entries) > self._maxsize:
            self._entries.popitem(last=False)
        return table


class FoldAccumulator:
    """Flat per-peak and per-spectrum arrays for one fold."""

    def __init__(self) -> None:
        self.peak_counts: list[np.ndarray] = []
        self.peak_supported: list[np.ndarray] = []
        self.peak_intensity: list[np.ndarray] = []
        self.peak_decimals: list[np.ndarray] = []
        self.spectrum_peaks: list[int] = []
        self.spectrum_cosine: list[float] = []
        self.spectrum_unique: list[int] = []
        self.spectrum_instrument: list[str] = []
        self.spectrum_adduct: list[str] = []
        self.zero_energy_spectra = 0

    def add(
        self,
        counts: np.ndarray,
        intensities: np.ndarray,
        decimals: np.ndarray,
        unique: int,
        instrument: str,
        adduct: str,
    ) -> None:
        supported = counts > 0
        self.peak_counts.append(counts.astype(np.int64, copy=False))
        self.peak_supported.append(supported)
        self.peak_intensity.append(intensities.astype(np.float64, copy=False))
        self.peak_decimals.append(decimals.astype(np.int16, copy=False))

        squared = np.square(intensities.astype(np.float64, copy=False))
        total = float(squared.sum())
        if total > 0.0:
            cosine = float(np.sqrt(squared[supported].sum() / total))
        else:
            cosine = 0.0
            self.zero_energy_spectra += 1

        self.spectrum_peaks.append(int(counts.shape[0]))
        self.spectrum_cosine.append(cosine)
        self.spectrum_unique.append(int(unique))
        self.spectrum_instrument.append(instrument)
        self.spectrum_adduct.append(adduct)

    def finish(self) -> dict[str, np.ndarray]:
        empty_f = np.empty(0, dtype=np.float64)
        return {
            "counts": np.concatenate(self.peak_counts) if self.peak_counts else np.empty(0, np.int64),
            "supported": np.concatenate(self.peak_supported)
            if self.peak_supported
            else np.empty(0, bool),
            "intensity": np.concatenate(self.peak_intensity) if self.peak_intensity else empty_f,
            "decimals": np.concatenate(self.peak_decimals)
            if self.peak_decimals
            else np.empty(0, np.int16),
            "spectrum_peaks": np.asarray(self.spectrum_peaks, dtype=np.int64),
            "cosine": np.asarray(self.spectrum_cosine, dtype=np.float64),
            "unique": np.asarray(self.spectrum_unique, dtype=np.int64),
        }


# --------------------------------------------------------------------------- metrics


def _stats(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"mean": None, "median": None, "p90": None, "p99": None, "max": None}
    return {
        "mean": float(values.mean()),
        "median": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def _peak_block(counts: np.ndarray, supported: np.ndarray, intensity: np.ndarray) -> dict:
    peaks = int(counts.size)
    if peaks == 0:
        return {"peaks": 0}
    intensity_total = float(intensity.sum())
    return {
        "peaks": peaks,
        "supported_peaks": int(supported.sum()),
        "supported_peak_fraction": float(supported.mean()),
        "intensity_weighted_coverage": (
            float(intensity[supported].sum() / intensity_total) if intensity_total > 0 else None
        ),
        "candidate_count_zero_fraction": float((counts == 0).mean()),
        "candidate_count_one_fraction": float((counts == 1).mean()),
        "candidate_count_two_plus_fraction": float((counts >= 2).mean()),
        "candidate_count": _stats(counts),
    }


def _spectrum_block(cosine: np.ndarray, unique: np.ndarray, fully: np.ndarray) -> dict:
    if cosine.size == 0:
        return {"spectra": 0}
    block = {
        "spectra": int(cosine.size),
        "oracle_cosine_mean": float(cosine.mean()),
        "oracle_cosine_median": float(np.percentile(cosine, 50)),
        "oracle_cosine_p10": float(np.percentile(cosine, 10)),
        "fully_supported_spectrum_fraction": float(fully.mean()),
        "unique_candidates_p50": float(np.percentile(unique, 50)),
        "unique_candidates_p90": float(np.percentile(unique, 90)),
        "unique_candidates_p99": float(np.percentile(unique, 99)),
        "unique_candidates_max": int(unique.max()),
    }
    for threshold in UNIQUE_CANDIDATE_THRESHOLDS:
        block[f"unique_candidates_le_{threshold}_fraction"] = float((unique <= threshold).mean())
    return block


def _fold_metrics(data: dict[str, np.ndarray], fully: np.ndarray, groups: dict) -> dict:
    counts, supported = data["counts"], data["supported"]
    intensity, decimals = data["intensity"], data["decimals"]
    cosine, unique = data["cosine"], data["unique"]

    block = _peak_block(counts, supported, intensity)
    block.update(_spectrum_block(cosine, unique, fully))
    block["low_precision_peaks"] = int((decimals <= 1).sum())

    for key, out_name in (("instrument", "by_instrument"), ("adduct", "by_adduct")):
        grouped = {}
        for name, peak_mask in groups["peak"][key].items():
            spectrum_mask = groups["spectrum"][key][name]
            entry = _peak_block(counts[peak_mask], supported[peak_mask], intensity[peak_mask])
            entry.update(
                _spectrum_block(cosine[spectrum_mask], unique[spectrum_mask], fully[spectrum_mask])
            )
            grouped[name] = entry
        block[out_name] = grouped

    by_decimals = {}
    for value in sorted({int(v) for v in np.unique(decimals)}):
        mask = decimals == value
        entry = _peak_block(counts[mask], supported[mask], intensity[mask])
        entry["low_precision"] = value <= 1
        by_decimals[str(value)] = entry
    block["by_mz_decimals"] = by_decimals
    return block


# --------------------------------------------------------------------------- io


def _list_column(table: pa.Table, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Offsets and flat values of a list column, without per-row Python objects."""
    column = table.column(name).combine_chunks()
    array = column.chunk(0) if isinstance(column, pa.ChunkedArray) else column
    offsets = array.offsets.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    values = array.values.to_numpy(zero_copy_only=False)
    return offsets, values


def audit(args: argparse.Namespace) -> int:
    spectra_dir = Path(args.spectra).resolve()
    out_dir = Path(args.out).resolve()
    manifest_path = spectra_dir / "manifest.json"
    if not manifest_path.is_file():
        fail(f"spectra manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema_version"] != REQUIRED_SCHEMA_VERSION:
        fail(
            f"{manifest_path}: schema_version is {manifest['schema_version']!r}, "
            f"this audit needs {REQUIRED_SCHEMA_VERSION!r}"
        )
    if out_dir.exists() and not args.overwrite:
        fail(f"output already exists: {out_dir} (pass --overwrite to replace it)")

    limits = {}
    if args.limit_spectra is not None:
        half = args.limit_spectra // 2
        limits = {"train": half, "valid": args.limit_spectra - half}

    cache = TableCache()
    accumulators = {fold: FoldAccumulator() for fold in AUDIT_FOLDS}
    failures: dict[str, list[dict]] = defaultdict(list)
    failure_counts: dict[str, int] = defaultdict(int)

    started = time.perf_counter()
    for fold in AUDIT_FOLDS:
        fold_dir = spectra_dir / fold
        if not fold_dir.is_dir():
            fail(f"fold directory missing: {fold_dir}")
        budget = limits.get(fold)
        accumulator = accumulators[fold]
        seen = 0
        for path in sorted(fold_dir.glob("part-*.parquet")):
            if budget is not None and seen >= budget:
                break
            table = pq.read_table(
                path,
                columns=[
                    "spectrum_uid",
                    "formula",
                    "adduct",
                    "instrument",
                    "mzs",
                    "intensities",
                    "mz_decimal_places",
                ],
            )
            if table.num_rows == 0:
                continue
            uids = table.column("spectrum_uid").to_pylist()
            formulas = table.column("formula").to_pylist()
            adducts = table.column("adduct").to_pylist()
            instruments = table.column("instrument").to_pylist()
            mz_offsets, mz_values = _list_column(table, "mzs")
            in_offsets, in_values = _list_column(table, "intensities")
            dp_offsets, dp_values = _list_column(table, "mz_decimal_places")

            for row in range(table.num_rows):
                if budget is not None and seen >= budget:
                    break
                mzs = mz_values[mz_offsets[row] : mz_offsets[row + 1]]
                intensities = in_values[in_offsets[row] : in_offsets[row + 1]]
                decimals = dp_values[dp_offsets[row] : dp_offsets[row + 1]]

                try:
                    subformulas = cache.get(formulas[row], heavy_cap=args.heavy_cap)
                except FormulaError as exc:
                    failure_counts["formula_parse"] += 1
                    if len(failures["formula_parse"]) < MAX_FAILURE_EXAMPLES:
                        failures["formula_parse"].append({"spectrum_uid": uids[row], "error": str(exc)})
                    continue
                except EnumerationCapExceeded as exc:
                    failure_counts["enumeration_cap"] += 1
                    if len(failures["enumeration_cap"]) < MAX_FAILURE_EXAMPLES:
                        failures["enumeration_cap"].append(
                            {"spectrum_uid": uids[row], "error": str(exc)}
                        )
                    continue

                try:
                    channels = channels_for_adduct(adducts[row])
                except UnsupportedChargeError as exc:
                    failure_counts["unsupported_charge"] += 1
                    if len(failures["unsupported_charge"]) < MAX_FAILURE_EXAMPLES:
                        failures["unsupported_charge"].append(
                            {"spectrum_uid": uids[row], "error": str(exc)}
                        )
                    continue

                try:
                    match = match_spectrum(
                        subformulas,
                        mzs,
                        decimals,
                        channels,
                        ppm=args.ppm,
                        peak_cap=args.peak_cap,
                    )
                except EnumerationCapExceeded as exc:
                    failure_counts["enumeration_cap"] += 1
                    if len(failures["enumeration_cap"]) < MAX_FAILURE_EXAMPLES:
                        failures["enumeration_cap"].append(
                            {"spectrum_uid": uids[row], "error": str(exc)}
                        )
                    continue

                accumulator.add(
                    match.peak_candidate_counts,
                    intensities,
                    decimals,
                    match.unique_candidates,
                    instruments[row],
                    adducts[row],
                )
                seen += 1
        print(f"[audit] {fold:<5} spectra={seen:,}", flush=True)
    elapsed = time.perf_counter() - started

    folds_block = {}
    total_spectra = total_peaks = 0
    for fold in AUDIT_FOLDS:
        accumulator = accumulators[fold]
        data = accumulator.finish()
        peaks_per = data["spectrum_peaks"]
        # A spectrum is fully supported when it has no unsupported peak; the
        # running count of unsupported peaks differenced at segment boundaries
        # gives that without a Python loop.
        ends = np.cumsum(peaks_per)
        starts = ends - peaks_per
        unsupported_cumulative = np.concatenate(([0], np.cumsum(~data["supported"])))
        fully = (unsupported_cumulative[ends] - unsupported_cumulative[starts]) == 0

        instruments = np.asarray(accumulator.spectrum_instrument)
        adducts = np.asarray(accumulator.spectrum_adduct)
        groups = {"peak": {"instrument": {}, "adduct": {}}, "spectrum": {"instrument": {}, "adduct": {}}}
        for key, labels in (("instrument", instruments), ("adduct", adducts)):
            for name in sorted(set(labels.tolist())):
                smask = labels == name
                groups["spectrum"][key][name] = smask
                groups["peak"][key][name] = np.repeat(smask, peaks_per)

        folds_block[fold] = _fold_metrics(data, fully, groups)
        folds_block[fold]["zero_energy_spectra"] = accumulator.zero_energy_spectra
        total_spectra += int(peaks_per.size)
        total_peaks += int(peaks_per.sum())

    report = {
        "report_version": REPORT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "spectra_dir": str(spectra_dir),
            "spectra_schema_version": manifest["schema_version"],
            "spectra_manifest_created_utc": manifest["created_utc"],
            "source_files": {k: v["sha256"] for k, v in manifest["inputs"].items()},
        },
        "policy": {
            "graph_free": True,
            "magma_used": False,
            "ppm": args.ppm,
            "tolerance_rule": (
                "decimals >= 2: max(mz * ppm, 0.5 * 10**-decimals); "
                "decimals <= 1: ppm only, peak flagged low_precision"
            ),
            "boundary_epsilon": BOUNDARY_EPS,
            "heavy_subformula_cap": args.heavy_cap,
            "peak_candidate_cap": args.peak_cap,
            "ion_channels": {
                "positive_base": "protonated",
                "negative_base": "deprotonated",
                "sodiated_if_adduct_has_Na": True,
                "potassiated_if_adduct_has_K": True,
                "chloride_retained_if_negative_Cl_adduct": True,
                "ammonium_retained": False,
                "formate_retained": False,
                "radical_channels": False,
            },
            "subformula_rule": "0 <= F[e] <= precursor[e] for every element including H; empty excluded",
            "limit_spectra": args.limit_spectra,
        },
        "runtime": {
            "elapsed_seconds": round(elapsed, 3),
            "spectra_per_second": round(total_spectra / elapsed, 1) if elapsed > 0 else None,
            "peaks_per_second": round(total_peaks / elapsed, 1) if elapsed > 0 else None,
            "subformula_cache": {
                "hits": cache.hits,
                "misses": cache.misses,
                "max_heavy_subformulas": cache.max_heavy_size,
                "max_total_subformulas": cache.max_total_size,
            },
        },
        "folds": folds_block,
        "totals": {"spectra": total_spectra, "peaks": total_peaks},
        "failures": {
            name: {"count": failure_counts.get(name, 0), "examples": failures.get(name, [])}
            for name in ("formula_parse", "unsupported_charge", "enumeration_cap")
        },
        "silent_truncation": 0,
        "test_fold_read": False,
        "test_metrics_computed": False,
    }

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir.parent / f"{out_dir.name}.tmp.{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    (tmp_dir / REPORT_NAME).write_text(json.dumps(report, indent=2, sort_keys=False) + "\n")
    if out_dir.exists():
        trash = out_dir.parent / f"{out_dir.name}.trash.{os.getpid()}"
        out_dir.rename(trash)
        os.replace(tmp_dir, out_dir)
        shutil.rmtree(trash)
    else:
        os.replace(tmp_dir, out_dir)

    for fold in AUDIT_FOLDS:
        block = folds_block[fold]
        print(
            f"[audit] {fold:<5} spectra={block.get('spectra', 0):>7,} peaks={block.get('peaks', 0):>9,} "
            f"supported={block.get('supported_peak_fraction', 0):.4f} "
            f"int_cov={block.get('intensity_weighted_coverage') or 0:.4f} "
            f"cos_mean={block.get('oracle_cosine_mean', 0):.4f}"
        )
    print(
        f"[audit] {elapsed:.1f}s, {report['runtime']['peaks_per_second']:,} peaks/s -> {out_dir}",
        flush=True,
    )

    total_failures = sum(failure_counts.values())
    if total_failures:
        print(f"[audit] {total_failures} failures recorded:", file=sys.stderr)
        for name, count in failure_counts.items():
            print(f"  {name}: {count}", file=sys.stderr)
            for example in failures[name][:5]:
                print(f"    {example['spectrum_uid']}: {example['error']}", file=sys.stderr)
        return 1
    print("[audit] OK - no formula, charge or enumeration failures")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spectra", required=True, help="spectra_v2 dataset directory")
    parser.add_argument("--out", required=True, help="report output directory")
    parser.add_argument("--ppm", type=float, default=DEFAULT_PPM)
    parser.add_argument("--heavy-cap", type=int, default=HEAVY_SUBFORMULA_CAP)
    parser.add_argument("--peak-cap", type=int, default=PEAK_CANDIDATE_CAP)
    parser.add_argument(
        "--limit-spectra",
        type=int,
        default=None,
        help="total spectra to audit, split evenly between train and valid (smoke runs)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return audit(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
