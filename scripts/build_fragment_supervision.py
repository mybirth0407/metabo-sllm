#!/usr/bin/env python3
"""Build fragment-formula supervision for the first model training run.

One output row is one spectrum.  It keeps the model input fields, the *whole*
raw spectrum as a full-spectrum target (no filtering, no normalisation, no
binning), the spectrum's de-duplicated formula candidates, and the
peak-to-candidate edges between them.

A peak is marked for fragment-identity supervision only when it has at least
one candidate and its m/z was printed with two or more decimals.  Peaks failing
either test stay in the spectrum -- they remain valid full-spectrum targets.

The model has 64 fragment slots.  This builder does **not** cut the data down
to 64: it stores every peak and every edge, and records the deterministic
ordering (intensity desc, m/z asc, peak index asc) that training will use to
pick which supervised peaks occupy those slots.

Candidate generation reuses the policy from formula_support_audit_v0 unchanged:
graph-free, no MAGMa, same enumeration, ion states, and mass tolerance.  Hitting
a safety cap fails the run rather than silently truncating.

The test fold is never read.

    build   --spectra <spectra_v2 dir> --out <fragment_supervision_v1 dir>
    verify  --spectra <spectra_v2 dir> --out <fragment_supervision_v1 dir>
"""

from __future__ import annotations

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from metabo_sllm.chem.candidates import SpectrumEdges  # noqa: E402
from metabo_sllm.chem.formula import is_valence_plausible  # noqa: E402
from metabo_sllm.chem.candidates import (  # noqa: E402
    BOUNDARY_EPS,
    DEFAULT_PPM,
    HEAVY_SUBFORMULA_CAP,
    PEAK_CANDIDATE_CAP,
    UnsupportedChargeError,
    channels_for_adduct,
    key_components,
    match_edges,
    theoretical_mz,
    tolerance_array,
)
from metabo_sllm.chem.formula import (  # noqa: E402
    EnumerationCapExceeded,
    FormulaError,
    SubformulaTable,
    parse_formula,
)
from metabo_sllm.data.sharding import (  # noqa: E402
    SHARD_FN,
    UID_FORMAT,
    shard_filename,
    shard_from_filename,
    shard_of,
)
from metabo_sllm.data.supervision import (  # noqa: E402
    FIXED_SLOT_COUNT,
    MIN_SUPERVISION_DECIMALS,
    supervision_mask,
    supervision_rank,
    top_slot_mask,
)

SCHEMA_VERSION = "fragment_supervision_v2"
SOURCE_SCHEMA_VERSION = "spectra_v2"
MANIFEST_NAME = "manifest.json"
DIAGNOSTICS_NAME = "diagnostics.json"
DEFAULT_NUM_SHARDS = 64
# The test fold is intentionally absent and must stay that way.
BUILD_FOLDS = ("train", "valid")
TABLE_CACHE_SIZE = 64

SOURCE_COLUMNS = [
    "spectrum_uid",
    "parent_spec",
    "fold",
    "collision_index",
    "collision_energy",
    "smiles",
    "formula",
    "inchikey",
    "adduct",
    "instrument",
    "precursor_mz",
    "mzs",
    "intensities",
    "mz_decimal_places",
]

SCHEMA = pa.schema(
    [
        # model input
        pa.field("spectrum_uid", pa.string(), nullable=False),
        pa.field("parent_spec", pa.string(), nullable=False),
        pa.field("fold", pa.string(), nullable=False),
        pa.field("collision_index", pa.int32(), nullable=False),
        pa.field("collision_energy", pa.float64(), nullable=False),
        pa.field("smiles", pa.string(), nullable=False),
        pa.field("formula", pa.string(), nullable=False),
        pa.field("inchikey", pa.string(), nullable=False),
        pa.field("adduct", pa.string(), nullable=False),
        pa.field("instrument", pa.string(), nullable=False),
        pa.field("precursor_mz", pa.float64(), nullable=False),
        # full-spectrum target, unmodified
        pa.field("mzs", pa.list_(pa.float64()), nullable=False),
        pa.field("intensities", pa.list_(pa.float32()), nullable=False),
        pa.field("mz_decimal_places", pa.list_(pa.int8()), nullable=False),
        # fragment-identity supervision, per peak
        pa.field("supervision_mask", pa.list_(pa.bool_()), nullable=False),
        pa.field("supervision_rank", pa.list_(pa.int32()), nullable=False),
        # de-duplicated candidates, per spectrum
        pa.field("candidate_neutral_formula", pa.list_(pa.string()), nullable=False),
        pa.field("candidate_ion_state", pa.list_(pa.string()), nullable=False),
        pa.field("candidate_theoretical_mz", pa.list_(pa.float64()), nullable=False),
        # peak <-> candidate edges
        pa.field("edge_peak_index", pa.list_(pa.int32()), nullable=False),
        pa.field("edge_candidate_index", pa.list_(pa.int32()), nullable=False),
        pa.field("edge_error_ppm", pa.list_(pa.float32()), nullable=False),
    ]
)

PEAK_LIST_COLUMNS = ("mzs", "intensities", "mz_decimal_places", "supervision_mask", "supervision_rank")
CANDIDATE_LIST_COLUMNS = (
    "candidate_neutral_formula",
    "candidate_ion_state",
    "candidate_theoretical_mz",
)
EDGE_LIST_COLUMNS = ("edge_peak_index", "edge_candidate_index", "edge_error_ppm")


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(paths: list[Path]) -> str:
    """Digest over a sorted list of files: each file's name and its digest."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def git_commit(repo: Path) -> dict:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", "-C", str(repo), *args],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def _list_column(table: pa.Table, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Offsets and flat values of a list column, without per-row Python objects."""
    column = table.column(name).combine_chunks()
    array = column.chunk(0) if isinstance(column, pa.ChunkedArray) else column
    offsets = array.offsets.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    values = array.values.to_numpy(zero_copy_only=False)
    return offsets, values


class TableCache:
    """Small LRU of :class:`SubformulaTable`; CE siblings are adjacent rows."""

    def __init__(self, maxsize: int = TABLE_CACHE_SIZE) -> None:
        self._entries: dict[str, SubformulaTable] = {}
        self._order: list[str] = []
        self._maxsize = maxsize
        self.hits = 0
        self.misses = 0

    def get(self, formula_text: str, *, heavy_cap: int) -> SubformulaTable:
        entry = self._entries.get(formula_text)
        if entry is not None:
            self.hits += 1
            return entry
        self.misses += 1
        table = SubformulaTable(parse_formula(formula_text), heavy_cap=heavy_cap)
        self._entries[formula_text] = table
        self._order.append(formula_text)
        if len(self._order) > self._maxsize:
            self._entries.pop(self._order.pop(0), None)
        return table


class Diagnostics:
    """Per-fold accumulators for the 64-slot diagnostic report."""

    def __init__(self) -> None:
        self.supervised_per_spectrum: list[int] = []
        self.unique_candidates: list[int] = []
        self.edges_per_spectrum: list[int] = []
        self.top64_l1_ratio: list[float] = []
        self.oracle64: list[float] = []
        self.excluded_l1_ratio: list[float] = []
        self.peaks = 0
        self.unique_peaks = 0
        self.ambiguous_peaks = 0
        self.unsupported_peaks = 0
        self.low_precision_peaks = 0
        self.supervised_peaks = 0
        self.candidates_total = 0
        self.candidates_multi_peak = 0
        self.intensity_total = 0.0
        self.intensity_top64 = 0.0
        self.energy_total = 0.0
        self.energy_top64 = 0.0
        self.zero_energy_spectra = 0

    def merge(self, other: "Diagnostics") -> None:
        """Fold one shard's counts into this fold's. Lists join, counters add.

        Every attribute is either a list of per-spectrum values or a running
        total, so shards can be accumulated in any order and the result is the
        same as having walked them one after another.
        """
        for name, value in vars(other).items():
            current = getattr(self, name)
            if isinstance(current, list):
                current.extend(value)
            else:
                setattr(self, name, current + value)

    def add(
        self,
        counts: np.ndarray,
        decimals: np.ndarray,
        intensities: np.ndarray,
        mask: np.ndarray,
        rank: np.ndarray,
        n_candidates: int,
        edge_candidate_index: np.ndarray,
    ) -> None:
        intensity = intensities.astype(np.float64, copy=False)
        slots = top_slot_mask(rank)

        self.peaks += int(counts.size)
        self.unique_peaks += int((counts == 1).sum())
        self.ambiguous_peaks += int((counts >= 2).sum())
        self.unsupported_peaks += int((counts == 0).sum())
        self.low_precision_peaks += int((decimals < MIN_SUPERVISION_DECIMALS).sum())
        self.supervised_peaks += int(mask.sum())

        self.supervised_per_spectrum.append(int(mask.sum()))
        self.unique_candidates.append(int(n_candidates))
        self.edges_per_spectrum.append(int(edge_candidate_index.size))

        self.candidates_total += int(n_candidates)
        if n_candidates:
            per_candidate = np.bincount(edge_candidate_index, minlength=n_candidates)
            self.candidates_multi_peak += int((per_candidate > 1).sum())

        total_l1 = float(intensity.sum())
        top_l1 = float(intensity[slots].sum())
        self.intensity_total += total_l1
        self.intensity_top64 += top_l1
        self.top64_l1_ratio.append(top_l1 / total_l1 if total_l1 > 0 else 0.0)
        self.excluded_l1_ratio.append(1.0 - (top_l1 / total_l1) if total_l1 > 0 else 0.0)

        squared = np.square(intensity)
        total_energy = float(squared.sum())
        top_energy = float(squared[slots].sum())
        self.energy_total += total_energy
        self.energy_top64 += top_energy
        if total_energy > 0:
            self.oracle64.append(float(np.sqrt(top_energy / total_energy)))
        else:
            self.oracle64.append(0.0)
            self.zero_energy_spectra += 1

    def report(self) -> dict:
        supervised = np.asarray(self.supervised_per_spectrum, dtype=np.int64)
        unique = np.asarray(self.unique_candidates, dtype=np.int64)
        edges = np.asarray(self.edges_per_spectrum, dtype=np.int64)
        top_l1 = np.asarray(self.top64_l1_ratio, dtype=np.float64)
        oracle = np.asarray(self.oracle64, dtype=np.float64)
        excluded = np.asarray(self.excluded_l1_ratio, dtype=np.float64)

        def dist(values: np.ndarray) -> dict:
            if values.size == 0:
                return {}
            return {
                "mean": float(values.mean()),
                "p50": float(np.percentile(values, 50)),
                "p90": float(np.percentile(values, 90)),
                "p99": float(np.percentile(values, 99)),
                "max": float(values.max()),
            }

        return {
            "spectra": int(supervised.size),
            "peaks": self.peaks,
            "fixed_slot_count": FIXED_SLOT_COUNT,
            "supervised_peaks_per_spectrum": dist(supervised),
            "spectra_over_slot_count_fraction": (
                float((supervised > FIXED_SLOT_COUNT).mean()) if supervised.size else None
            ),
            "top64_l1_intensity_retained": {
                "per_spectrum_mean": float(top_l1.mean()) if top_l1.size else None,
                "per_spectrum_p10": float(np.percentile(top_l1, 10)) if top_l1.size else None,
                "aggregate": (
                    self.intensity_top64 / self.intensity_total if self.intensity_total else None
                ),
            },
            "top64_excluded_l1_intensity_fraction": {
                "per_spectrum_mean": float(excluded.mean()) if excluded.size else None,
                "aggregate": (
                    1.0 - self.intensity_top64 / self.intensity_total
                    if self.intensity_total
                    else None
                ),
            },
            "top64_oracle_cosine_upper_bound": {
                "mean": float(oracle.mean()) if oracle.size else None,
                "p50": float(np.percentile(oracle, 50)) if oracle.size else None,
                "p10": float(np.percentile(oracle, 10)) if oracle.size else None,
                "aggregate": (
                    float(np.sqrt(self.energy_top64 / self.energy_total))
                    if self.energy_total
                    else None
                ),
            },
            "unique_candidates_per_spectrum": dist(unique),
            "edges_per_spectrum": dist(edges),
            "peak_classes": {
                "unique": self.unique_peaks,
                "ambiguous": self.ambiguous_peaks,
                "unsupported": self.unsupported_peaks,
                "low_precision": self.low_precision_peaks,
                "supervised": self.supervised_peaks,
            },
            "candidate_reuse": {
                "candidates_total": self.candidates_total,
                "candidates_matching_multiple_peaks": self.candidates_multi_peak,
                "fraction": (
                    self.candidates_multi_peak / self.candidates_total
                    if self.candidates_total
                    else None
                ),
                "edges_per_candidate": (
                    int(edges.sum()) / self.candidates_total if self.candidates_total else None
                ),
            },
            "zero_energy_spectra": self.zero_energy_spectra,
        }


def _drop_implausible(table: SubformulaTable, edges: SpectrumEdges, n_peaks: int):
    """Remove candidates no real neutral formula could be, and their edges.

    A bag member with a negative RDBE, or more monovalent atoms than its heavy
    atoms can carry, is a mass coincidence rather than a fragment; left in the
    bag it lets the likelihood reward probability mass on something that cannot
    exist. In the v1 artifact 7.9 % of candidates were of this kind, and no
    predicted formula that landed in a bag was.
    """
    keys = np.unique(edges.edge_key)
    if not keys.size:
        return edges, 0
    _, heavy_index, hydrogen = key_components(table, keys)
    plausible = np.fromiter(
        (
            is_valence_plausible(table.decode(h, y))
            for h, y in zip(heavy_index.tolist(), hydrogen.tolist(), strict=True)
        ),
        dtype=bool,
        count=keys.size,
    )
    if plausible.all():
        return edges, 0
    keep = np.isin(edges.edge_key, keys[plausible])
    peak = edges.edge_peak_index[keep]
    counts = np.bincount(peak, minlength=n_peaks).astype(
        edges.peak_candidate_counts.dtype, copy=False
    )
    return SpectrumEdges(counts, peak, edges.edge_key[keep]), int((~plausible).sum())


def _build_row(
    row: int,
    columns: dict,
    table: SubformulaTable,
    channels: tuple,
    mzs: np.ndarray,
    intensities: np.ndarray,
    decimals: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    edges = match_edges(
        table, mzs, decimals, channels, ppm=args.ppm, peak_cap=args.peak_cap
    )
    edges, dropped = _drop_implausible(table, edges, int(mzs.shape[0]))
    keys = np.unique(edges.edge_key)
    candidate_index = np.searchsorted(keys, edges.edge_key).astype(np.int32, copy=False)
    candidate_mz = theoretical_mz(table, channels, keys)

    channel_index, heavy_index, hydrogen = key_components(table, keys)
    formulas = [
        table.formula_string(h, y)
        for h, y in zip(heavy_index.tolist(), hydrogen.tolist(), strict=True)
    ]
    ion_states = [channels[c].name for c in channel_index.tolist()]

    edge_mz = candidate_mz[candidate_index] if keys.size else np.empty(0, dtype=np.float64)
    error_ppm = (
        (mzs[edges.edge_peak_index] - edge_mz) / edge_mz * 1e6
        if keys.size
        else np.empty(0, dtype=np.float64)
    )

    mask = supervision_mask(edges.peak_candidate_counts, decimals)
    rank = supervision_rank(mask, mzs, intensities)

    record = {
        "spectrum_uid": columns["spectrum_uid"][row],
        "parent_spec": columns["parent_spec"][row],
        "fold": columns["fold"][row],
        "collision_index": columns["collision_index"][row],
        "collision_energy": columns["collision_energy"][row],
        "smiles": columns["smiles"][row],
        "formula": columns["formula"][row],
        "inchikey": columns["inchikey"][row],
        "adduct": columns["adduct"][row],
        "instrument": columns["instrument"][row],
        "precursor_mz": columns["precursor_mz"][row],
        "mzs": mzs,
        "intensities": intensities,
        "mz_decimal_places": decimals,
        "supervision_mask": mask,
        "supervision_rank": rank,
        "candidate_neutral_formula": formulas,
        "candidate_ion_state": ion_states,
        "candidate_theoretical_mz": candidate_mz,
        "edge_peak_index": edges.edge_peak_index.astype(np.int32, copy=False),
        "edge_candidate_index": candidate_index,
        "edge_error_ppm": error_ppm.astype(np.float32, copy=False),
    }
    return (
        record,
        edges.peak_candidate_counts,
        mask,
        rank,
        int(keys.size),
        candidate_index,
        dropped,
    )


def _shard_rows(source_path: Path, args: argparse.Namespace, budget: int | None):
    """Every supervision row in one source shard, with its own diagnostics.

    A shard is self-contained: shard assignment hashes ``parent_spec``, so a
    molecule's collision-energy siblings all land here together and the
    subformula cache stays warm without being shared with anyone else. That is
    what lets shards be built in separate processes.
    """
    cache = TableCache()
    diagnostics = Diagnostics()
    rows: list[dict] = []
    totals = {
        "spectra": 0,
        "peaks": 0,
        "candidates": 0,
        "edges": 0,
        "parents": set(),
        "cache_hits": 0,
        "cache_misses": 0,
        "valence_dropped": 0,
    }
    if budget is not None and budget <= 0:
        return rows, diagnostics, totals

    source = pq.read_table(source_path, columns=SOURCE_COLUMNS)
    if source.num_rows:
        columns = {
            name: source.column(name).to_pylist()
            for name in SOURCE_COLUMNS
            if name not in ("mzs", "intensities", "mz_decimal_places")
        }
        mz_offsets, mz_values = _list_column(source, "mzs")
        in_offsets, in_values = _list_column(source, "intensities")
        dp_offsets, dp_values = _list_column(source, "mz_decimal_places")

        for row in range(source.num_rows):
            if budget is not None and totals["spectra"] >= budget:
                break
            mzs = mz_values[mz_offsets[row] : mz_offsets[row + 1]]
            intensities = in_values[in_offsets[row] : in_offsets[row + 1]]
            decimals = dp_values[dp_offsets[row] : dp_offsets[row + 1]]
            uid = columns["spectrum_uid"][row]

            try:
                subformulas = cache.get(columns["formula"][row], heavy_cap=args.heavy_cap)
                channels = channels_for_adduct(columns["adduct"][row])
            except (FormulaError, UnsupportedChargeError, EnumerationCapExceeded) as exc:
                fail(f"{uid}: {exc}")

            try:
                record, counts, mask, rank, n_candidates, edge_index, dropped = _build_row(
                    row, columns, subformulas, channels, mzs, intensities, decimals, args
                )
            except EnumerationCapExceeded as exc:
                fail(f"{uid}: {exc}")

            rows.append(record)
            diagnostics.add(
                counts, decimals, intensities, mask, rank, n_candidates, edge_index
            )
            totals["spectra"] += 1
            totals["peaks"] += int(mzs.shape[0])
            totals["candidates"] += n_candidates
            totals["edges"] += int(edge_index.size)
            totals["valence_dropped"] += dropped
            totals["parents"].add(columns["parent_spec"][row])

    # Reported per shard rather than globally: each worker keeps its own cache,
    # and a shard holds all of a molecule's collision-energy siblings, so this
    # still measures the reuse that matters.
    totals["cache_hits"] = cache.hits
    totals["cache_misses"] = cache.misses
    return rows, diagnostics, totals


def _build_shard(payload, budget: int | None = None):
    """Build one shard and write it. Runs in the parent or in a worker."""
    source_path, out_path, fold, shard, args = payload
    try:
        rows, diagnostics, totals = _shard_rows(Path(source_path), args, budget)
    except SystemExit as exc:  # ``fail`` inside a worker must reach the parent
        raise RuntimeError(f"{fold}/{shard_filename(shard)}: {exc}") from exc

    out_table = (
        pa.table({name: [r[name] for r in rows] for name in SCHEMA.names}, schema=SCHEMA)
        if rows
        else SCHEMA.empty_table()
    )
    pq.write_table(out_table, Path(out_path), compression="zstd")
    shard_stat = {
        "shard": shard,
        "path": f"{fold}/{shard_filename(shard)}",
        "rows": len(rows),
        "peaks": int(sum(len(r["mzs"]) for r in rows)),
        "candidates": int(sum(len(r["candidate_ion_state"]) for r in rows)),
        "edges": int(sum(len(r["edge_peak_index"]) for r in rows)),
    }
    return shard_stat, diagnostics, totals


def build(args: argparse.Namespace) -> int:
    spectra_dir = Path(args.spectra).resolve()
    out_dir = Path(args.out).resolve()
    source_manifest_path = spectra_dir / MANIFEST_NAME
    if not source_manifest_path.is_file():
        fail(f"source manifest not found: {source_manifest_path}")
    source_manifest = json.loads(source_manifest_path.read_text())
    if source_manifest["schema_version"] != SOURCE_SCHEMA_VERSION:
        fail(
            f"{source_manifest_path}: schema_version is "
            f"{source_manifest['schema_version']!r}, need {SOURCE_SCHEMA_VERSION!r}"
        )
    num_shards = source_manifest["builder"]["num_shards"]
    if num_shards != DEFAULT_NUM_SHARDS:
        fail(f"source uses {num_shards} shards, this builder expects {DEFAULT_NUM_SHARDS}")
    if out_dir.exists() and not args.overwrite:
        fail(f"output already exists: {out_dir} (pass --overwrite to replace it)")

    limits = {}
    if args.limit_spectra is not None:
        half = args.limit_spectra // 2
        limits = {"train": half, "valid": args.limit_spectra - half}

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir.parent / f"{out_dir.name}.tmp.{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    # ``func`` is argparse's subcommand callback; nothing downstream needs it
    # and it only complicates pickling the namespace out to a worker.
    worker_args = argparse.Namespace(
        **{key: value for key, value in vars(args).items() if key != "func"}
    )
    diagnostics = {fold: Diagnostics() for fold in BUILD_FOLDS}
    cache_stats = {"hits": 0, "misses": 0}
    valence_dropped = 0
    fold_stats = {}
    shard_stats: dict[str, list[dict]] = {}
    started = time.perf_counter()

    for fold in BUILD_FOLDS:
        source_fold = spectra_dir / fold
        if not source_fold.is_dir():
            fail(f"source fold missing: {source_fold}")
        (tmp_dir / fold).mkdir()
        budget = limits.get(fold)
        seen = 0
        stats = {
            "spectra": 0,
            "peaks": 0,
            "candidates": 0,
            "edges": 0,
            "parents": set(),
        }
        shard_stats[fold] = []

        payloads = []
        for shard in range(num_shards):
            source_path = source_fold / shard_filename(shard)
            if not source_path.is_file():
                fail(f"source shard missing: {source_path}")
            payloads.append(
                (
                    str(source_path),
                    str(tmp_dir / fold / shard_filename(shard)),
                    fold,
                    shard,
                    worker_args,
                )
            )

        # Shards are independent, so they go out to processes. A capped build
        # (--limit-spectra) cannot: its budget is consumed shard by shard, and
        # which rows land in the output would depend on who finished first.
        if budget is None and args.workers > 1:
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=min(args.workers, len(payloads)), mp_context=context
            ) as pool:
                results = list(pool.map(_build_shard, payloads))
        else:
            results = []
            for payload in payloads:
                remaining = None if budget is None else max(budget - seen, 0)
                result = _build_shard(payload, budget=remaining)
                seen += result[2]["spectra"]
                results.append(result)

        for shard_stat, shard_diagnostics, totals in results:
            shard_stats[fold].append(shard_stat)
            diagnostics[fold].merge(shard_diagnostics)
            stats["spectra"] += totals["spectra"]
            stats["peaks"] += totals["peaks"]
            stats["candidates"] += totals["candidates"]
            stats["edges"] += totals["edges"]
            stats["parents"].update(totals["parents"])
            cache_stats["hits"] += totals["cache_hits"]
            cache_stats["misses"] += totals["cache_misses"]
            valence_dropped += totals["valence_dropped"]

        stats["parents"] = len(stats["parents"])
        fold_stats[fold] = stats
        print(
            f"[build] {fold:<5} spectra={stats['spectra']:>7,} peaks={stats['peaks']:>9,} "
            f"candidates={stats['candidates']:>10,} edges={stats['edges']:>10,}",
            flush=True,
        )

    elapsed = time.perf_counter() - started
    total_spectra = sum(s["spectra"] for s in fold_stats.values())
    total_peaks = sum(s["peaks"] for s in fold_stats.values())

    source_shards = [
        spectra_dir / fold / shard_filename(shard)
        for fold in BUILD_FOLDS
        for shard in range(num_shards)
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "builder": {
            "script": "scripts/build_fragment_supervision.py",
            "git": git_commit(Path(__file__).resolve().parents[1]),
            "num_shards": num_shards,
            "folds": list(BUILD_FOLDS),
            "uid_format": UID_FORMAT,
            "shard_fn": SHARD_FN,
            "limit_spectra": args.limit_spectra,
            "workers": args.workers,
            "candidate_order": "ascending candidate key (ion channel, heavy mass index, hydrogen)",
            "candidate_filter": "rdbe >= 0 and monovalent atoms <= 2(C+Si) + 2 + (N+P)",
        },
        "inputs": {
            "spectra_dir": str(spectra_dir),
            "spectra_schema_version": source_manifest["schema_version"],
            "spectra_manifest_sha256": sha256_file(source_manifest_path),
            "spectra_shards_sha256": sha256_tree(source_shards),
            "source_files": {k: v["sha256"] for k, v in source_manifest["inputs"].items()},
        },
        "candidate_policy": {
            "graph_free": True,
            "magma_used": False,
            "ppm": args.ppm,
            "tolerance_rule": (
                "decimals >= 2: max(mz * ppm, 0.5 * 10**-decimals); "
                "decimals <= 1: ppm only, peak flagged low precision"
            ),
            "boundary_epsilon": BOUNDARY_EPS,
            "heavy_subformula_cap": args.heavy_cap,
            "peak_candidate_cap": args.peak_cap,
            "subformula_rule": (
                "0 <= F[e] <= precursor[e] for every element including H; empty excluded"
            ),
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
            "identity": "(neutral_formula, ion_state)",
            "unchanged_from": "formula_support_audit_v0",
        },
        "supervision_policy": {
            "mask_rule": (
                f"candidate_count > 0 AND mz_decimal_places >= {MIN_SUPERVISION_DECIMALS}"
            ),
            "excluded_from_identity_loss": ["peaks with no candidate", "low-precision peaks"],
            "raw_spectrum_filtered": False,
            "intensity_normalised": False,
            "binned": False,
            "fixed_slot_count": FIXED_SLOT_COUNT,
            "slot_order": "intensity desc, then m/z asc, then peak index asc",
            "slot_truncation_applied_to_artifact": False,
        },
        "folds": {
            fold: {
                "parents": stats["parents"],
                "spectra": stats["spectra"],
                "peaks": stats["peaks"],
                "candidates": stats["candidates"],
                "edges": stats["edges"],
            }
            for fold, stats in fold_stats.items()
        },
        "totals": {
            "spectra": total_spectra,
            "peaks": total_peaks,
            "candidates": sum(s["candidates"] for s in fold_stats.values()),
            "edges": sum(s["edges"] for s in fold_stats.values()),
        },
        "shards": shard_stats,
        "runtime": {
            "elapsed_seconds": round(elapsed, 3),
            "spectra_per_second": round(total_spectra / elapsed, 1) if elapsed else None,
            "peaks_per_second": round(total_peaks / elapsed, 1) if elapsed else None,
            "subformula_cache": {"hits": cache_stats["hits"], "misses": cache_stats["misses"]},
        },
        "candidates_dropped_by_valence_filter": valence_dropped,
        "enumeration_cap_hits": 0,
        "overflow": 0,
        "silent_truncation": 0,
        "test_fold_read": False,
        "test_metrics_computed": False,
    }
    (tmp_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")

    report = {
        "report_version": f"{SCHEMA_VERSION}_diagnostics",
        "created_utc": manifest["created_utc"],
        "purpose": (
            "documents the information lost by the fixed 64-slot design; "
            "not a gate for re-choosing the slot count"
        ),
        "fixed_slot_count": FIXED_SLOT_COUNT,
        "folds": {fold: diagnostics[fold].report() for fold in BUILD_FOLDS},
        "runtime": manifest["runtime"],
        "caps": {"enumeration_cap_hits": 0, "overflow": 0, "silent_truncation": 0},
        "test_fold_read": False,
    }
    (tmp_dir / DIAGNOSTICS_NAME).write_text(json.dumps(report, indent=2) + "\n")

    if out_dir.exists():
        trash = out_dir.parent / f"{out_dir.name}.trash.{os.getpid()}"
        out_dir.rename(trash)
        os.replace(tmp_dir, out_dir)
        shutil.rmtree(trash)
    else:
        os.replace(tmp_dir, out_dir)

    print(
        f"[build] done in {elapsed:.1f}s: {total_spectra:,} spectra / {total_peaks:,} peaks / "
        f"{manifest['totals']['edges']:,} edges -> {out_dir}",
        flush=True,
    )
    return 0


# --------------------------------------------------------------------------- verify


def verify(args: argparse.Namespace) -> int:
    spectra_dir = Path(args.spectra).resolve()
    out_dir = Path(args.out).resolve()
    manifest_path = out_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        fail(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    source_manifest = json.loads((spectra_dir / MANIFEST_NAME).read_text())
    num_shards = manifest["builder"]["num_shards"]
    limited = manifest["builder"]["limit_spectra"] is not None
    ppm = manifest["candidate_policy"]["ppm"]

    problems: list[str] = []
    seen_uids: set[str] = set()
    duplicate_uids = 0
    parent_folds: dict[str, set[str]] = {}
    parent_shards: dict[str, set[int]] = {}
    actual = {fold: {"spectra": 0, "peaks": 0} for fold in BUILD_FOLDS}
    bad_peak_index = bad_candidate_index = 0
    duplicate_identity = 0
    mask_without_edge = 0
    low_precision_supervised = 0
    mask_rule_violations = 0
    tolerance_violations = 0
    rank_violations = 0
    raw_mismatch = 0

    if (out_dir / "test").exists():
        problems.append("output contains a test fold directory")
    if manifest["test_fold_read"] or manifest["test_metrics_computed"]:
        problems.append("manifest claims the test fold was touched")

    for fold in BUILD_FOLDS:
        fold_dir = out_dir / fold
        if not fold_dir.is_dir():
            problems.append(f"{fold}: directory missing")
            continue
        files = sorted(fold_dir.glob("part-*.parquet"))
        if len(files) != num_shards:
            problems.append(f"{fold}: expected {num_shards} shards, found {len(files)}")

        for path in files:
            shard = shard_from_filename(path.name)
            if not pq.read_schema(path).equals(SCHEMA, check_metadata=False):
                problems.append(f"{fold}/{path.name}: schema differs from {SCHEMA_VERSION}")
                continue
            table = pq.read_table(path)
            if table.num_rows == 0:
                continue
            source = pq.read_table(
                spectra_dir / fold / shard_filename(shard), columns=SOURCE_COLUMNS
            )

            uids = table.column("spectrum_uid").to_pylist()
            parents = table.column("parent_spec").to_pylist()
            folds = table.column("fold").to_pylist()
            source_uids = source.column("spectrum_uid").to_pylist()[: len(uids)]
            if uids != source_uids:
                raw_mismatch += 1
                problems.append(f"{fold}/{path.name}: spectrum_uid order differs from the source")
                continue

            for name in ("mzs", "intensities", "mz_decimal_places"):
                out_off, out_val = _list_column(table, name)
                src_off, src_val = _list_column(source, name)
                for row in range(table.num_rows):
                    a = out_val[out_off[row] : out_off[row + 1]]
                    b = src_val[src_off[row] : src_off[row + 1]]
                    if a.shape != b.shape or not np.array_equal(a, b):
                        raw_mismatch += 1
                        break

            mz_off, mz_val = _list_column(table, "mzs")
            it_off, it_val = _list_column(table, "intensities")
            dp_off, dp_val = _list_column(table, "mz_decimal_places")
            mk_off, mk_val = _list_column(table, "supervision_mask")
            rk_off, rk_val = _list_column(table, "supervision_rank")
            cm_off, cm_val = _list_column(table, "candidate_theoretical_mz")
            ep_off, ep_val = _list_column(table, "edge_peak_index")
            ec_off, ec_val = _list_column(table, "edge_candidate_index")
            formulas = table.column("candidate_neutral_formula").to_pylist()
            ion_states = table.column("candidate_ion_state").to_pylist()

            for row in range(table.num_rows):
                uid = uids[row]
                if uid in seen_uids:
                    duplicate_uids += 1
                seen_uids.add(uid)
                parent_folds.setdefault(parents[row], set()).add(folds[row])
                parent_shards.setdefault(parents[row], set()).add(shard)
                if shard_of(parents[row], num_shards) != shard:
                    problems.append(f"{fold}/{path.name}: {parents[row]} is in the wrong shard")

                mzs = mz_val[mz_off[row] : mz_off[row + 1]]
                intensities = it_val[it_off[row] : it_off[row + 1]]
                decimals = dp_val[dp_off[row] : dp_off[row + 1]]
                mask = mk_val[mk_off[row] : mk_off[row + 1]].astype(bool)
                rank = rk_val[rk_off[row] : rk_off[row + 1]]
                candidate_mz = cm_val[cm_off[row] : cm_off[row + 1]]
                edge_peak = ep_val[ep_off[row] : ep_off[row + 1]]
                edge_cand = ec_val[ec_off[row] : ec_off[row + 1]]

                n_peaks = int(mzs.shape[0])
                n_candidates = int(candidate_mz.shape[0])
                actual[fold]["spectra"] += 1
                actual[fold]["peaks"] += n_peaks

                if edge_peak.size and (edge_peak.min() < 0 or edge_peak.max() >= n_peaks):
                    bad_peak_index += 1
                if edge_cand.size and (edge_cand.min() < 0 or edge_cand.max() >= n_candidates):
                    bad_candidate_index += 1
                    continue

                identities = list(
                    zip(formulas[row], ion_states[row], strict=True)
                )
                if len(set(identities)) != n_candidates:
                    duplicate_identity += 1

                has_edge = np.zeros(n_peaks, dtype=bool)
                if edge_peak.size:
                    has_edge[edge_peak] = True
                expected_mask = has_edge & (decimals >= MIN_SUPERVISION_DECIMALS)
                if not np.array_equal(mask, expected_mask):
                    mask_rule_violations += 1
                mask_without_edge += int((mask & ~has_edge).sum())
                low_precision_supervised += int(
                    (mask & (decimals < MIN_SUPERVISION_DECIMALS)).sum()
                )

                if not np.array_equal(rank, supervision_rank(mask, mzs, intensities)):
                    rank_violations += 1

                if edge_peak.size:
                    tolerance = tolerance_array(mzs[edge_peak], decimals[edge_peak], ppm)
                    delta = np.abs(mzs[edge_peak] - candidate_mz[edge_cand])
                    tolerance_violations += int((delta > tolerance + 1e-6).sum())

    for fold in BUILD_FOLDS:
        recorded = manifest["folds"][fold]
        if recorded["spectra"] != actual[fold]["spectra"]:
            problems.append(
                f"{fold}: manifest spectra {recorded['spectra']} vs {actual[fold]['spectra']}"
            )
        if recorded["peaks"] != actual[fold]["peaks"]:
            problems.append(
                f"{fold}: manifest peaks {recorded['peaks']} vs {actual[fold]['peaks']}"
            )
        if not limited:
            source_fold = source_manifest["folds"][fold]
            if source_fold["spectra"] != actual[fold]["spectra"]:
                problems.append(
                    f"{fold}: spectra_v2 has {source_fold['spectra']} spectra, "
                    f"output has {actual[fold]['spectra']}"
                )
            if source_fold["peaks"] != actual[fold]["peaks"]:
                problems.append(
                    f"{fold}: spectra_v2 has {source_fold['peaks']} peaks, "
                    f"output has {actual[fold]['peaks']}"
                )

    mixed_fold = [p for p, f in parent_folds.items() if len(f) > 1]
    mixed_shard = [p for p, s in parent_shards.items() if len(s) > 1]
    for label, count in (
        ("duplicate spectrum_uid", duplicate_uids),
        ("edges with an out-of-range peak index", bad_peak_index),
        ("edges with an out-of-range candidate index", bad_candidate_index),
        ("spectra with duplicate candidate identities", duplicate_identity),
        ("supervised peaks without an edge", mask_without_edge),
        ("low-precision peaks marked supervised", low_precision_supervised),
        ("spectra whose supervision mask breaks the rule", mask_rule_violations),
        ("spectra whose supervision rank is not reproducible", rank_violations),
        ("edges outside the mass tolerance", tolerance_violations),
        ("shards whose raw spectrum differs from spectra_v2", raw_mismatch),
        ("parents spanning folds", len(mixed_fold)),
        ("parents spanning shards", len(mixed_shard)),
    ):
        if count:
            problems.append(f"{label}: {count}")
    if manifest["silent_truncation"] or manifest["enumeration_cap_hits"] or manifest["overflow"]:
        problems.append("manifest records a cap, overflow or truncation event")

    print(f"[verify] {out_dir}")
    print(f"[verify] schema_version={manifest['schema_version']} num_shards={num_shards}")
    for fold in BUILD_FOLDS:
        stats = manifest["folds"][fold]
        print(
            f"[verify] {fold:<5} spectra={actual[fold]['spectra']:>7,} "
            f"peaks={actual[fold]['peaks']:>9,} candidates={stats['candidates']:>10,} "
            f"edges={stats['edges']:>10,}"
        )
    if problems:
        print(f"[verify] FAILED ({len(problems)} problems)", file=sys.stderr)
        for problem in problems[:40]:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("[verify] OK - all invariants hold")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    builder = sub.add_parser("build")
    builder.add_argument("--spectra", required=True, help="spectra_v2 dataset directory")
    builder.add_argument("--out", required=True, help="output directory")
    builder.add_argument("--ppm", type=float, default=DEFAULT_PPM)
    builder.add_argument("--heavy-cap", type=int, default=HEAVY_SUBFORMULA_CAP)
    builder.add_argument("--peak-cap", type=int, default=PEAK_CANDIDATE_CAP)
    builder.add_argument(
        "--limit-spectra",
        type=int,
        default=None,
        help="total spectra, split evenly between train and valid (smoke runs)",
    )
    builder.add_argument(
        "--workers",
        type=int,
        default=24,
        help="processes to build shards with; 1 forces the serial path",
    )
    builder.add_argument("--overwrite", action="store_true")
    builder.set_defaults(func=build)

    checker = sub.add_parser("verify")
    checker.add_argument("--spectra", required=True, help="spectra_v2 dataset directory")
    checker.add_argument("--out", required=True, help="directory to verify")
    checker.set_defaults(func=verify)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
