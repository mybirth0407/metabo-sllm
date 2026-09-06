#!/usr/bin/env python3
"""Materialise NIST23 spectra for one split into fold-partitioned Parquet shards.

One output row is a single ``(parent_spec, collision block)`` pair.  All
collision-energy siblings of a parent land in the same shard, chosen from
``SHA-256(parent_spec)`` so the layout is reproducible across processes and
machines (unlike Python's salted ``hash()``).

Two subcommands:

    build   read labels/HDF5/split, write shards + manifest.json
    verify  re-read a finished output directory and check its invariants

The build writes into a sibling temporary directory and only renames it into
place once every record parsed cleanly, so a partial run never masquerades as
a finished dataset.  Source files are opened read-only and never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from ast import literal_eval
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from metabo_sllm.data.ms_parser import MsParseError, parse_ms  # noqa: E402

SCHEMA_VERSION = "spectra_v2"
DEFAULT_NUM_SHARDS = 64
FOLD_ORDER = ("train", "valid", "test")
FOLD_ALIASES = {"val": "valid"}
MANIFEST_NAME = "manifest.json"
UID_FORMAT = "{parent_spec}:{collision_index:02d}"
SHARD_FN = "int.from_bytes(sha256(parent_spec)[:8]) % num_shards"

_COMMON_FIELDS = [
    pa.field("spectrum_uid", pa.string(), nullable=False),
    pa.field("parent_spec", pa.string(), nullable=False),
    pa.field("fold", pa.string(), nullable=False),
    pa.field("collision_index", pa.int32(), nullable=False),
    pa.field("collision_energy", pa.float64(), nullable=False),
    pa.field("collision_energy_raw", pa.string(), nullable=False),
    pa.field("smiles", pa.string(), nullable=False),
    pa.field("formula", pa.string(), nullable=False),
    pa.field("inchikey", pa.string(), nullable=False),
    pa.field("adduct", pa.string(), nullable=False),
    pa.field("instrument", pa.string(), nullable=False),
    pa.field("precursor_mz", pa.float64(), nullable=False),
    pa.field("mzs", pa.list_(pa.float64()), nullable=False),
    pa.field("intensities", pa.list_(pa.float32()), nullable=False),
]

# v2 adds the source-text decimal places of each m/z; v1 is kept so datasets
# already on disk stay verifiable with this script.
SCHEMAS = {
    "spectra_v1": pa.schema(_COMMON_FIELDS),
    "spectra_v2": pa.schema(
        [*_COMMON_FIELDS, pa.field("mz_decimal_places", pa.list_(pa.int8()), nullable=False)]
    ),
}
SCHEMA = SCHEMAS[SCHEMA_VERSION]

# Per-row list columns that must all have the same length.
LIST_COLUMNS = {
    "spectra_v1": ("mzs", "intensities"),
    "spectra_v2": ("mzs", "intensities", "mz_decimal_places"),
}

# labels.tsv column -> output column
LABEL_COLUMNS = {
    "smiles": "smiles",
    "formula": "formula",
    "inchikey": "inchikey",
    "ionization": "adduct",
    "instrument": "instrument",
}


# --------------------------------------------------------------------------- helpers


def spectrum_uid(parent_spec: str, collision_index: int) -> str:
    return UID_FORMAT.format(parent_spec=parent_spec, collision_index=collision_index)


def shard_of(parent_spec: str, num_shards: int) -> int:
    digest = hashlib.sha256(parent_spec.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % num_shards


def shard_filename(shard: int) -> str:
    return f"part-{shard:05d}.parquet"


def shard_from_filename(name: str) -> int:
    return int(Path(name).stem.split("-")[1])


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def describe_input(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


# --------------------------------------------------------------------------- inputs


def load_split(path: Path) -> dict[str, str]:
    """Read ``spec -> fold`` from a split TSV, mapping ``val`` to ``valid``."""
    frame = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    if frame.columns[:2] != ["spec", "Fold_0"]:
        fail(f"{path}: expected columns ['spec', 'Fold_0'], got {frame.columns}")

    mapping: dict[str, str] = {}
    for spec, raw_fold in zip(frame["spec"].to_list(), frame["Fold_0"].to_list(), strict=True):
        if spec in mapping:
            fail(f"{path}: duplicate spec {spec!r}")
        fold = FOLD_ALIASES.get(raw_fold, raw_fold)
        if fold not in FOLD_ORDER:
            fail(f"{path}: unknown fold {raw_fold!r} for spec {spec!r}")
        mapping[spec] = fold
    if not mapping:
        fail(f"{path}: split is empty")
    return mapping


def load_labels(path: Path, wanted: set[str]) -> dict[str, dict]:
    """Read the label rows for ``wanted`` specs, with collision energies parsed."""
    frame = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    required = {"spec", "precursor", "collision_energies", *LABEL_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        fail(f"{path}: missing columns {sorted(missing)}")

    frame = frame.filter(pl.col("spec").is_in(list(wanted)))
    records: dict[str, dict] = {}
    for row in frame.iter_rows(named=True):
        spec = row["spec"]
        if spec in records:
            fail(f"{path}: duplicate spec {spec!r}")
        try:
            energies = [str(item) for item in literal_eval(row["collision_energies"])]
        except (SyntaxError, ValueError) as exc:
            fail(f"{path}: spec {spec!r} has unparseable collision_energies: {exc}")
        try:
            precursor = float(row["precursor"])
        except (TypeError, ValueError):
            fail(f"{path}: spec {spec!r} has unparseable precursor {row['precursor']!r}")
        record = {out: row[src] for src, out in LABEL_COLUMNS.items()}
        for column, value in record.items():
            if value is None:
                fail(f"{path}: spec {spec!r} has a null {column}")
        record["precursor_mz"] = precursor
        record["collision_energies"] = energies
        records[spec] = record

    absent = sorted(wanted - set(records))
    if absent:
        fail(f"{path}: {len(absent)} split specs are missing from labels, e.g. {absent[:5]}")
    return records


def read_record(handle: h5py.File, key: str) -> bytes:
    value = handle[key][0]
    if isinstance(value, str):  # h5py may hand back str for some vlen dtypes
        return value.encode("utf-8", "surrogateescape")
    return bytes(value)


# --------------------------------------------------------------------------- build


def _table_from_rows(rows: list[dict]) -> pa.Table:
    if not rows:
        return SCHEMA.empty_table()
    columns = {name: [row[name] for row in rows] for name in SCHEMA.names}
    return pa.table(columns, schema=SCHEMA)


def build(args: argparse.Namespace) -> int:
    labels_path = Path(args.labels).resolve()
    hdf5_path = Path(args.hdf5).resolve()
    split_path = Path(args.split).resolve()
    out_path = Path(args.out).resolve()

    for path in (labels_path, hdf5_path, split_path):
        if not path.is_file():
            fail(f"input not found: {path}")
    if out_path.exists() and not args.overwrite:
        fail(f"output already exists: {out_path} (pass --overwrite to replace it)")
    if args.num_shards < 1:
        fail("--num-shards must be >= 1")

    split_map = load_split(split_path)
    parents = sorted(split_map)
    if args.limit_parents is not None:
        parents = parents[: args.limit_parents]
    labels = load_labels(labels_path, set(parents))

    groups: dict[tuple[str, int], list[str]] = defaultdict(list)
    for parent in parents:  # already sorted, so each group stays sorted
        groups[(split_map[parent], shard_of(parent, args.num_shards))].append(parent)

    print(f"[build] {len(parents)} parents from {split_path.name}", flush=True)
    print("[build] hashing inputs ...", flush=True)
    inputs = {
        "labels_tsv": describe_input(labels_path),
        "spec_files_hdf5": describe_input(hdf5_path),
        "split_tsv": describe_input(split_path),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.parent / f"{out_path.name}.tmp.{os.getpid()}"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True)

    failures: list[dict] = []
    fold_stats = {fold: {"parents": 0, "spectra": 0, "peaks": 0} for fold in FOLD_ORDER}
    shard_stats: dict[str, list[dict]] = {fold: [] for fold in FOLD_ORDER}
    seen_uids: set[str] = set()
    duplicate_uids = 0
    empty_spectra = 0
    inchikey_folds: dict[str, set[str]] = defaultdict(set)

    with h5py.File(hdf5_path, "r") as handle:
        for fold in FOLD_ORDER:
            (tmp_path / fold).mkdir()
            for shard in range(args.num_shards):
                rows: list[dict] = []
                shard_parents = 0
                shard_peaks = 0
                for parent in groups.get((fold, shard), ()):
                    key = f"{parent}.ms"
                    if key not in handle:
                        failures.append({"parent_spec": parent, "error": f"missing HDF5 key {key!r}"})
                        continue
                    label = labels[parent]
                    try:
                        blocks = parse_ms(
                            read_record(handle, key),
                            expected_energies=label["collision_energies"],
                            source=key,
                        )
                    except MsParseError as exc:
                        failures.append({"parent_spec": parent, "error": str(exc)})
                        continue

                    shard_parents += 1
                    inchikey_folds[label["inchikey"]].add(fold)
                    for block in blocks:
                        uid = spectrum_uid(parent, block.collision_index)
                        if uid in seen_uids:
                            duplicate_uids += 1
                        seen_uids.add(uid)
                        if len(block) == 0:
                            empty_spectra += 1
                        shard_peaks += len(block)
                        rows.append(
                            {
                                "spectrum_uid": uid,
                                "parent_spec": parent,
                                "fold": fold,
                                "collision_index": block.collision_index,
                                "collision_energy": block.collision_energy,
                                "collision_energy_raw": block.collision_energy_raw,
                                "smiles": label["smiles"],
                                "formula": label["formula"],
                                "inchikey": label["inchikey"],
                                "adduct": label["adduct"],
                                "instrument": label["instrument"],
                                "precursor_mz": label["precursor_mz"],
                                "mzs": block.mzs,
                                "intensities": block.intensities,
                                "mz_decimal_places": block.mz_decimal_places,
                            }
                        )

                # Parents arrive sorted and blocks come out in file order, which is
                # collision_index order, so rows are already (parent_spec, collision_index).
                rows.sort(key=lambda row: (row["parent_spec"], row["collision_index"]))
                target = tmp_path / fold / shard_filename(shard)
                pq.write_table(_table_from_rows(rows), target, compression="zstd")

                fold_stats[fold]["parents"] += shard_parents
                fold_stats[fold]["spectra"] += len(rows)
                fold_stats[fold]["peaks"] += shard_peaks
                shard_stats[fold].append(
                    {
                        "shard": shard,
                        "path": f"{fold}/{shard_filename(shard)}",
                        "rows": len(rows),
                        "parents": shard_parents,
                        "peaks": shard_peaks,
                    }
                )
            stats = fold_stats[fold]
            print(
                f"[build] {fold:<5} parents={stats['parents']:>7,} "
                f"spectra={stats['spectra']:>8,} peaks={stats['peaks']:>10,}",
                flush=True,
            )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "builder": {
            "script": "scripts/build_nist23_spectra.py",
            "num_shards": args.num_shards,
            "limit_parents": args.limit_parents,
            "uid_format": UID_FORMAT,
            "shard_fn": SHARD_FN,
            "fold_order": list(FOLD_ORDER),
            "fold_map": FOLD_ALIASES,
        },
        "inputs": inputs,
        "arrow_schema": [[field.name, str(field.type)] for field in SCHEMA],
        "folds": fold_stats,
        "totals": {
            "parents": sum(s["parents"] for s in fold_stats.values()),
            "spectra": sum(s["spectra"] for s in fold_stats.values()),
            "peaks": sum(s["peaks"] for s in fold_stats.values()),
        },
        "shards": shard_stats,
        "n_parse_failures": len(failures),
        "parse_failures": failures[:100],
        "n_empty_spectra": empty_spectra,
        "n_duplicate_spectrum_uid": duplicate_uids,
        # Diagnostic only, not a gate: the split file keys on spec, so a molecule
        # can in principle appear in two folds without any spec appearing twice.
        "n_inchikey_cross_fold": sum(1 for folds in inchikey_folds.values() if len(folds) > 1),
    }
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")

    if failures:
        print(f"[build] {len(failures)} parse failures; leaving output at {tmp_path}", file=sys.stderr)
        for failure in failures[:10]:
            print(f"  {failure['parent_spec']}: {failure['error']}", file=sys.stderr)
        return 1

    if out_path.exists():
        trash = out_path.parent / f"{out_path.name}.trash.{os.getpid()}"
        out_path.rename(trash)
        os.replace(tmp_path, out_path)
        shutil.rmtree(trash)
    else:
        os.replace(tmp_path, out_path)

    totals = manifest["totals"]
    print(
        f"[build] done: {totals['parents']:,} parents / {totals['spectra']:,} spectra / "
        f"{totals['peaks']:,} peaks -> {out_path}",
        flush=True,
    )
    return 0


# --------------------------------------------------------------------------- verify


def verify(args: argparse.Namespace) -> int:
    out_path = Path(args.out).resolve()
    manifest_path = out_path / MANIFEST_NAME
    if not manifest_path.is_file():
        fail(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    num_shards = manifest["builder"]["num_shards"]
    version = manifest["schema_version"]
    if version not in SCHEMAS:
        fail(f"{manifest_path}: unknown schema_version {version!r}")
    schema = SCHEMAS[version]
    list_columns = LIST_COLUMNS[version]

    problems: list[str] = []
    actual = {fold: {"parents": set(), "spectra": 0, "peaks": 0} for fold in FOLD_ORDER}
    seen_uids: set[str] = set()
    duplicate_uids = 0
    parent_folds: dict[str, set[str]] = defaultdict(set)
    length_mismatch = 0
    shard_mismatch = 0
    unsorted_shards: list[str] = []

    for fold in FOLD_ORDER:
        fold_dir = out_path / fold
        if not fold_dir.is_dir():
            problems.append(f"{fold}: directory missing")
            continue
        files = sorted(fold_dir.glob("part-*.parquet"))
        if len(files) != num_shards:
            problems.append(f"{fold}: expected {num_shards} shards, found {len(files)}")
        for path in files:
            shard = shard_from_filename(path.name)
            if not pq.read_schema(path).equals(schema, check_metadata=False):
                problems.append(f"{fold}/{path.name}: schema differs from {version}")
                continue
            # Only the columns the checks need; peak values stay in Arrow buffers
            # because list lengths come from the offsets alone.
            table = pq.read_table(
                path,
                columns=[
                    "spectrum_uid",
                    "parent_spec",
                    "fold",
                    "collision_index",
                    *list_columns,
                ],
            )
            if table.num_rows == 0:
                continue

            uids = table.column("spectrum_uid").to_pylist()
            specs = table.column("parent_spec").to_pylist()
            indices = table.column("collision_index").to_pylist()
            folds = set(table.column("fold").to_pylist())
            lengths = {
                name: pc.list_value_length(table.column(name)).to_numpy(zero_copy_only=False)
                for name in list_columns
            }
            mz_lengths = lengths["mzs"]

            if folds != {fold}:
                problems.append(f"{fold}/{path.name}: fold column holds {sorted(folds)}")
            for name, values in lengths.items():
                if name != "mzs":
                    length_mismatch += int(np.count_nonzero(values != mz_lengths))

            for uid in uids:
                if uid in seen_uids:
                    duplicate_uids += 1
                seen_uids.add(uid)
            for spec in specs:
                parent_folds[spec].add(fold)
                if shard_of(spec, num_shards) != shard:
                    shard_mismatch += 1

            keys = list(zip(specs, indices, strict=True))
            if keys != sorted(keys):
                unsorted_shards.append(f"{fold}/{path.name}")

            actual[fold]["parents"].update(specs)
            actual[fold]["spectra"] += table.num_rows
            actual[fold]["peaks"] += int(mz_lengths.sum())

    for fold in FOLD_ORDER:
        recorded = manifest["folds"].get(fold, {})
        for field, value in (
            ("parents", len(actual[fold]["parents"])),
            ("spectra", actual[fold]["spectra"]),
            ("peaks", actual[fold]["peaks"]),
        ):
            if recorded.get(field) != value:
                problems.append(
                    f"{fold}: manifest {field}={recorded.get(field)} but Parquet holds {value}"
                )

    if duplicate_uids:
        problems.append(f"duplicate spectrum_uid: {duplicate_uids}")
    mixed = [spec for spec, folds in parent_folds.items() if len(folds) > 1]
    if mixed:
        problems.append(f"parents spanning several folds: {len(mixed)} (e.g. {mixed[:5]})")
    if length_mismatch:
        problems.append(f"rows where a list column disagrees in length with mzs: {length_mismatch}")
    if manifest["n_parse_failures"]:
        problems.append(f"manifest records {manifest['n_parse_failures']} parse failures")
    if shard_mismatch:
        problems.append(f"rows in the wrong shard for SHA-256(parent_spec): {shard_mismatch}")
    if unsorted_shards:
        problems.append(
            f"shards not sorted by (parent_spec, collision_index): "
            f"{len(unsorted_shards)} (e.g. {unsorted_shards[:3]})"
        )

    print(f"[verify] {out_path}")
    print(f"[verify] schema_version={manifest['schema_version']} num_shards={num_shards}")
    for fold in FOLD_ORDER:
        stats = actual[fold]
        print(
            f"[verify] {fold:<5} parents={len(stats['parents']):>7,} "
            f"spectra={stats['spectra']:>8,} peaks={stats['peaks']:>10,}"
        )
    print(
        f"[verify] totals parents={sum(len(s['parents']) for s in actual.values()):,} "
        f"spectra={sum(s['spectra'] for s in actual.values()):,} "
        f"peaks={sum(s['peaks'] for s in actual.values()):,}"
    )
    print(
        f"[verify] empty_spectra={manifest['n_empty_spectra']} "
        f"parse_failures={manifest['n_parse_failures']} "
        f"inchikey_cross_fold={manifest['n_inchikey_cross_fold']}"
    )

    if problems:
        print(f"[verify] FAILED ({len(problems)} problems)", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("[verify] OK - all checks passed")
    return 0


# --------------------------------------------------------------------------- cli


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    builder = sub.add_parser("build", help="write shards + manifest for one split")
    builder.add_argument("--labels", required=True, help="path to labels.tsv")
    builder.add_argument("--hdf5", required=True, help="path to spec_files.hdf5")
    builder.add_argument("--split", required=True, help="path to the split TSV")
    builder.add_argument("--out", required=True, help="output dataset directory")
    builder.add_argument("--num-shards", type=int, default=DEFAULT_NUM_SHARDS)
    builder.add_argument(
        "--limit-parents",
        type=int,
        default=None,
        help="build only the first N parents in sorted order (smoke runs)",
    )
    builder.add_argument("--overwrite", action="store_true", help="replace an existing output")
    builder.set_defaults(func=build)

    checker = sub.add_parser("verify", help="check a finished output directory")
    checker.add_argument("--out", required=True, help="dataset directory to verify")
    checker.set_defaults(func=verify)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
