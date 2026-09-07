"""Reader over a ``fragment_supervision_v1`` artifact.

The dataset hands out whole rows exactly as stored -- every peak, every
candidate, every edge.  Choosing what reaches the GPU is the collator's job,
so nothing is dropped here.

Only the folds named by the caller are opened.  The default is ``train``; the
test fold is never a default and the smoke/overfit entry points never ask for
it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from metabo_sllm.chem.formula import parse_formula
from metabo_sllm.model.input_formatter import format_row

__all__ = ["FragmentSupervisionDataset", "ROW_COLUMNS"]

SCHEMA_VERSION = "fragment_supervision_v2"
# v2 differs from v1 only in which candidates the bags hold -- the build
# drops valence-implausible ones -- and the columns are identical, so both
# versions are read.
ACCEPTED_SCHEMA_VERSIONS = ("fragment_supervision_v1", SCHEMA_VERSION)
MANIFEST_NAME = "manifest.json"

ROW_COLUMNS = (
    "spectrum_uid",
    "parent_spec",
    "fold",
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
    "supervision_mask",
    "supervision_rank",
    "candidate_neutral_formula",
    "candidate_ion_state",
    "candidate_theoretical_mz",
    "edge_peak_index",
    "edge_candidate_index",
    "edge_error_ppm",
)

_LIST_COLUMNS = frozenset(
    {
        "mzs",
        "intensities",
        "mz_decimal_places",
        "supervision_mask",
        "supervision_rank",
        "candidate_theoretical_mz",
        "edge_peak_index",
        "edge_candidate_index",
        "edge_error_ppm",
    }
)
_STRING_LIST_COLUMNS = frozenset({"candidate_neutral_formula", "candidate_ion_state"})


class FragmentSupervisionDataset:
    """Map-style dataset over one fold of a fragment-supervision artifact."""

    def __init__(
        self,
        root: str | Path,
        fold: str = "train",
        *,
        shards: Sequence[int] | None = None,
        limit: int | None = None,
        exclude_zero_target: bool = False,
        slots: int = 64,
    ) -> None:
        self.root = Path(root)
        self.fold = fold
        manifest_path = self.root / MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest["schema_version"] not in ACCEPTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"{manifest_path}: schema_version is {self.manifest['schema_version']!r}, "
                f"need one of {ACCEPTED_SCHEMA_VERSIONS!r}"
            )
        if fold not in self.manifest["builder"]["folds"]:
            raise ValueError(f"fold {fold!r} is not present in {self.root}")

        fold_dir = self.root / fold
        paths = sorted(fold_dir.glob("part-*.parquet"))
        if shards is not None:
            wanted = {int(s) for s in shards}
            paths = [p for p in paths if int(p.stem.split("-")[1]) in wanted]
        if not paths:
            raise FileNotFoundError(f"no shards selected under {fold_dir}")

        tables = []
        rows = 0
        for path in paths:
            table = pq.read_table(path, columns=list(ROW_COLUMNS))
            if table.num_rows == 0:
                continue
            tables.append(table)
            rows += table.num_rows
            if limit is not None and rows >= limit:
                break
        if not tables:
            raise ValueError(f"no rows found under {fold_dir}")
        self.table = pa.concat_tables(tables).combine_chunks()
        self.zero_target_stats = self._zero_target_stats(self.table, slots)
        if exclude_zero_target:
            keep = self._has_target(self.table, slots)
            self.table = self.table.filter(pa.array(keep))
        self.excluded_zero_target = exclude_zero_target
        if limit is not None and self.table.num_rows > limit:
            self.table = self.table.slice(0, limit)

        self._columns = {name: self.table.column(name) for name in ROW_COLUMNS}
        self._cost_table: dict[str, np.ndarray] | None = None

    def cost_table(
        self,
        *,
        tokenizer=None,
        max_text_length: int = 512,
        slots: int = 64,
    ) -> dict[str, np.ndarray]:
        """Per-row scoring cost, so batches can be built by work rather than count.

        Memory is driven by ``num_candidate_element_steps``, which spans two
        orders of magnitude between spectra; a sampler counting spectra alone
        would either starve the GPU or run it out of memory.  Everything is
        read straight from the Arrow buffers -- no row is materialised.

        ``num_tokens`` needs the tokenizer; without one it comes back as zeros
        and the token budget is inactive.
        """
        if self._cost_table is not None:
            return self._cost_table

        rows = self.table.num_rows
        rank_offsets, rank_values = self._list_column(self.table, "supervision_rank")
        peak_offsets, _ = self._list_column(self.table, "mzs")
        edge_offsets, edge_values = self._list_column(self.table, "edge_peak_index")

        num_targets = self._target_counts(self.table, slots)
        num_peaks = (peak_offsets[1:] - peak_offsets[:-1]).astype(np.int64)

        # An edge only costs anything when the peak it points at made the cut.
        in_top = (rank_values >= 0) & (rank_values < slots)
        edges_per_row = (edge_offsets[1:] - edge_offsets[:-1]).astype(np.int64)
        row_of_edge = np.repeat(np.arange(rows, dtype=np.int64), edges_per_row)
        global_peak = rank_offsets[row_of_edge] + edge_values.astype(np.int64)
        linked = in_top[global_peak]
        num_linked = np.bincount(row_of_edge[linked], minlength=rows).astype(np.int64)

        sizes: dict[str, int] = {}
        num_elements = np.empty(rows, dtype=np.int64)
        for index, formula in enumerate(self.table.column("formula").to_pylist()):
            size = sizes.get(formula)
            if size is None:
                size = len(parse_formula(formula))
                sizes[formula] = size
            num_elements[index] = size

        if tokenizer is not None:
            encoded = tokenizer(
                self.texts(), truncation=True, max_length=max_text_length, padding=False
            )["input_ids"]
            num_tokens = np.asarray([len(item) for item in encoded], dtype=np.int64)
        else:
            num_tokens = np.zeros(rows, dtype=np.int64)

        table = {
            "num_targets": num_targets,
            "num_peaks": num_peaks,
            "num_candidates_linked_to_top64": num_linked,
            "num_elements": num_elements,
            "num_candidate_element_steps": num_linked * num_elements,
            "num_tokens": num_tokens,
        }
        if tokenizer is not None:
            self._cost_table = table
        return table

    def texts(self) -> list[str]:
        """The conditioning string for every row, in dataset order."""
        columns = {
            name: self.table.column(name).to_pylist()
            for name in (
                "smiles",
                "formula",
                "adduct",
                "collision_energy",
                "instrument",
                "precursor_mz",
            )
        }
        return [
            format_row({name: values[index] for name, values in columns.items()})
            for index in range(self.table.num_rows)
        ]

    @staticmethod
    def _list_column(table: pa.Table, name: str) -> tuple[np.ndarray, np.ndarray]:
        column = table.column(name).combine_chunks()
        array = column.chunk(0) if isinstance(column, pa.ChunkedArray) else column
        offsets = array.offsets.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        return offsets, array.values.to_numpy(zero_copy_only=False)

    @classmethod
    def _target_counts(cls, table: pa.Table, slots: int) -> np.ndarray:
        """Supervision targets per row, without materialising rows."""
        offsets, values = cls._list_column(table, "supervision_rank")
        inside = ((values >= 0) & (values < slots)).astype(np.int64)
        cumulative = np.concatenate(([0], np.cumsum(inside)))
        return cumulative[offsets[1:]] - cumulative[offsets[:-1]]

    @classmethod
    def _has_target(cls, table: pa.Table, slots: int) -> np.ndarray:
        return cls._target_counts(table, slots) > 0

    @classmethod
    def _zero_target_stats(cls, table: pa.Table, slots: int) -> dict:
        counts = cls._target_counts(table, slots)
        offsets, values = cls._list_column(table, "intensities")
        cumulative = np.concatenate(([0.0], np.cumsum(values.astype(np.float64))))
        per_row = cumulative[offsets[1:]] - cumulative[offsets[:-1]]
        empty = counts == 0
        total = float(per_row.sum())
        return {
            "rows": int(counts.size),
            "zero_target_rows": int(empty.sum()),
            "zero_target_fraction": float(empty.mean()) if counts.size else 0.0,
            "zero_target_intensity_fraction": (
                float(per_row[empty].sum() / total) if total > 0 else 0.0
            ),
            "zero_target_peaks": int(
                (offsets[1:] - offsets[:-1])[empty].sum()
            ),
        }

    def __len__(self) -> int:
        return self.table.num_rows


    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        row: dict = {}
        for name, column in self._columns.items():
            value = column[index]
            if name in _LIST_COLUMNS:
                row[name] = np.asarray(value.as_py())
            elif name in _STRING_LIST_COLUMNS:
                row[name] = value.as_py()
            else:
                row[name] = value.as_py()
        # keep dtypes stable even for empty lists
        row["mzs"] = row["mzs"].astype(np.float64, copy=False).reshape(-1)
        row["intensities"] = row["intensities"].astype(np.float64, copy=False).reshape(-1)
        row["mz_decimal_places"] = row["mz_decimal_places"].astype(np.int64, copy=False).reshape(-1)
        row["supervision_mask"] = row["supervision_mask"].astype(bool, copy=False).reshape(-1)
        row["supervision_rank"] = row["supervision_rank"].astype(np.int64, copy=False).reshape(-1)
        row["edge_peak_index"] = row["edge_peak_index"].astype(np.int64, copy=False).reshape(-1)
        row["edge_candidate_index"] = (
            row["edge_candidate_index"].astype(np.int64, copy=False).reshape(-1)
        )
        row["candidate_theoretical_mz"] = (
            row["candidate_theoretical_mz"].astype(np.float64, copy=False).reshape(-1)
        )
        return row
