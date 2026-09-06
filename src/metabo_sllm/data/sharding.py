"""Deterministic spectrum identity and shard assignment.

Every artifact derived from a split has to place the same parent in the same
shard, so this rule lives in one place rather than being restated per builder.
The digest is SHA-256 of ``parent_spec``: Python's ``hash()`` is salted per
process and would make the layout irreproducible across runs and machines.

Sharding by ``parent_spec`` also keeps all collision-energy siblings of one
molecule in a single shard.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = [
    "FOLD_ALIASES",
    "FOLD_ORDER",
    "SHARD_FN",
    "UID_FORMAT",
    "shard_filename",
    "shard_from_filename",
    "shard_of",
    "spectrum_uid",
]

FOLD_ORDER = ("train", "valid", "test")
FOLD_ALIASES = {"val": "valid"}
UID_FORMAT = "{parent_spec}:{collision_index:02d}"
SHARD_FN = "int.from_bytes(sha256(parent_spec)[:8]) % num_shards"


def spectrum_uid(parent_spec: str, collision_index: int) -> str:
    return UID_FORMAT.format(parent_spec=parent_spec, collision_index=collision_index)


def shard_of(parent_spec: str, num_shards: int) -> int:
    digest = hashlib.sha256(parent_spec.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % num_shards


def shard_filename(shard: int) -> str:
    return f"part-{shard:05d}.parquet"


def shard_from_filename(name: str) -> int:
    return int(Path(name).stem.split("-")[1])
