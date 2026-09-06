"""Tests pinning the shard-assignment and spectrum-identity rules.

These values are baked into every artifact on disk, so a change here is a
change to the dataset layout, not a refactor.
"""

from __future__ import annotations

import pytest

from metabo_sllm.data.sharding import (
    shard_filename,
    shard_from_filename,
    shard_of,
    spectrum_uid,
)


@pytest.mark.parametrize(
    ("parent_spec", "expected"),
    [("nist_1035492", 12), ("nist_1035166", 41), ("nist_1035697", 60)],
)
def test_shard_assignment_is_pinned(parent_spec, expected):
    assert shard_of(parent_spec, 64) == expected


def test_shard_assignment_is_stable_across_calls():
    first = [shard_of(f"nist_{i}", 64) for i in range(200)]
    second = [shard_of(f"nist_{i}", 64) for i in range(200)]
    assert first == second
    assert all(0 <= s < 64 for s in first)


def test_shard_assignment_spreads_across_all_shards():
    shards = {shard_of(f"nist_{i}", 64) for i in range(20_000)}
    assert shards == set(range(64))


def test_spectrum_uid_format():
    assert spectrum_uid("nist_1035492", 0) == "nist_1035492:00"
    assert spectrum_uid("nist_1035492", 7) == "nist_1035492:07"
    assert spectrum_uid("nist_1035492", 10) == "nist_1035492:10"
    assert spectrum_uid("nist_1035492", 123) == "nist_1035492:123"


def test_shard_filename_round_trip():
    for shard in (0, 7, 63):
        name = shard_filename(shard)
        assert name == f"part-{shard:05d}.parquet"
        assert shard_from_filename(name) == shard
