"""Tests for the NIST23 ``.ms`` parser contract.

Run with the package on the path::

    PYTHONPATH=src python3 -m pytest tests/test_ms_parser.py -q
"""

from __future__ import annotations

from ast import literal_eval
from pathlib import Path

import numpy as np
import pytest

from metabo_sllm.data.ms_parser import MsParseError, parse_ms

NIST23 = Path("/NHNHOME/26moe001_B/BASE/metabo_data/data/spec_datasets/nist23")
HDF5 = NIST23 / "spec_files.hdf5"
LABELS = NIST23 / "labels.tsv"
EXAMPLE = "nist_1035492"

HEADER = b">compound Example\n>formula C2H6O\n#NUM PEAKS 3\n#smiles CCO\n\n"

TWO_BLOCKS = HEADER + (
    b">collision 10\n"
    b"100.5 200.25\n"
    b"110.0 0.0\n"
    b"\n"
    b">collision 20\n"
    b"50.125 1.5\n"
)


def test_two_blocks_parse_with_declared_dtypes():
    blocks = parse_ms(TWO_BLOCKS)

    assert [b.collision_index for b in blocks] == [0, 1]
    assert [b.collision_energy for b in blocks] == [10.0, 20.0]
    assert [b.collision_energy_raw for b in blocks] == ["10", "20"]
    assert [len(b) for b in blocks] == [2, 1]

    first, second = blocks
    assert first.mzs.dtype == np.float64
    assert first.intensities.dtype == np.float32
    assert first.mzs.tolist() == [100.5, 110.0]
    assert first.intensities.tolist() == [200.25, 0.0]
    assert second.mzs.tolist() == [50.125]
    assert second.intensities.tolist() == [1.5]


def test_peaks_are_neither_sorted_nor_merged():
    content = HEADER + b">collision 5\n300.0 1.0\n100.0 2.0\n100.0 3.0\n200.0 4.0\n"
    (block,) = parse_ms(content)

    assert block.mzs.tolist() == [300.0, 100.0, 100.0, 200.0]
    assert block.intensities.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_arrays_are_read_only():
    (block,) = parse_ms(HEADER + b">collision 5\n100.0 1.0\n")
    with pytest.raises(ValueError):
        block.mzs[0] = 0.0
    with pytest.raises(ValueError):
        block.intensities[0] = 0.0
    with pytest.raises(ValueError):
        block.mz_decimal_places[0] = 0


def test_mz_decimal_places_come_from_the_source_text():
    content = HEADER + b">collision 5\n100 1.0\n100.1 1.0\n100.10 1.0\n100.1000 1.0\n"
    (block,) = parse_ms(content)

    assert block.mz_decimal_places.dtype == np.int8
    assert block.mz_decimal_places.tolist() == [0, 1, 2, 4]
    # the float value cannot distinguish "100.1" from "100.1000"
    assert block.mzs.tolist() == [100.0, 100.1, 100.1, 100.1]


def test_list_lengths_always_agree():
    blocks = parse_ms(TWO_BLOCKS)
    for block in blocks:
        assert block.mzs.shape == block.intensities.shape == block.mz_decimal_places.shape
        assert len(block) == block.mz_decimal_places.shape[0]


def test_empty_block_has_empty_decimal_places():
    blocks = parse_ms(HEADER + b">collision 5\n>collision 6\n100.0 1.0\n")

    assert blocks[0].mz_decimal_places.shape == (0,)
    assert blocks[0].mz_decimal_places.dtype == np.int8


def test_crlf_and_blank_lines_are_tolerated():
    content = HEADER.replace(b"\n", b"\r\n") + b">collision 5\r\n\r\n100.0 1.0\r\n\r\n"
    (block,) = parse_ms(content)

    assert block.collision_energy_raw == "5"
    assert block.mzs.tolist() == [100.0]


def test_block_without_peaks_is_allowed_and_empty():
    blocks = parse_ms(HEADER + b">collision 5\n>collision 6\n100.0 1.0\n")

    assert len(blocks[0]) == 0
    assert blocks[0].mzs.dtype == np.float64
    assert blocks[0].intensities.dtype == np.float32
    assert len(blocks[1]) == 1


def test_float_collision_energy_keeps_its_raw_token():
    (block,) = parse_ms(HEADER + b">collision 35.5\n100.0 1.0\n")

    assert block.collision_energy == 35.5
    assert block.collision_energy_raw == "35.5"


@pytest.mark.parametrize(
    "peak_line",
    [
        b"100.0\n",  # one token
        b"100.0 1.0 2.0\n",  # three tokens
        b"abc 1.0\n",  # non-numeric m/z
        b"100.0 xyz\n",  # non-numeric intensity
        b"nan 1.0\n",  # NaN m/z
        b"inf 1.0\n",  # Inf m/z
        b"100.0 nan\n",  # NaN intensity
        b"100.0 -inf\n",  # -Inf intensity
        b"100.0 -1.0\n",  # negative intensity
        b"-100.0 1.0\n",  # negative m/z
        b"0.0 1.0\n",  # zero m/z
    ],
)
def test_malformed_peak_lines_raise(peak_line):
    with pytest.raises(MsParseError):
        parse_ms(HEADER + b">collision 5\n" + peak_line)


def test_peak_before_first_block_raises():
    with pytest.raises(MsParseError, match="before the first"):
        parse_ms(b">compound Example\n100.0 1.0\n>collision 5\n110.0 1.0\n")


@pytest.mark.parametrize("meta", [b"#NUM PEAKS 2\n", b">formula C2H6O\n"])
def test_metadata_inside_a_block_raises(meta):
    with pytest.raises(MsParseError, match="metadata line inside"):
        parse_ms(HEADER + b">collision 5\n100.0 1.0\n" + meta + b"110.0 1.0\n")


@pytest.mark.parametrize("header", [b">collision\n", b">collision 5 6\n", b">collision abc\n"])
def test_malformed_collision_header_raises(header):
    with pytest.raises(MsParseError):
        parse_ms(HEADER + header + b"100.0 1.0\n")


def test_expected_energies_match():
    assert len(parse_ms(TWO_BLOCKS, expected_energies=["10", "20"])) == 2


@pytest.mark.parametrize(
    "expected",
    [
        ["10"],  # too few
        ["10", "20", "30"],  # too many
        ["20", "10"],  # wrong order
        ["10", "21"],  # wrong value
    ],
)
def test_expected_energies_mismatch_raises(expected):
    with pytest.raises(MsParseError, match="collision-energy mismatch"):
        parse_ms(TWO_BLOCKS, expected_energies=expected)


def test_non_bytes_input_raises():
    with pytest.raises(MsParseError, match="expected bytes"):
        parse_ms(TWO_BLOCKS.decode("ascii"))


# --------------------------------------------------------------------------- real data


def _load_example() -> bytes:
    import h5py

    with h5py.File(HDF5, "r") as handle:
        return bytes(handle[f"{EXAMPLE}.ms"][0])


@pytest.mark.skipif(not HDF5.is_file(), reason="NIST23 HDF5 not available")
def test_real_record_matches_known_values():
    blocks = parse_ms(_load_example(), source=f"{EXAMPLE}.ms")

    assert len(blocks) == 11
    assert blocks[0].collision_energy_raw == "2"
    assert blocks[0].collision_energy == 2.0
    assert len(blocks[0]) == 8
    assert blocks[0].mzs[0] == pytest.approx(339.29)
    assert blocks[0].intensities[0] == pytest.approx(17.78)
    assert [b.collision_index for b in blocks] == list(range(11))
    # "339.29" and "17.78" are printed with two decimals in this record
    assert blocks[0].mz_decimal_places[0] == 2
    for block in blocks:
        assert block.mzs.shape == block.intensities.shape == block.mz_decimal_places.shape


@pytest.mark.skipif(
    not (HDF5.is_file() and LABELS.is_file()), reason="NIST23 inputs not available"
)
def test_real_record_agrees_with_labels_collision_energies():
    energies = None
    with LABELS.open() as handle:
        header = handle.readline().rstrip("\n").split("\t")
        spec_at, ce_at = header.index("spec"), header.index("collision_energies")
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if fields[spec_at] == EXAMPLE:
                energies = [str(item) for item in literal_eval(fields[ce_at])]
                break
    assert energies is not None, f"{EXAMPLE} missing from labels.tsv"

    blocks = parse_ms(_load_example(), expected_energies=energies, source=f"{EXAMPLE}.ms")
    assert [b.collision_energy_raw for b in blocks] == energies
