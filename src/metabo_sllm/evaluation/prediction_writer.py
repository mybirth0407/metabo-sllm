"""Persist validation predictions.

The artifact is written so a prediction can be re-scored later without rerunning
the model, and so nothing about the test fold can enter it: the writer records
the fold it was given and refuses ``test``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

__all__ = ["PREDICTION_SCHEMA", "write_predictions"]

PREDICTION_SCHEMA = pa.schema(
    [
        pa.field("spectrum_uid", pa.string(), nullable=False),
        pa.field("fold", pa.string(), nullable=False),
        pa.field("predicted_formulas", pa.list_(pa.string()), nullable=False),
        pa.field("predicted_ion_states", pa.list_(pa.string()), nullable=False),
        pa.field("predicted_mz", pa.list_(pa.float64()), nullable=False),
        pa.field("predicted_intensities", pa.list_(pa.float64()), nullable=False),
        pa.field("predicted_presence", pa.list_(pa.float64()), nullable=False),
        pa.field("predicted_slot_index", pa.list_(pa.int32()), nullable=False),
        pa.field("cos_at_20", pa.float64(), nullable=False),
        pa.field("cos_at_100", pa.float64(), nullable=False),
        pa.field("active_slots", pa.int32(), nullable=False),
        pa.field("zero_prediction", pa.bool_(), nullable=False),
    ]
)


def write_predictions(
    path: str | Path,
    predictions,
    scores,
    *,
    fold: str,
    metadata: dict | None = None,
) -> Path:
    """Write one row per spectrum. ``test`` is refused outright."""
    if fold == "test":
        raise ValueError("refusing to write predictions for the test fold")

    by_uid = {score.spectrum_uid: score for score in scores}
    rows = {name: [] for name in PREDICTION_SCHEMA.names}
    for prediction in predictions:
        score = by_uid.get(prediction.spectrum_uid)
        rows["spectrum_uid"].append(prediction.spectrum_uid)
        rows["fold"].append(fold)
        rows["predicted_formulas"].append([f.formula for f in prediction.fragments])
        rows["predicted_ion_states"].append([f.ion_state for f in prediction.fragments])
        rows["predicted_mz"].append([f.mz for f in prediction.fragments])
        rows["predicted_intensities"].append(
            [f.weighted_intensity for f in prediction.fragments]
        )
        rows["predicted_presence"].append([f.presence for f in prediction.fragments])
        rows["predicted_slot_index"].append([f.slot_index for f in prediction.fragments])
        rows["cos_at_20"].append(float(score.cosine[20]) if score else 0.0)
        rows["cos_at_100"].append(float(score.cosine[100]) if score else 0.0)
        rows["active_slots"].append(int(prediction.active_slots))
        rows["zero_prediction"].append(bool(score.zero_prediction) if score else True)

    table = pa.table(rows, schema=PREDICTION_SCHEMA)
    if metadata:
        table = table.replace_schema_metadata(
            {"metabo_sllm": json.dumps(metadata).encode("utf-8")}
        )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, destination, compression="zstd")
    return destination
