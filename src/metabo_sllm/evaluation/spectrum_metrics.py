"""Binned cosine against the observed spectrum, ms-pred style.

The experimental spectrum keeps every bin; the prediction is cut to its top
``K`` bins before scoring.  That asymmetry is the point of ``cos@K``: a model
is not allowed to cover the target by predicting everywhere, but it is not
punished for peaks it never claimed either.

Same-bin handling differs by side and matches ms-pred: experimental
intensities in one bin are *max*-pooled, predicted ones are summed, because two
predicted fragments landing in the same bin really do add up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "EVALUATION_SPACES",
    "PRIMARY_SPACE",
    "BinningConfig",
    "SpectrumScore",
    "bin_experimental",
    "bin_index",
    "bin_prediction",
    "cosine_at_k",
    "observation_in_space",
    "score_prediction",
    "summarise_scores",
]


@dataclass(frozen=True)
class BinningConfig:
    """Canonical ms-pred evaluation grid.

    ``num_bins`` and ``ppm_tol`` come from the NIST23 baseline configs on this
    machine (glacier and massformer ``args.yaml``); ``upper_limit`` is
    ms-pred's 1500 Da default, which those 15000 bins span.
    """

    upper_limit: float = 1500.0
    num_bins: int = 15000
    min_pred_intensity: float = 1.0e-5
    top_k: tuple[int, ...] = (20, 100)


# The intensity head is trained against ``sqrt(y)/||sqrt(y)||`` over the whole
# spectrum, so its output is a square-root-space quantity.  ``canonical_sqrt``
# is therefore the space where prediction and observation actually match, and
# it is ms-pred's own contract: the square root is applied once, at
# preprocessing (``common/misc_utils.py``), and never again at evaluation.
# ``legacy_raw`` is what this project reported before that was pinned down -- a
# square-root prediction against a raw observation.  It is kept and reported
# alongside so older runs stay comparable, not because it measures the model.
PRIMARY_SPACE = "canonical_sqrt"
EVALUATION_SPACES = (PRIMARY_SPACE, "legacy_raw")


def observation_in_space(intensity: np.ndarray, space: str) -> np.ndarray:
    """Bring the observation into ``space``. The prediction is never touched."""
    intensity = np.clip(np.asarray(intensity, dtype=np.float64), 0.0, None)
    if space == "canonical_sqrt":
        return np.sqrt(intensity)
    if space == "legacy_raw":
        return intensity
    raise ValueError(f"unknown intensity space {space!r}")


def bin_index(mz: np.ndarray, config: BinningConfig | None = None) -> np.ndarray:
    """``floor(mz * (num_bins - 1) / upper_limit) + 1``."""
    config = config or BinningConfig()
    scaled = np.asarray(mz, dtype=np.float64) * ((config.num_bins - 1) / config.upper_limit)
    return np.floor(scaled).astype(np.int64) + 1


def _in_range(index: np.ndarray, config: BinningConfig) -> np.ndarray:
    return (index >= 0) & (index < config.num_bins)


def bin_experimental(
    mz: np.ndarray, intensity: np.ndarray, config: BinningConfig | None = None
) -> np.ndarray:
    """Observed spectrum on the grid; collisions take the maximum."""
    config = config or BinningConfig()
    output = np.zeros(config.num_bins, dtype=np.float64)
    mz = np.asarray(mz, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    if mz.size == 0:
        return output
    index = bin_index(mz, config)
    keep = _in_range(index, config)
    if not keep.any():
        return output
    np.maximum.at(output, index[keep], intensity[keep])
    return output


def bin_prediction(
    mz: np.ndarray, intensity: np.ndarray, config: BinningConfig | None = None
) -> tuple[np.ndarray, int]:
    """Predicted spectrum on the grid; collisions add. Returns (bins, dropped)."""
    config = config or BinningConfig()
    output = np.zeros(config.num_bins, dtype=np.float64)
    mz = np.asarray(mz, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    if mz.size == 0:
        return output, 0
    index = bin_index(mz, config)
    keep = _in_range(index, config)
    np.add.at(output, index[keep], intensity[keep])
    peak = output.max()
    if peak > 0:
        output /= peak
        output[output < config.min_pred_intensity] = 0.0
    return output, int((~keep).sum())


def cosine_at_k(prediction: np.ndarray, experimental: np.ndarray, k: int) -> float:
    """Cosine after keeping only the ``k`` strongest predicted bins."""
    prediction = np.asarray(prediction, dtype=np.float64)
    experimental = np.asarray(experimental, dtype=np.float64)
    nonzero = int(np.count_nonzero(prediction))
    if nonzero == 0:
        return 0.0
    if nonzero > k:
        cut = np.argpartition(prediction, -k)[-k:]
        trimmed = np.zeros_like(prediction)
        trimmed[cut] = prediction[cut]
        prediction = trimmed
    denominator = np.linalg.norm(prediction) * np.linalg.norm(experimental)
    if denominator == 0.0:
        return 0.0
    return float(prediction @ experimental / denominator)


@dataclass
class SpectrumScore:
    spectrum_uid: str
    # space -> k -> value.  Nested by space so that no reader can pick up a
    # cosine without saying which space it belongs to.
    cosine: dict[str, dict[int, float]]
    predicted_peaks: int
    predicted_bins: int
    duplicate_bin_collisions: int
    out_of_range: int
    zero_prediction: bool


def score_prediction(
    spectrum_uid: str,
    predicted_mz: np.ndarray,
    predicted_intensity: np.ndarray,
    experimental_mz: np.ndarray,
    experimental_intensity: np.ndarray,
    config: BinningConfig | None = None,
) -> SpectrumScore:
    """``cos@K`` in every evaluation space, plus what the prediction looked like."""
    config = config or BinningConfig()
    predicted_bins, dropped = bin_prediction(predicted_mz, predicted_intensity, config)
    cosine = {}
    for space in EVALUATION_SPACES:
        experimental_bins = bin_experimental(
            experimental_mz, observation_in_space(experimental_intensity, space), config
        )
        cosine[space] = {
            k: cosine_at_k(predicted_bins, experimental_bins, k) for k in config.top_k
        }

    occupied = int(np.count_nonzero(predicted_bins))
    index = bin_index(np.asarray(predicted_mz, dtype=np.float64), config)
    inside = int(_in_range(index, config).sum()) if index.size else 0
    return SpectrumScore(
        spectrum_uid=spectrum_uid,
        cosine=cosine,
        predicted_peaks=int(np.asarray(predicted_mz).size),
        predicted_bins=occupied,
        duplicate_bin_collisions=max(0, inside - occupied),
        out_of_range=dropped,
        zero_prediction=occupied == 0,
    )


def summarise_scores(scores: list[SpectrumScore], config: BinningConfig | None = None) -> dict:
    """Fold-level summary: the distribution of each ``cos@K`` and prediction shape."""
    config = config or BinningConfig()
    if not scores:
        return {"spectra": 0}

    # Each space gets its own block, and there is no bare ``cos@K`` key: a run
    # written before the spaces were separated is then trivially distinguishable
    # from one written after, rather than silently changing meaning.
    summary: dict = {"spectra": len(scores), "primary_space": PRIMARY_SPACE}
    for space in EVALUATION_SPACES:
        block = {}
        for k in config.top_k:
            values = np.asarray([score.cosine[space][k] for score in scores], dtype=np.float64)
            block[f"cos@{k}"] = {
                "mean": float(values.mean()),
                "median": float(np.percentile(values, 50)),
                "p10": float(np.percentile(values, 10)),
                "p90": float(np.percentile(values, 90)),
            }
        summary[space] = block
    peaks = np.asarray([score.predicted_peaks for score in scores], dtype=np.float64)
    bins = np.asarray([score.predicted_bins for score in scores], dtype=np.float64)
    collisions = np.asarray([score.duplicate_bin_collisions for score in scores])
    summary["predicted_peaks"] = {
        "mean": float(peaks.mean()),
        "median": float(np.percentile(peaks, 50)),
        "max": int(peaks.max()),
    }
    summary["predicted_active_bins_mean"] = float(bins.mean())
    summary["zero_prediction_spectra"] = int(sum(score.zero_prediction for score in scores))
    summary["out_of_range_predictions"] = int(sum(score.out_of_range for score in scores))
    summary["duplicate_bin_collisions"] = int(collisions.sum())
    summary["duplicate_bin_rate"] = (
        float(collisions.sum() / peaks.sum()) if peaks.sum() > 0 else 0.0
    )
    return summary
