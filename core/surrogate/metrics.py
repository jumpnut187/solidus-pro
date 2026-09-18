"""Accuracy metrics for surrogate predictions vs. the full solver.

Every metric here is a plain comparison of two arrays.  Callers decide which
cells to compare; ``schedule_evaluation_mask`` gives the standard choice
(active, non-chill cells with a positive reference end time) so that empty
grid cells, which trivially predict 0 vs 0, never inflate a headline.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from core.part_manager import PART_TYPE_CHILL
from core.surrogate.fields import PART_TYPE_DIVISOR

#: Tolerance on channel 1 (part_type / PART_TYPE_DIVISOR) when recovering chill
#: cells; matches core/deep_model/dataset.py so both lanes score the same cells.
CHILL_CHANNEL_TOL = 0.05

_NAN_FIELD: Dict[str, Any] = {
    "mae": float("nan"),
    "rmse": float("nan"),
    "max_error": float("nan"),
    "p95_error": float("nan"),
    "relative_rmse": float("nan"),
    "baseline_rmse": float("nan"),
    "skill": float("nan"),
    "n_compared": 0,
}


def constant_mean_baseline_rmse(ref: np.ndarray) -> float:
    """RMSE of predicting the reference mean everywhere (population std)."""
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((ref - ref.mean()) ** 2)))


def train_mean_baseline_rmse(train_ref: np.ndarray, val_ref: np.ndarray) -> float:
    """RMSE on ``val_ref`` of the constant predictor equal to ``mean(train_ref)``.

    This is the baseline a realizable constant model achieves: it is fit on
    the training targets and scored on validation.  ``constant_mean_baseline_rmse``
    on the validation targets alone is the *oracle* constant (val std) and
    understates skill; use this function for manifests.
    """
    train_ref = np.asarray(train_ref, dtype=np.float64).reshape(-1)
    val_ref = np.asarray(val_ref, dtype=np.float64).reshape(-1)
    train_ref = train_ref[np.isfinite(train_ref)]
    val_ref = val_ref[np.isfinite(val_ref)]
    if train_ref.size == 0 or val_ref.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((val_ref - train_ref.mean()) ** 2)))


def apply_train_baseline(
    schedule_metrics: Dict[str, Dict[str, Any]], baseline_rmse: float, field: str = "end_time"
) -> Dict[str, Dict[str, Any]]:
    """Overwrite ``baseline_rmse`` / ``skill`` of one field with a train-mean baseline."""
    entry = schedule_metrics.get(field)
    if entry is None:
        return schedule_metrics
    entry["baseline_rmse"] = float(baseline_rmse)
    rmse = entry.get("rmse")
    entry["skill"] = skill_score(
        float(rmse) if rmse is not None else float("nan"), float(baseline_rmse)
    )
    return schedule_metrics


def skill_score(rmse: float, baseline_rmse: float) -> float:
    """``1 - rmse / baseline_rmse`` clipped to [0, 1]; NaN when undefined."""
    if not np.isfinite(rmse) or not np.isfinite(baseline_rmse):
        return float("nan")
    if baseline_rmse <= 1e-12:
        return 1.0 if rmse <= 1e-12 else 0.0
    return float(np.clip(1.0 - rmse / baseline_rmse, 0.0, 1.0))


def compute_field_metrics(pred: np.ndarray, ref: np.ndarray) -> Dict[str, Any]:
    """Compare two per-voxel fields and summarize the error distribution.

    Args:
        pred: Predicted field (float array).
        ref: Reference field (float array).

    Returns:
        Dict with ``mae``, ``rmse``, ``max_error``, ``p95_error``,
        ``relative_rmse`` (relative to the reference RMS), ``baseline_rmse``
        (constant-mean predictor on the same cells), ``skill``
        (``1 - rmse / baseline_rmse`` clipped to [0, 1]) and ``n_compared``.
    """
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if pred.shape != ref.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs ref {ref.shape}")

    finite = np.isfinite(pred) & np.isfinite(ref)
    if not np.any(finite):
        return dict(_NAN_FIELD)

    pred_f, ref_f = pred[finite], ref[finite]
    error = np.abs(pred_f - ref_f)
    mae = float(np.mean(error))
    rmse = float(np.sqrt(np.mean((pred_f - ref_f) ** 2)))
    max_error = float(np.max(error))
    p95 = float(np.percentile(error, 95))
    ref_rms = float(np.sqrt(np.mean(ref_f ** 2)))
    rel = rmse / ref_rms if ref_rms > 1e-12 else float("nan")
    baseline = constant_mean_baseline_rmse(ref_f)
    return {
        "mae": mae,
        "rmse": rmse,
        "max_error": max_error,
        "p95_error": p95,
        "relative_rmse": rel,
        "baseline_rmse": baseline,
        "skill": skill_score(rmse, baseline),
        "n_compared": int(np.sum(finite)),
    }


def field_confidence(errors: np.ndarray, scale: float) -> np.ndarray:
    """Convert absolute errors into a per-voxel confidence in [0, 1]."""
    errors = np.asarray(errors, dtype=np.float64)
    if scale <= 0:
        return np.zeros_like(errors, dtype=np.float32)
    conf = 1.0 - np.abs(errors) / scale
    return np.clip(conf, 0.0, 1.0).astype(np.float32)


def schedule_metrics(
    pred_start: np.ndarray,
    pred_end: np.ndarray,
    pred_local: np.ndarray,
    ref_start: np.ndarray,
    ref_end: np.ndarray,
    ref_local: np.ndarray,
) -> Dict[str, Any]:
    """Compare full schedule fields and report per-field metrics."""
    return {
        "start_time": compute_field_metrics(pred_start, ref_start),
        "end_time": compute_field_metrics(pred_end, ref_end),
        "local_time": compute_field_metrics(pred_local, ref_local),
    }


def schedule_evaluation_mask(channels: np.ndarray, ref_end: np.ndarray) -> np.ndarray:
    """Cells that count toward accuracy: active, not chill, reference end > 0.

    Args:
        channels: ``(C, ...)`` encoded grid; channel 0 is occupancy, channel 1
            the part-type code divided by ``PART_TYPE_DIVISOR``.
        ref_end: Reference end-time field with the same cell count as one
            channel (any shape).

    Returns:
        Flat boolean mask of length ``R^3``.
    """
    c = np.asarray(channels)
    c = c.reshape(c.shape[0], -1)
    end = np.asarray(ref_end, dtype=np.float64).reshape(-1)
    if end.shape[0] != c.shape[1]:
        raise ValueError(f"ref_end has {end.shape[0]} cells, channels have {c.shape[1]}")
    active = c[0] > 0.5
    chill = np.abs(c[1] - PART_TYPE_CHILL / PART_TYPE_DIVISOR) < CHILL_CHANNEL_TOL
    return active & ~chill & np.isfinite(end) & (end > 0.0)


def masked_schedule_metrics(
    pred: np.ndarray,
    ref_start: np.ndarray,
    ref_end: np.ndarray,
    ref_local: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, Any]:
    """``schedule_metrics`` restricted to ``mask`` cells.

    Args:
        pred: ``(3, ...)`` prediction ordered start, end, local.
        ref_start, ref_end, ref_local: Reference fields, any shape with the
            same cell count as one prediction field.
        mask: Boolean mask with that same cell count.

    Returns:
        Same per-field dict as :func:`schedule_metrics`.  When ``mask`` selects
        no cells every metric is NaN and ``n_compared`` is 0.
    """
    pred = np.asarray(pred)
    if pred.ndim < 2 or pred.shape[0] != 3:
        raise ValueError(f"pred must be (3, ...), got {pred.shape}")
    flat_pred = pred.reshape(3, -1)
    m = np.asarray(mask, dtype=bool).reshape(-1)
    if m.shape[0] != flat_pred.shape[1]:
        raise ValueError(f"mask has {m.shape[0]} cells, pred has {flat_pred.shape[1]}")
    if not np.any(m):
        return {
            "start_time": dict(_NAN_FIELD),
            "end_time": dict(_NAN_FIELD),
            "local_time": dict(_NAN_FIELD),
        }
    refs = [np.asarray(r, dtype=np.float64).reshape(-1) for r in (ref_start, ref_end, ref_local)]
    for r in refs:
        if r.shape[0] != m.shape[0]:
            raise ValueError(f"reference has {r.shape[0]} cells, mask has {m.shape[0]}")
    return schedule_metrics(
        flat_pred[0][m], flat_pred[1][m], flat_pred[2][m],
        refs[0][m], refs[1][m], refs[2][m],
    )


def _nan_max(values) -> float:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return max(vals) if vals else float("nan")


def aggregate_schedule_metrics(schedule_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Roll the per-field dicts into a single headline accuracy summary.

    ``rmse_s`` / ``mae_s`` / ``max_error_s`` are the worst field.  The
    end-time field carries the headline comparison against a constant-mean
    predictor: ``baseline_rmse_s`` and ``skill`` (``1 - rmse / baseline``,
    clipped to [0, 1]).  ``confidence`` is that skill score; a model that
    does no better than predicting the mean reports 0.
    """
    fields = ["end_time", "start_time", "local_time"]
    end = schedule_metrics.get("end_time", {}) or {}
    rmse_s = _nan_max(schedule_metrics.get(f, {}).get("rmse") for f in fields)
    mae_s = _nan_max(schedule_metrics.get(f, {}).get("mae") for f in fields)
    max_err = _nan_max(schedule_metrics.get(f, {}).get("max_error") for f in fields)
    rel = end.get("relative_rmse")
    end_rmse = end.get("rmse")
    end_mae = end.get("mae")
    baseline = end.get("baseline_rmse")
    end_rmse_f = float(end_rmse) if end_rmse is not None else float("nan")
    baseline_f = float(baseline) if baseline is not None else float("nan")
    skill = skill_score(end_rmse_f, baseline_f)
    return {
        "rmse_s": float(rmse_s),
        "mae_s": float(mae_s),
        "max_error_s": float(max_err),
        "end_rmse_s": end_rmse_f,
        "end_mae_s": float(end_mae) if end_mae is not None else float("nan"),
        "baseline_rmse_s": baseline_f,
        "skill": float(round(skill, 3)) if np.isfinite(skill) else float("nan"),
        "relative_rmse_end": float(rel) if rel is not None else float("nan"),
        "confidence": float(round(skill, 3)) if np.isfinite(skill) else float("nan"),
        "n_compared": int(end.get("n_compared") or 0),
    }


def format_metrics(metrics: Dict[str, Any]) -> str:
    """Render a headline accuracy summary for the UI."""
    rmse = metrics.get("rmse_s", 0.0)
    mae = metrics.get("mae_s", 0.0)
    conf = metrics.get("confidence", 0.0)
    baseline = metrics.get("baseline_rmse_s")
    if rmse is None or math_isnan(rmse):
        return "accuracy unknown"
    text = f"RMSE {rmse:.1f}s · MAE {0.0 if mae is None else mae:.1f}s"
    if baseline is not None and not math_isnan(baseline):
        text += f" · baseline {baseline:.1f}s"
    if conf is not None and not math_isnan(conf):
        text += f" · conf {conf:.2f}"
    return text


def math_isnan(value) -> bool:
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False
