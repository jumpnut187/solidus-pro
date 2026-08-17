"""Accuracy metrics for surrogate predictions vs. the full solver."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def compute_field_metrics(pred: np.ndarray, ref: np.ndarray) -> Dict[str, Any]:
    """Compare two per-voxel fields and summarize the error distribution.

    Args:
        pred: Predicted field (float array).
        ref: Reference field (float array).

    Returns:
        Dict with ``mae``, ``rmse``, ``max_error``, ``p95_error`` and
        ``relative_rmse`` (relative to the reference RMS).
    """
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if pred.shape != ref.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs ref {ref.shape}")

    finite = np.isfinite(pred) & np.isfinite(ref)
    if not np.any(finite):
        return {"mae": float("nan"), "rmse": float("nan"), "max_error": float("nan"),
                "p95_error": float("nan"), "relative_rmse": float("nan")}

    pred_f, ref_f = pred[finite], ref[finite]
    error = np.abs(pred_f - ref_f)
    mae = float(np.mean(error))
    rmse = float(np.sqrt(np.mean((pred_f - ref_f) ** 2)))
    max_error = float(np.max(error))
    p95 = float(np.percentile(error, 95))
    ref_rms = float(np.sqrt(np.mean(ref_f ** 2)))
    rel = rmse / ref_rms if ref_rms > 1e-12 else float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "max_error": max_error,
        "p95_error": p95,
        "relative_rmse": rel,
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


def aggregate_schedule_metrics(schedule_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Roll the per-field dicts into a single headline accuracy summary."""
    fields = ["end_time", "start_time", "local_time"]
    rmse_s = max((schedule_metrics.get(f, {}).get("rmse") or 0.0) for f in fields)
    mae_s = max((schedule_metrics.get(f, {}).get("mae") or 0.0) for f in fields)
    max_err = max((schedule_metrics.get(f, {}).get("max_error") or 0.0) for f in fields)
    rel = schedule_metrics.get("end_time", {}).get("relative_rmse")
    confidence = max(0.0, 1.0 - float(rel or 1.0))
    return {
        "rmse_s": float(rmse_s),
        "mae_s": float(mae_s),
        "max_error_s": float(max_err),
        "relative_rmse_end": float(rel) if rel is not None else float("nan"),
        "confidence": float(round(confidence, 3)),
    }


def format_metrics(metrics: Dict[str, Any]) -> str:
    """Render a headline accuracy summary for the UI."""
    rmse = metrics.get("rmse_s", 0.0)
    mae = metrics.get("mae_s", 0.0)
    conf = metrics.get("confidence", 0.0)
    if isinstance(rmse, float) and math_isnan(rmse):
        return "accuracy unknown"
    return f"RMSE {rmse:.1f}s · MAE {mae:.1f}s · conf {conf:.2f}"


def math_isnan(value) -> bool:
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False