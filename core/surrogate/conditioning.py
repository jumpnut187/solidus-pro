"""Alloy + configuration conditioning vectors for surrogate models.

Surrogate models consume two kinds of input:

1. Per-voxel channel fields (see ``fields.py``) sampled on a fixed grid.
2. A single global conditioning vector describing the alloy and the
   simulation configuration.

The conditioning vector is normalized to [0, 1] using deterministic
reference ranges so a trained model is comparable across runs.  The
inverse-normalization is exact (raw = v * (hi - lo) + lo), which lets a
model reconstruct physical values (seconds, inches, °C) from the vector
alone — the analytic reference model relies on this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from core.materials import AlloyProperties


ALLOY_FEATURE_RANGES: Dict[str, tuple] = {
    "liquidus_temp": (1000.0, 1600.0),
    "solidus_temp": (900.0, 1550.0),
    "pouring_temp": (1100.0, 1800.0),
    "density_liquid": (6000.0, 9000.0),
    "density_solid": (6500.0, 9500.0),
    "thermal_conductivity": (10.0, 60.0),
    "specific_heat": (300.0, 900.0),
    "latent_heat": (150000.0, 400000.0),
    "critical_fraction_solid": (0.3, 0.8),
    "linear_shrinkage": (1.0, 7.0),
    "niyama_threshold": (0.5, 1.5),
    "hot_tear_susceptibility": (0.05, 0.9),
    "partition_coefficient": (0.1, 0.5),
    "melting_point_solvent": (900.0, 1700.0),
    "eutectic_fraction": (0.0, 1.0),
}

CONFIG_FEATURE_RANGES: Dict[str, tuple] = {
    "max_dim": (1.0, 60.0),
    # Mesh length scale, inches: the voxel pitch on a voxel run, the tet edge
    # length (``TetVolumeMesh.base_voxel_size``) on a tet run. The mesh type
    # itself is record metadata (``mesh_type``), not a feature, so the vector
    # length and every trained model's input shape stay the same.
    "voxel_size": (0.02, 0.5),
    "feeding_zone_vertical_bias": (0.0, 1.0),
    "chvorinov_mold_constant": (150.0, 400.0),
    "safety_factor_no_risers": (1.0, 3.0),
    "safety_factor_with_risers": (1.0, 3.0),
    "min_simulation_time": (60.0, 600.0),
    "max_simulation_time": (3600.0, 21600.0),
}

ALLOY_FEATURES: List[str] = list(ALLOY_FEATURE_RANGES.keys())
CONFIG_FEATURES: List[str] = list(CONFIG_FEATURE_RANGES.keys())


def _normalize(value: float, lo: float, hi: float) -> float:
    """Map ``value`` into [0, 1] using the reference range."""
    if hi <= lo:
        return 0.5
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def denormalize(value: float, lo: float, hi: float) -> float:
    """Recover the raw value from a normalized [0, 1] feature."""
    return float(value * (hi - lo) + lo)


@dataclass
class Conditioning:
    """A conditioning vector plus its raw reconstruction."""

    vector: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    names: List[str] = field(default_factory=list)
    raw: Dict[str, float] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return int(self.vector.shape[0])

    def to_dict(self) -> Dict[str, Any]:
        return {"vector": self.vector.tolist(), "names": list(self.names), "raw": dict(self.raw)}


def _alloy_raw(alloy: AlloyProperties) -> Dict[str, float]:
    """Extract the raw (un-normalized) alloy feature dictionary."""
    melting_solvent = getattr(alloy, "melting_point_solvent", None)
    if melting_solvent is None:
        melting_solvent = float(getattr(alloy, "solidus_temp", 1400.0) or 1400.0)
    return {
        "liquidus_temp": float(getattr(alloy, "liquidus_temp", 1450.0) or 1450.0),
        "solidus_temp": float(getattr(alloy, "solidus_temp", 1400.0) or 1400.0),
        "pouring_temp": float(getattr(alloy, "pouring_temp", 1600.0) or 1600.0),
        "density_liquid": float(getattr(alloy, "density_liquid", 7000.0) or 7000.0),
        "density_solid": float(getattr(alloy, "density_solid", 7500.0) or 7500.0),
        "thermal_conductivity": float(getattr(alloy, "thermal_conductivity", 45.0) or 45.0),
        "specific_heat": float(getattr(alloy, "specific_heat", 600.0) or 600.0),
        "latent_heat": float(getattr(alloy, "latent_heat", 270000.0) or 270000.0),
        "critical_fraction_solid": float(getattr(alloy, "critical_fraction_solid", 0.5) or 0.5),
        "linear_shrinkage": float(getattr(alloy, "linear_shrinkage", 2.0) or 2.0),
        "niyama_threshold": float(getattr(alloy, "niyama_threshold", 1.0) or 1.0),
        "hot_tear_susceptibility": float(getattr(alloy, "hot_tear_susceptibility", 0.3) or 0.3),
        "partition_coefficient": float(getattr(alloy, "partition_coefficient", 0.17) or 0.17),
        "melting_point_solvent": melting_solvent,
        "eutectic_fraction": float(getattr(alloy, "eutectic_fraction", 0.0) or 0.0),
    }


def build_alloy_conditioning(alloy: AlloyProperties) -> Conditioning:
    """Build a normalized conditioning vector from an alloy."""
    raw = _alloy_raw(alloy)
    values = [_normalize(raw[name], *ALLOY_FEATURE_RANGES[name]) for name in ALLOY_FEATURES]
    return Conditioning(
        vector=np.asarray(values, dtype=np.float32),
        names=list(ALLOY_FEATURES),
        raw=raw,
    )


def build_config_conditioning(config: Dict[str, float]) -> Conditioning:
    """Build a normalized conditioning vector from configuration scalars.

    Missing keys fall back to the midpoint of their reference range so a model
    still sees a well-formed vector when callers omit optional config.
    """
    raw = {}
    for name in CONFIG_FEATURES:
        lo, hi = CONFIG_FEATURE_RANGES[name]
        value = config.get(name)
        if value is None:
            value = (lo + hi) / 2.0
        raw[name] = float(value)
    values = [_normalize(raw[name], *CONFIG_FEATURE_RANGES[name]) for name in CONFIG_FEATURES]
    return Conditioning(
        vector=np.asarray(values, dtype=np.float32),
        names=list(CONFIG_FEATURES),
        raw=raw,
    )


def build_conditioning_vector(alloy: AlloyProperties, config: Dict[str, float]) -> Conditioning:
    """Combine alloy + config conditioning into a single vector."""
    alloy_c = build_alloy_conditioning(alloy)
    config_c = build_config_conditioning(config)
    names = list(alloy_c.names) + list(config_c.names)
    raw = dict(alloy_c.raw)
    raw.update(config_c.raw)
    return Conditioning(
        vector=np.concatenate([alloy_c.vector, config_c.vector]).astype(np.float32),
        names=names,
        raw=raw,
    )


def raw_from_conditioning(conditioning: Conditioning) -> Dict[str, float]:
    """Reconstruct raw feature values from a conditioning vector (exact).

    Falls back to the stored ``raw`` dict when present, otherwise performs
    the exact inverse normalization.
    """
    if conditioning.raw:
        return dict(conditioning.raw)
    raw: Dict[str, float] = {}
    ranges = dict(ALLOY_FEATURE_RANGES)
    ranges.update(CONFIG_FEATURE_RANGES)
    for name, value in zip(conditioning.names, np.asarray(conditioning.vector)):
        lo, hi = ranges[name]
        raw[name] = denormalize(float(value), lo, hi)
    return raw