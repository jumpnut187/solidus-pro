from __future__ import annotations

import json
import math
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, fields as dataclass_fields
from typing import Any, Dict, List, Optional, Union

import numpy as np

from core.solidification_params import DEFAULT_MAX_SIMULATION_TIME_S

from core.surrogate.backend import Backend
from core.surrogate.conditioning import Conditioning, raw_from_conditioning
from core.surrogate.fields import CHANNELS, PART_TYPE_DIVISOR

OUTPUTS: List[str] = [
    "solidification_start_times",
    "solidification_end_times",
    "local_solidification_time",
]

WCB_DIFFUSIVITY = 45 / (7200 * 600)
WCB_LATENT = 272000.0
_TIME_STEP_FLOOR = 0.1

def _material_factor(raw: Dict[str, float]) -> float:
    conductivity = max(float(raw.get("thermal_conductivity", 45.0)), 1e-9)
    density = max(float(raw.get("density_solid", 7500.0)), 1.0)
    specific_heat = max(float(raw.get("specific_heat", 600.0)), 1.0)
    diffusivity = conductivity / (density * specific_heat)
    latent = max(float(raw.get("latent_heat", 270000.0)), 1.0)
    return float(math.sqrt((WCB_DIFFUSIVITY / diffusivity) * (latent / WCB_LATENT)))

def _decode_chill_boost(channel: np.ndarray, log6: float) -> np.ndarray:
    return np.maximum(np.exp(np.asarray(channel, dtype=np.float64) * log6), 1.0)

def _mushy_fraction(raw: Dict[str, float]) -> float:
    liquidus = float(raw.get("liquidus_temp", 1450.0))
    solidus = float(raw.get("solidus_temp", 1400.0))
    pouring = float(raw.get("pouring_temp", 1600.0))
    superheat = max(pouring - liquidus, 1.0)
    freezing_range = max(liquidus - solidus, 1.0)
    return float(np.clip(freezing_range / (superheat + freezing_range), 0.15, 0.45))

@dataclass
class ModelManifest:

    name: str
    version: str
    kind: str
    backend: str
    grid_resolution: int = 64
    part_type_divisor: float = PART_TYPE_DIVISOR
    channels: List[str] = field(default_factory=lambda: list(CHANNELS))
    outputs: List[str] = field(default_factory=lambda: list(OUTPUTS))
    alloy_features: List[str] = field(default_factory=list)
    config_features: List[str] = field(default_factory=list)
    supported_analyses: List[str] = field(default_factory=list)
    alloy_coverage: List[str] = field(default_factory=list)
    accuracy: Dict[str, Any] = field(default_factory=dict)
    trained_on: str = ""
    units: str = "inches/seconds/celsius"
    notes: str = ""
    artifact_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelManifest":
        allowed = {f.name for f in dataclass_fields(cls)}
        clean = {k: v for k, v in data.items() if k in allowed}
        return cls(**clean)

@dataclass
class ModelLoadResult:

    model: Optional["SurrogateModel"] = None
    error: str = ""

class SurrogateModel(ABC):

    manifest: ModelManifest

    @property
    def backend(self) -> Backend:
        return Backend(self.manifest.backend)

    @abstractmethod
    def predict(self, channels: np.ndarray, conditioning: Conditioning) -> np.ndarray:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    def describe(self) -> str:
        m = self.manifest
        lines = [
            f"Model: {m.name} (v{m.version})",
            f"  Backend: {m.backend}",
            f"  Grid resolution: {m.grid_resolution}^3",
            f"  Outputs: {', '.join(m.outputs)}",
            f"  Alloy coverage: {', '.join(m.alloy_coverage) or 'any'}",
            f"  Supported analyses: {', '.join(m.supported_analyses) or 'all schedule-derived'}",
        ]
        if m.accuracy:
            acc = m.accuracy
            lines.append(
                f"  Accuracy: RMSE={_fmt_num(acc.get('rmse_s'), 1)}s "
                f"MAE={_fmt_num(acc.get('mae_s'), 1)}s "
                f"baseline={_fmt_num(acc.get('baseline_rmse_s'), 1)}s "
                f"conf={_fmt_num(acc.get('confidence'), 2)}"
            )
        if m.notes:
            lines.append(f"  Notes: {m.notes}")
        return "\n".join(lines)

def _fmt_num(value: Any, digits: int) -> str:
    if value is None:
        return "n/a"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(f):
        return "n/a"
    return f"{f:.{digits}f}"

class NumpyDemoModel(SurrogateModel):

    def __init__(self, manifest: Optional[ModelManifest] = None, resolution: int = 64):
        self.manifest = manifest or build_demo_manifest(resolution=resolution)

    def predict(self, channels: np.ndarray, conditioning: Conditioning) -> np.ndarray:
        from core.surrogate.fields import _LOG6

        if channels.ndim != 4:
            raise ValueError(f"Expected 4D channels, got shape {channels.shape}")
        raw = raw_from_conditioning(conditioning)
        mold_constant = float(raw.get("chvorinov_mold_constant", 280.0))
        material_factor = _material_factor(raw)
        mushy = _mushy_fraction(raw)
        bias = float(raw.get("feeding_zone_vertical_bias", 0.0)) + 0.3

        c = channels.reshape(channels.shape[0], -1)
        active = c[0] > 0.5
        mod_in = c[3] * float(raw.get("max_dim", 10.0))
        surf_in = c[2] * float(raw.get("max_dim", 10.0))
        chill_in = _decode_chill_boost(c[5], _LOG6)
        riser_in = np.maximum(c[6], 0.15)
        height = c[4]

        completion = mold_constant * material_factor * (mod_in ** 2)
        completion = completion / chill_in / riser_in
        completion = completion * (1.0 + bias * height)
        completion = completion * (1.0 + 0.15 * surf_in / max(float(raw.get("max_dim", 10.0)), 1.0))

        end = completion.copy()
        end[~active] = 0.0
        end = np.maximum(end, _TIME_STEP_FLOOR)
        local = end * mushy
        local[~active] = 0.0
        start = end - local
        start[~active] = 0.0
        start = np.maximum(start, 0.0)

        res = channels.shape[1]
        out = np.stack([start, end, local]).reshape(3, res, res, res)
        return out.astype(np.float32)

    def close(self) -> None:
        pass

class NumpyWeightsModel(SurrogateModel):

    FEATURE_NAMES: List[str] = [
        "intercept",
        "log1p(modulus_in)",
        "log1p(surface_distance_in)",
        "log1p(chill_boost)",
        "log1p(riser_slowdown)",
        "log1p(height)",
        "log1p(1 - height)",
        "log1p(max_dim)",
        "log1p(mold_constant)",
        "log1p(material_factor)",
        "log1p(1 + vertical_bias)",
        "log1p(mushy_fraction)",
    ]

    def __init__(
        self,
        manifest: ModelManifest,
        weights: np.ndarray,
        local_weights: Optional[np.ndarray] = None,
    ):
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape[0] != len(self.FEATURE_NAMES):
            raise ValueError(
                f"Weights length {weights.shape[0]} != features {len(self.FEATURE_NAMES)}"
            )
        if local_weights is not None:
            local_weights = np.asarray(local_weights, dtype=np.float64)
            if local_weights.shape != weights.shape:
                raise ValueError(
                    f"local_weights shape {local_weights.shape} != weights shape {weights.shape}"
                )
        self.manifest = manifest
        self.weights = weights
        self.local_weights = local_weights

    @property
    def has_local_head(self) -> bool:
        return self.local_weights is not None

    @staticmethod
    def _features(channels: np.ndarray, conditioning: Conditioning) -> tuple[np.ndarray, np.ndarray]:
        from core.surrogate.fields import _LOG6

        raw = raw_from_conditioning(conditioning)
        max_dim = float(raw.get("max_dim", 10.0))
        mold_constant = float(raw.get("chvorinov_mold_constant", 280.0))
        material_factor = _material_factor(raw)
        bias = float(raw.get("feeding_zone_vertical_bias", 0.0)) + 0.3
        mushy = _mushy_fraction(raw)

        c = channels.reshape(channels.shape[0], -1)
        active = c[0] > 0.5
        mod_in = c[3] * max_dim
        surf_in = c[2] * max_dim
        chill_in = _decode_chill_boost(c[5], _LOG6)
        riser_in = np.maximum(c[6], 1e-6)
        height = np.clip(c[4], 0.0, 1.0)

        eps = 1e-9
        ones = np.ones_like(mod_in)
        X = np.stack(
            [
                ones,
                np.log1p(np.maximum(mod_in, eps)),
                np.log1p(np.maximum(surf_in, eps)),
                np.log1p(chill_in),
                np.log1p(riser_in),
                np.log1p(np.maximum(height, eps)),
                np.log1p(np.maximum(1.0 - height, eps)),
                ones * np.log1p(max_dim),
                ones * np.log1p(mold_constant),
                ones * np.log1p(material_factor),
                ones * np.log1p(1.0 + bias),
                ones * np.log1p(mushy),
            ],
            axis=0,
        )
        return X, active

    @staticmethod
    def _fit_log_target(
        channels_list: List[np.ndarray],
        conditioning_list: List[Conditioning],
        target_list: List[np.ndarray],
        ridge: float = 1e-3,
    ) -> np.ndarray:
        X_rows: List[np.ndarray] = []
        y_rows: List[np.ndarray] = []
        for channels, conditioning, target in zip(channels_list, conditioning_list, target_list):
            X, active = NumpyWeightsModel._features(channels, conditioning)
            target_flat = np.asarray(target, dtype=np.float64).reshape(-1)
            keep = active & (target_flat > 0) & np.isfinite(target_flat)
            X_rows.append(X[:, keep])
            y_rows.append(np.log(np.maximum(target_flat[keep], 1e-6)))
        Xm = np.concatenate(X_rows, axis=1)
        ym = np.concatenate(y_rows)
        if ym.size == 0:
            raise ValueError("No active cells with a positive target to fit")
        n_features = Xm.shape[0]
        if ridge > 0:
            Xm = np.concatenate([Xm, np.eye(n_features) * math.sqrt(ridge)], axis=1)
            ym = np.concatenate([ym, np.zeros(n_features)])
        weights, *_ = np.linalg.lstsq(Xm.T, ym, rcond=None)
        return weights

    @staticmethod
    def _fit_weights(
        channels_list: List[np.ndarray],
        conditioning_list: List[Conditioning],
        end_list: List[np.ndarray],
        ridge: float = 1e-3,
    ) -> np.ndarray:
        return NumpyWeightsModel._fit_log_target(channels_list, conditioning_list, end_list, ridge)

    @staticmethod
    def _fit_local_weights(
        channels_list: List[np.ndarray],
        conditioning_list: List[Conditioning],
        local_list: List[np.ndarray],
        ridge: float = 1e-3,
    ) -> np.ndarray:
        return NumpyWeightsModel._fit_log_target(channels_list, conditioning_list, local_list, ridge)

    def predict(self, channels: np.ndarray, conditioning: Conditioning) -> np.ndarray:
        raw = raw_from_conditioning(conditioning)
        max_time_cap = float(raw.get("max_simulation_time", DEFAULT_MAX_SIMULATION_TIME_S))
        X, active = self._features(channels, conditioning)
        log_end = self.weights @ X
        end = np.exp(log_end)
        end = np.minimum(end, max_time_cap)
        end = np.maximum(end, _TIME_STEP_FLOOR)
        end[~active] = 0.0
        if self.local_weights is None:
            local = end * _mushy_fraction(raw)
        else:
            local = np.exp(self.local_weights @ X)
            local = np.minimum(local, end)
        local = np.maximum(local, 0.0)
        local[~active] = 0.0
        start = end - local
        start[~active] = 0.0
        start = np.maximum(start, 0.0)

        res = channels.shape[1]
        out = np.stack([start, end, local]).reshape(3, res, res, res)
        return out.astype(np.float32)

    def close(self) -> None:
        pass

    def describe(self) -> str:
        text = super().describe()
        if self.local_weights is None:
            text += (
                "\n  Local-time head: none (weights.npz predates local_weights); "
                "using local = end * mushy_fraction"
            )
        else:
            text += "\n  Local-time head: learned log(local_time) ridge fit"
        return text

def build_demo_manifest(resolution: int = 64, name: str = "Reference Chvorinov Model") -> ModelManifest:
    from core.surrogate.conditioning import ALLOY_FEATURES, CONFIG_FEATURES

    return ModelManifest(
        name=name,
        version="1.0.0",
        kind="numpy_demo",
        backend=Backend.NUMPY.value,
        grid_resolution=int(resolution),
        alloy_features=list(ALLOY_FEATURES),
        config_features=list(CONFIG_FEATURES),
        supported_analyses=[
            "shrinkage",
            "porosity",
            "hotspot",
            "niyama",
            "sand_burn_in",
            "gas_porosity",
            "fraction_liquid",
            "chvorinov",
        ],
        alloy_coverage=[],
        notes="Deterministic physics-derived reference; no learned weights.",
    )

def _save_manifest(manifest: ModelManifest, directory: str) -> str:
    path = os.path.join(directory, "manifest.json")
    with open(path, "w") as fh:
        json.dump(manifest.to_dict(), fh, indent=2)
    return path

def save_numpy_demo(directory: str, resolution: int = 64) -> str:
    os.makedirs(directory, exist_ok=True)
    manifest = build_demo_manifest(resolution=resolution)
    _save_manifest(manifest, directory)
    return directory

def _read_manifest(directory: str) -> Optional[ModelManifest]:
    path = os.path.join(directory, "manifest.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            data = json.load(fh)
        return ModelManifest.from_dict(data)
    except Exception:
        return None

def load_model(path: Union[str, os.PathLike]) -> ModelLoadResult:
    path = os.fspath(path)
    if os.path.isfile(path) and path.endswith(".json"):
        directory = os.path.dirname(path) or "."
        manifest = _read_manifest(directory)
    elif os.path.isdir(path):
        directory = path
        manifest = _read_manifest(directory)
    else:
        return ModelLoadResult(error=f"Not a model directory or manifest: {path}")

    if manifest is None:
        return ModelLoadResult(error=f"No manifest.json found at {path}")

    if not math.isclose(manifest.part_type_divisor, PART_TYPE_DIVISOR, rel_tol=1e-9):
        return ModelLoadResult(
            error=(
                f"Model {manifest.name!r} was trained with "
                f"part_type_divisor={manifest.part_type_divisor}, but the current "
                f"encoding uses {PART_TYPE_DIVISOR}. Retrain the model against the "
                "current part-type encoding, or restore the previous divisor."
            )
        )

    if manifest.kind == "numpy_demo":
        return ModelLoadResult(model=NumpyDemoModel(manifest))
    if manifest.kind == "numpy_weights":
        weights_path = os.path.join(directory, "weights.npz")
        if not os.path.exists(weights_path):
            return ModelLoadResult(error=f"Missing weights.npz for {manifest.name}")
        try:
            with np.load(weights_path) as npz:
                weights = npz["weights"]
                local_weights = npz["local_weights"] if "local_weights" in npz.files else None
            model = NumpyWeightsModel(manifest, weights, local_weights)
            model.manifest.artifact_path = weights_path
            return ModelLoadResult(model=model)
        except Exception as exc:
            return ModelLoadResult(error=f"Failed to load weights.npz: {exc}")
    if manifest.kind == "torch_unet":
        from core.deep_model.inference import load_deep_model

        return load_deep_model(directory)
    if manifest.kind in ("coreml", "mlx"):
        return ModelLoadResult(
            error=(
                f"Model kind '{manifest.kind}' is not available yet. "
                "The Core ML / MLX artifact path is superseded by the deep "
                "learning module (core/deep_model) — load a torch_unet model "
                "instead."
            )
        )
    return ModelLoadResult(error=f"Unknown model kind: {manifest.kind}")
