from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json
import os
import tempfile
import threading

import numpy as np
from scipy import ndimage

from core.part_manager import CODE_TO_PART_TYPE, PART_TYPE_COLORS, PART_TYPE_CAST

SCALAR_MODE_SOURCES: Dict[str, Tuple[str, ...]] = {
    "shrinkage": ("shrinkage",),
    "porosity": ("porosity",),
    "feed_modulus": ("feed_modulus",),
    "hotspot": ("hotspot",),
    "solidification_time": ("solidification_times",),
    "niyama_raw": ("niyama_raw",),
    "niyama": ("niyama_risk", "niyama"),
    "combined_feed_risk": ("combined_feed_risk",),
    "hot_tear": ("hot_tear",),
    "sand_burn_in": ("sand_burn_in",),
    "gas_generation": ("gas_generation",),
    "gas_entrapment": ("gas_entrapment",),
    "gas_porosity": ("gas_porosity_risk", "gas_porosity"),
    "feeding_quality": ("feeding_quality",),
    "thermal_stress": ("thermal_stress",),
    "distortion": ("distortion",),
    "crack_susceptibility": ("crack_susceptibility",),
    "sdas": ("sdas",),
    "predicted_uts": ("predicted_uts",),
    "fill_time": ("fill_time",),
}

MODE_UNITS: Dict[str, str] = {
    "feed_modulus": "in",
    "solidification_time": "s",
    "fill_time": "s",
    "thermal_stress": "MPa",
    "distortion": "in",
    "sdas": "µm",
    "predicted_uts": "MPa",
    "feeding_path_lengths": "in",
}

TIME_MODES = ("fraction_liquid", "fill_animation")

_NON_NEGATIVE_ONLY_MODES = frozenset({"fill_time"})

_QUANT_MAX = 255.0

from core.dense_grid import (
    MAX_LATTICE_CELLS,
    Lattice,
    LatticeTooLargeError,
    derive_lattice,
    flat_indices as _flat_indices,
    nearest_known_map as _nearest_known_map,
    scatter_and_fill as _scatter_and_fill,
)

class RemovedTetResultError(RuntimeError):
    pass

def _reject_removed_tet_result(marker: Dict[str, Any]) -> None:
    if marker.get("mesh_type") == "tet":
        raise RemovedTetResultError(
            "This result was produced by the removed tetrahedral mesher; re-run the simulation."
        )

@dataclass
class ScalarGrid:

    values: np.ndarray
    minimum: float
    maximum: float
    mean: float
    constant: bool = False
    empty: bool = False

    def dequantize(self) -> np.ndarray:
        span = self.maximum - self.minimum
        if span <= 0.0:
            return np.full(self.values.shape, self.minimum, dtype=np.float32)
        return (self.minimum + (self.values.astype(np.float32) / _QUANT_MAX) * span).astype(np.float32)

@dataclass
class FrameStack:

    frames: np.ndarray
    times_s: np.ndarray
    minimum: float = 0.0
    maximum: float = 1.0

    @property
    def count(self) -> int:
        return int(self.frames.shape[0])

@dataclass
class FieldGrids:

    lattice: Lattice
    mask: np.ndarray
    scalars: Dict[str, ScalarGrid] = dataclass_field(default_factory=dict)
    stacks: Dict[str, FrameStack] = dataclass_field(default_factory=dict)
    voxel_count: int = 0
    cast_voxel_count: int = 0
    total_volume_in3: float = 0.0
    stats: Dict[str, Any] = dataclass_field(default_factory=dict)
    part_type_counts: Dict[int, int] = dataclass_field(default_factory=dict)
    downsampled: bool = False

def _quantize(dense: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    span = maximum - minimum
    if span <= 0.0:
        return np.zeros(dense.shape, dtype=np.uint8)
    scaled = (dense.astype(np.float64) - minimum) / span
    return np.clip(np.rint(scaled * _QUANT_MAX), 0.0, _QUANT_MAX).astype(np.uint8)

def build_field_grids(
    visualization: Dict[str, Any],
    summary: Optional[Dict[str, Any]] = None,
    *,
    downsampled: Optional[bool] = None,
) -> FieldGrids:
    summary = summary or {}
    _reject_removed_tet_result(visualization)
    centers = _as_array(visualization.get("centers"))
    if centers.size == 0:
        empty_lattice = Lattice(dims=(1, 1, 1), origin=(0.0, 0.0, 0.0), spacing=1.0)
        return FieldGrids(lattice=empty_lattice, mask=np.zeros(1, dtype=np.uint8))
    centers = centers.reshape((-1, 3))
    n_voxels = int(centers.size // 3)

    sizes = _as_array(visualization.get("sizes"))
    lattice = derive_lattice(centers, sizes if sizes.size == n_voxels else None)
    cell_count = lattice.cell_count
    cell_indices = _flat_indices(centers, lattice)

    raw_part_types = visualization.get("part_types")
    if raw_part_types is not None and len(raw_part_types) == n_voxels:
        part_types = np.asarray(raw_part_types, dtype=np.int32)
    else:
        part_types = np.full(n_voxels, PART_TYPE_CAST, dtype=np.int32)

    mask = np.zeros(cell_count, dtype=np.uint8)
    mask[cell_indices] = np.clip(part_types + 1, 0, 255).astype(np.uint8)

    nearest = _nearest_known_map(mask != 0, lattice)

    reported = summary.get("voxel_count")
    strided = downsampled
    if strided is None:
        strided = bool(isinstance(reported, (int, float)) and int(reported) > n_voxels)

    grids = FieldGrids(
        lattice=lattice,
        mask=mask,
        downsampled=bool(strided),
        voxel_count=n_voxels,
        cast_voxel_count=int(np.sum(part_types == PART_TYPE_CAST)),
        total_volume_in3=float(summary.get("total_volume", 0.0) or 0.0),
        stats=_flatten_stats(summary),
    )
    codes, counts = np.unique(part_types, return_counts=True)
    grids.part_type_counts = {int(code): int(count) for code, count in zip(codes, counts)}

    for mode, source_keys in SCALAR_MODE_SOURCES.items():
        values = _first_present(visualization, source_keys, n_voxels)
        if values is None:
            continue
        sample_mask = np.isfinite(values)
        if mode in _NON_NEGATIVE_ONLY_MODES:
            sample_mask &= values >= 0.0
        if not np.any(sample_mask):
            continue
        usable = values[sample_mask]
        minimum = float(np.min(usable))
        maximum = float(np.max(usable))
        mean = float(np.mean(usable))
        dense = _scatter_and_fill(
            np.where(sample_mask, values, 0.0).astype(np.float32),
            cell_indices,
            sample_mask,
            nearest,
            lattice,
        )
        constant = bool(maximum <= minimum)
        grids.scalars[mode] = ScalarGrid(
            values=_quantize(dense, minimum, maximum),
            minimum=minimum,
            maximum=maximum,
            mean=mean,
            constant=constant,
            empty=bool(constant and abs(maximum) < 1e-12),
        )

    fraction_liquid = _build_frame_stack(
        visualization.get("fraction_liquid_history"),
        visualization.get("fraction_liquid_times"),
        cell_indices,
        nearest,
        lattice,
        n_voxels,
    )
    if fraction_liquid is not None:
        grids.stacks["fraction_liquid"] = fraction_liquid

    fill = _build_frame_stack(
        visualization.get("fill_animation_history"),
        visualization.get("fill_animation_times"),
        cell_indices,
        nearest,
        lattice,
        n_voxels,
    )
    if fill is not None:
        grids.stacks["fill_animation"] = fill

    return grids

def _as_array(value: Any) -> np.ndarray:
    if value is None:
        return np.empty(0, dtype=np.float64)
    return np.asarray(value, dtype=np.float64)

def _first_present(
    visualization: Dict[str, Any], keys: Iterable[str], expected: int
) -> Optional[np.ndarray]:
    for key in keys:
        raw = visualization.get(key)
        if raw is None:
            continue
        arr = np.asarray(raw, dtype=np.float64).reshape(-1)
        if arr.size == expected:
            return arr
    return None

def _build_frame_stack(
    history: Any,
    times: Any,
    cell_indices: np.ndarray,
    nearest: np.ndarray,
    lattice: Lattice,
    expected: int,
) -> Optional[FrameStack]:
    if not isinstance(history, list) or not history:
        return None
    frames: List[np.ndarray] = []
    for snapshot in history:
        arr = np.asarray(snapshot, dtype=np.float64).reshape(-1)
        if arr.size != expected:
            return None
        finite = np.isfinite(arr)
        dense = _scatter_and_fill(
            np.where(finite, np.clip(arr, 0.0, 1.0), 0.0).astype(np.float32),
            cell_indices,
            finite,
            nearest,
            lattice,
        )
        frames.append(_quantize(dense, 0.0, 1.0))

    time_arr = np.asarray(times if times is not None else [], dtype=np.float32).reshape(-1)
    if time_arr.size != len(frames):
        time_arr = np.linspace(0.0, 1.0, len(frames), dtype=np.float32).reshape(-1)
    return FrameStack(frames=np.stack(frames, axis=0), times_s=time_arr)

def _flatten_stats(summary: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in summary.items():
        if key in {"raw_ranges", "available_modes"}:
            continue
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (int, float, str, bool)):
                    flat[f"{key.removesuffix('_stats')}_{sub_key}"] = _scalar(sub_value)
        elif isinstance(value, (int, float, str, bool)):
            flat[key] = _scalar(value)
    return flat

def _scalar(value: Any) -> Any:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        number = float(value)
        return number if np.isfinite(number) else 0.0
    return str(value)

def build_grid_metadata(grids: FieldGrids) -> Dict[str, Any]:
    modes: List[str] = []
    empty_modes: List[str] = []
    constant_modes: List[str] = []
    ranges: Dict[str, Dict[str, float]] = {}
    units: Dict[str, str] = {}
    for mode, grid in grids.scalars.items():
        ranges[mode] = {"min": grid.minimum, "max": grid.maximum, "mean": grid.mean}
        if mode in MODE_UNITS:
            units[mode] = MODE_UNITS[mode]
        if grid.constant:
            constant_modes.append(mode)
        if grid.empty:
            empty_modes.append(mode)
        else:
            modes.append(mode)

    frames: Dict[str, Dict[str, Any]] = {}
    if "fraction_liquid" in grids.stacks:
        stack = grids.stacks["fraction_liquid"]
        frames["fraction_liquid"] = {"count": stack.count, "times_s": [float(t) for t in stack.times_s]}
        modes.append("fraction_liquid")
    if "fill_animation" in grids.stacks:
        stack = grids.stacks["fill_animation"]
        frames["fill"] = {"count": stack.count, "times_s": [float(t) for t in stack.times_s]}
        modes.append("fill_animation")

    nx, ny, nz = grids.lattice.dims
    return {
        "dims": [int(nx), int(ny), int(nz)],
        "origin": [float(v) for v in grids.lattice.origin],
        "spacing": float(grids.lattice.spacing),
        "modes": sorted(set(modes)),
        "emptyModes": sorted(set(empty_modes)),
        "constantModes": sorted(set(constant_modes)),
        "ranges": ranges,
        "units": units,
        "frames": frames,
        "voxelCount": int(grids.voxel_count),
        "castVoxelCount": int(grids.cast_voxel_count),
        "cellCount": int(grids.lattice.cell_count),
        "totalVolumeIn3": float(grids.total_volume_in3),
        "downsampled": bool(grids.downsampled),
        "partTypes": _part_type_table(grids.part_type_counts),
        "stats": grids.stats,
        "quantization": {"dtype": "uint8", "levels": int(_QUANT_MAX) + 1},
    }

def _part_type_table(counts: Dict[int, int]) -> List[Dict[str, Any]]:
    table: List[Dict[str, Any]] = []
    for code in sorted(counts):
        part_type = CODE_TO_PART_TYPE.get(int(code))
        color = PART_TYPE_COLORS.get(part_type) if part_type is not None else None
        table.append(
            {
                "code": int(code) + 1,
                "partTypeCode": int(code),
                "id": part_type.value if part_type is not None else f"type_{int(code)}",
                "label": str(part_type) if part_type is not None else f"Type {int(code)}",
                "color": [float(c) for c in color] if color is not None else [0.7, 0.75, 0.8],
                "voxelCount": int(counts[code]),
            }
        )
    return table

SIDECAR_VERSION = 1

def sidecar_path(artifact_path: Path | str) -> Path:
    path = Path(artifact_path)
    return path.with_name(path.stem + ".grid.npz")

def save_sidecar(path: Path, grids: FieldGrids) -> None:
    meta: Dict[str, Any] = {
        "stats": grids.stats,
        "part_type_counts": {str(k): v for k, v in grids.part_type_counts.items()},
    }
    payload: Dict[str, np.ndarray] = {
        "version": np.asarray([SIDECAR_VERSION], dtype=np.int32),
        "dims": np.asarray(grids.lattice.dims, dtype=np.int32),
        "origin": np.asarray(grids.lattice.origin, dtype=np.float64),
        "spacing": np.asarray([grids.lattice.spacing], dtype=np.float64),
        "mask": grids.mask,
        "counts": np.asarray(
            [grids.voxel_count, grids.cast_voxel_count, 1 if grids.downsampled else 0],
            dtype=np.int64,
        ),
        "total_volume": np.asarray([grids.total_volume_in3], dtype=np.float64),
        "meta_json": np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
    }
    for mode, grid in grids.scalars.items():
        payload[f"scalar__{mode}"] = grid.values
        payload[f"scalar_range__{mode}"] = np.asarray(
            [
                grid.minimum,
                grid.maximum,
                grid.mean,
                1.0 if grid.empty else 0.0,
                1.0 if grid.constant else 0.0,
            ],
            dtype=np.float64,
        )
    for key, stack in grids.stacks.items():
        payload[f"stack__{key}"] = stack.frames
        payload[f"stack_times__{key}"] = stack.times_s

    path.parent.mkdir(parents=True, exist_ok=True)
    handle_fd, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle_fd, "wb") as handle:
            # numpy's stub types **kwds poorly; the arrays are the archive members.
            arrays: Dict[str, Any] = payload
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

def load_sidecar(path: Path) -> Optional[FieldGrids]:
    try:
        with np.load(path, allow_pickle=False) as data:
            if int(data["version"][0]) != SIDECAR_VERSION:
                return None
            dims = tuple(int(v) for v in data["dims"])
            lattice = Lattice(
                dims=(dims[0], dims[1], dims[2]),
                origin=(
                    float(data["origin"][0]),
                    float(data["origin"][1]),
                    float(data["origin"][2]),
                ),
                spacing=float(data["spacing"][0]),
            )
            counts = data["counts"]
            meta = json.loads(bytes(data["meta_json"]).decode("utf-8"))
            _reject_removed_tet_result(meta)
            grids = FieldGrids(
                lattice=lattice,
                mask=np.asarray(data["mask"], dtype=np.uint8),
                voxel_count=int(counts[0]),
                cast_voxel_count=int(counts[1]),
                downsampled=bool(counts[2]) if len(counts) > 2 else False,
                total_volume_in3=float(data["total_volume"][0]),
                stats=dict(meta.get("stats", {})),
                part_type_counts={int(k): int(v) for k, v in meta.get("part_type_counts", {}).items()},
            )
            for key in data.files:
                if key.startswith("scalar__"):
                    mode = key.split("__", 1)[1]
                    rng = data[f"scalar_range__{mode}"]
                    grids.scalars[mode] = ScalarGrid(
                        values=np.asarray(data[key], dtype=np.uint8),
                        minimum=float(rng[0]),
                        maximum=float(rng[1]),
                        mean=float(rng[2]),
                        empty=bool(rng[3] > 0.5),
                        constant=bool(rng[4] > 0.5)
                        if len(rng) > 4
                        else bool(float(rng[1]) <= float(rng[0])),
                    )
                elif key.startswith("stack__"):
                    stack_key = key.split("__", 1)[1]
                    grids.stacks[stack_key] = FrameStack(
                        frames=np.asarray(data[key], dtype=np.uint8),
                        times_s=np.asarray(data[f"stack_times__{stack_key}"], dtype=np.float32),
                    )
            return grids
    except RemovedTetResultError:
        raise
    except Exception:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return None

def write_sidecar_for_artifact(
    artifact_path: Path | str,
    artifact: Dict[str, Any],
    *,
    grid_source: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    try:
        if grid_source:
            grids = build_field_grids(grid_source, artifact.get("summary", {}), downsampled=False)
        else:
            grids = build_field_grids(artifact.get("visualization", {}), artifact.get("summary", {}))
        if grids.voxel_count == 0:
            return None
        path = sidecar_path(artifact_path)
        save_sidecar(path, grids)
        return path
    except Exception:
        return None

def grids_nbytes(grids: FieldGrids) -> int:
    total = int(grids.mask.nbytes)
    for grid in grids.scalars.values():
        total += int(grid.values.nbytes)
    for stack in grids.stacks.values():
        total += int(stack.frames.nbytes) + int(stack.times_s.nbytes)
    return total

class _GridCache:

    def __init__(self, max_bytes: int = 256 * 1024 * 1024) -> None:
        self._max_bytes = max_bytes
        self._entries: "OrderedDict[Tuple[str, int, int], Tuple[FieldGrids, int]]" = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    @property
    def nbytes(self) -> int:
        with self._lock:
            return self._bytes

    def get(self, key: Tuple[str, int, int]) -> Optional[FieldGrids]:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.move_to_end(key)
            return entry[0]

    def put(self, key: Tuple[str, int, int], grids: FieldGrids) -> None:
        size = grids_nbytes(grids)
        with self._lock:
            existing = self._entries.pop(key, None)
            if existing is not None:
                self._bytes -= existing[1]
            if size > self._max_bytes:
                return
            self._entries[key] = (grids, size)
            self._bytes += size
            while self._bytes > self._max_bytes and self._entries:
                _, (_, evicted) = self._entries.popitem(last=False)
                self._bytes -= evicted

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0

_CACHE = _GridCache()

def clear_cache() -> None:
    _CACHE.clear()

def load_grids(
    project_id: str,
    artifact_path: Path | str,
    *,
    artifact: Optional[Dict[str, Any]] = None,
) -> Optional[FieldGrids]:
    path = Path(artifact_path)
    try:
        stat = path.stat()
    except OSError:
        if artifact is None:
            return None
        stat = None

    key = (project_id, int(stat.st_mtime_ns) if stat else 0, int(stat.st_size) if stat else 0)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    sidecar = sidecar_path(path)
    grids: Optional[FieldGrids] = None
    if stat is not None and sidecar.exists() and sidecar.stat().st_mtime_ns >= stat.st_mtime_ns:
        grids = load_sidecar(sidecar)

    if grids is None:
        source = artifact
        if source is None:
            if not path.exists():
                return None
            try:
                with path.open("r", encoding="utf-8") as handle:
                    source = json.load(handle)
            except (json.JSONDecodeError, UnicodeDecodeError, OSError):
                return None
        grids = build_field_grids(source.get("visualization", {}), source.get("summary", {}))
        if grids.voxel_count == 0:
            return None
        try:
            save_sidecar(sidecar, grids)
        except OSError:
            pass

    _CACHE.put(key, grids)
    return grids

def mask_bytes(grids: FieldGrids) -> bytes:
    return np.ascontiguousarray(grids.mask, dtype=np.uint8).tobytes()

def scalar_bytes(grid: ScalarGrid, dtype: str) -> bytes:
    if dtype == "f32":
        return np.ascontiguousarray(grid.dequantize(), dtype="<f4").tobytes()
    return np.ascontiguousarray(grid.values, dtype=np.uint8).tobytes()

def frame_bytes(stack: FrameStack, start: int, count: int) -> Tuple[bytes, int, int]:
    total = stack.count
    begin = max(int(start), 0)
    if begin >= total:
        return b"", begin, 0
    length = int(np.clip(count, 0, total - begin))
    if length <= 0:
        return b"", begin, 0
    block = stack.frames[begin : begin + length]
    return np.ascontiguousarray(block, dtype=np.uint8).tobytes(), begin, length

@dataclass
class Envelope:

    positions: np.ndarray
    normals: np.ndarray
    indices: np.ndarray
    scalars: Optional[np.ndarray]
    threshold: float
    minimum: float
    maximum: float

    @property
    def vertex_count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def triangle_count(self) -> int:
        return int(self.indices.size // 3)

def masked_field_volume(
    grids: FieldGrids,
    mode: str,
    *,
    cast_only: bool = False,
    frame_index: Optional[int] = None,
) -> Optional[Tuple[np.ndarray, float, float]]:
    nx, ny, nz = grids.lattice.dims
    if frame_index is not None:
        stack = grids.stacks.get(mode)
        if stack is None:
            return None
        index = int(np.clip(frame_index, 0, stack.count - 1))
        dense = stack.frames[index].astype(np.float32) / _QUANT_MAX
        minimum, maximum = 0.0, 1.0
    else:
        grid = grids.scalars.get(mode)
        if grid is None:
            return None
        dense = grid.dequantize()
        minimum, maximum = grid.minimum, grid.maximum

    keep = grids.mask != 0
    if cast_only:
        keep &= grids.mask == PART_TYPE_CAST + 1
    if not np.any(keep):
        return None

    span = max(maximum - minimum, 1e-6)
    outside = minimum - span
    blanked = np.where(keep, dense, outside).astype(np.float32)
    return blanked.reshape((nz, ny, nx)).transpose(2, 1, 0), minimum, maximum

def sample_field_at(
    grids: FieldGrids,
    mode: str,
    points: np.ndarray,
    *,
    frame_index: Optional[int] = None,
) -> Optional[np.ndarray]:
    nx, ny, nz = grids.lattice.dims
    if frame_index is not None:
        stack = grids.stacks.get(mode)
        if stack is None:
            return None
        index = int(np.clip(frame_index, 0, stack.count - 1))
        dense = stack.frames[index].astype(np.float32) / _QUANT_MAX
    else:
        grid = grids.scalars.get(mode)
        if grid is None:
            return None
        dense = grid.dequantize()

    volume = dense.reshape((nz, ny, nx))
    origin = grids.lattice.origin
    spacing = grids.lattice.spacing if grids.lattice.spacing > 0 else 1.0
    coords = np.vstack(
        [
            (points[:, 2] - origin[2]) / spacing,
            (points[:, 1] - origin[1]) / spacing,
            (points[:, 0] - origin[0]) / spacing,
        ]
    )
    sampled = ndimage.map_coordinates(volume, coords, order=1, mode="nearest")
    return np.asarray(sampled, dtype=np.float32)

def build_envelope(
    grids: FieldGrids,
    mode: str,
    threshold: float,
    *,
    cast_only: bool = False,
    with_scalars: bool = True,
    frame_index: Optional[int] = None,
    smooth_sigma: float = 0.75,
) -> Optional[Envelope]:
    from core.surface_extraction import contour_dense_grid

    prepared = masked_field_volume(grids, mode, cast_only=cast_only, frame_index=frame_index)
    if prepared is None:
        return None
    volume, minimum, maximum = prepared

    surface = contour_dense_grid(
        volume,
        grids.lattice.origin,
        grids.lattice.spacing,
        threshold,
        smooth_sigma=smooth_sigma,
    )

    scalars: Optional[np.ndarray] = None
    if with_scalars and surface.vertex_count:
        scalars = sample_field_at(grids, mode, surface.positions, frame_index=frame_index)

    return Envelope(
        positions=surface.positions,
        normals=surface.normals,
        indices=surface.indices,
        scalars=scalars,
        threshold=float(threshold),
        minimum=float(minimum),
        maximum=float(maximum),
    )

def envelope_bytes(envelope: Envelope) -> bytes:
    parts = [
        np.ascontiguousarray(envelope.positions, dtype="<f4").tobytes(),
        np.ascontiguousarray(envelope.normals, dtype="<f4").tobytes(),
        np.ascontiguousarray(envelope.indices, dtype="<u4").tobytes(),
    ]
    if envelope.scalars is not None:
        parts.append(np.ascontiguousarray(envelope.scalars, dtype="<f4").tobytes())
    return b"".join(parts)
