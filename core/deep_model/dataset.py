"""Torch dataset over captured surrogate records for deep training.

Reuses ``core.surrogate.dataset.load_dataset`` (schema v2 ``.npz`` records
from ``data/surrogate_dataset``) — no new capture code.  Records captured at
any resolution (16/32/64) are resampled to the training resolution (64^3
default) with occupancy-aware block pooling (linear-zoom fallback).

Each item yields the channels, conditioning vector, target schedule fields
(start / end / local), and an occupancy mask marking cells to include in the
loss and metrics (occupancy > 0.5, chill cells excluded, reference end > 0).

Train / validation membership comes from
``core.surrogate.split.grouped_train_val_indices`` so no geometry appears on
both sides of the split.

Grid axis order is ``(z, y, x)`` (``core.surrogate.fields._scatter`` packs
``ix + R * (iy + R * iz)``), so ``normalized_height`` varies along axis 0 and
``augment_batch`` only flips / rotates the trailing two axes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from core.deep_model.model import DEFAULT_TRAINING_RESOLUTION, TARGET_NAMES
from core.surrogate.dataset import DEFAULT_DATASET_DIR, load_dataset
from core.surrogate.fields import PART_TYPE_DIVISOR
from core.surrogate.split import (
    DEFAULT_SEED,
    DEFAULT_VAL_FRACTION,
    geometry_group_key,
    grouped_train_val_indices,
)
from core.part_manager import PART_TYPE_CHILL

CHILL_PART_TYPE = PART_TYPE_CHILL  # encoded as part_type / PART_TYPE_DIVISOR
CHILL_CHANNEL_TOL = 0.05
MIN_GEOMETRY_GROUPS = 2


def _block_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    """Block-average an array whose trailing 3 dims are divisible by factor."""
    target = arr.shape[-1] // factor
    shape = arr.shape[:-3] + (target, factor, target, factor, target, factor)
    return arr.reshape(shape).mean(axis=(-5, -3, -1))


def _block_sum(arr: np.ndarray, factor: int) -> np.ndarray:
    target = arr.shape[-1] // factor
    shape = arr.shape[:-3] + (target, factor, target, factor, target, factor)
    return arr.reshape(shape).sum(axis=(-5, -3, -1))


def _zoom(arr: np.ndarray, target: int) -> np.ndarray:
    from scipy.ndimage import zoom

    factor = target / arr.shape[-1]
    return zoom(arr, (1.0, factor, factor, factor), order=1).astype(np.float32)


def resample_record(
    channels: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    local: np.ndarray,
    target: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scatter-consistent resampling of a captured record to ``target``.

    Mirrors the ``_scatter`` semantics used at capture time: a cell is
    occupied when *any* subcell holds voxels, and channel / target values are
    the average over the occupied subcells only (never diluted by empty
    cells).  Without this, mean-pooling drops sparse geometry below the
    occupancy > 0.5 activity threshold.
    """
    channels = np.asarray(channels, dtype=np.float32)
    src = channels.shape[-1]
    if src == target:
        occ = (channels[0] > 0.5).astype(np.float32)
        channels[0] = occ
        return channels, start, end, local

    def _scatter_like(arr: np.ndarray) -> np.ndarray:
        """Average over occupied subcells, binary occupancy for channel 0.

        Only the multi-channel ``channels`` array re-binarizes its occupancy
        channel after pooling; single-field schedule grids (``start``/``end``/
        ``local``, passed as ``(1, R, R, R)``) must keep their pooled values.
        """
        arr = np.asarray(arr, dtype=np.float32)
        is_channels = arr.ndim == 4 and arr.shape[0] > 1
        if src % target == 0:
            factor = src // target
            count = _block_sum(channels[0][None], factor)[0]
            out = _block_sum(arr, factor) / np.maximum(count[None], 1)
            if arr.ndim == 4:
                out[:, count == 0] = 0.0
                if is_channels:
                    out[0] = (count > 0).astype(np.float32)
            else:
                out[count == 0] = 0.0
            return out.astype(np.float32)
        out = _zoom(arr[None], target)[0] if arr.ndim == 3 else _zoom(arr, target)
        if is_channels:
            out[0] = (out[0] > 0.5).astype(np.float32)
        return out.astype(np.float32)

    res_channels = _scatter_like(channels)
    res_start = _scatter_like(start.reshape(1, src, src, src))[0]
    res_end = _scatter_like(end.reshape(1, src, src, src))[0]
    res_local = _scatter_like(local.reshape(1, src, src, src))[0]
    return res_channels, res_start, res_end, res_local


def resample_grid(arr: np.ndarray, target: int) -> np.ndarray:
    """Plain block-average resample of a ``(C, R, R, R)`` grid (kept for tests)."""
    arr = np.asarray(arr, dtype=np.float32)
    src = arr.shape[-1]
    if src == target:
        return arr
    if src % target == 0:
        return _block_mean(arr, src // target).astype(np.float32)
    return _zoom(arr, target)


@dataclass
class DeepItem:
    """One training item as torch tensors."""

    channels: torch.Tensor          # (7, R, R, R) float32
    conditioning: torch.Tensor      # (K,) float32
    start: torch.Tensor             # (R, R, R)
    end: torch.Tensor               # (R, R, R)
    local: torch.Tensor             # (R, R, R)
    mask: torch.Tensor              # (R, R, R) bool - cells in the loss


def _occupancy_mask(channels: np.ndarray) -> np.ndarray:
    """Occupied cells, excluding chill cells (part_type 2)."""
    occ = channels[0] > 0.5
    chill = np.abs(channels[1] - CHILL_PART_TYPE / PART_TYPE_DIVISOR) < CHILL_CHANNEL_TOL
    return occ & ~chill


def evaluation_mask(channels: np.ndarray, ref_end: np.ndarray) -> np.ndarray:
    """Cells that count for the loss and for every reported metric.

    Active (occupancy > 0.5), not chill, and reference end time > 0.  Shared
    by the dataset items and ``training._evaluate`` so the loss mask and the
    metric mask are the same set of cells.
    """
    return _occupancy_mask(channels) & (np.asarray(ref_end) > 0.0)


def log1p_targets(start: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
    """Stack the network targets ``log1p(start), log1p(local)`` -> ``(B, 2, ...)``."""
    return torch.stack(
        [torch.log1p(torch.clamp(start, min=0.0)), torch.log1p(torch.clamp(local, min=0.0))],
        dim=1,
    )


def augment_batch(batch: Dict[str, torch.Tensor], rng: torch.Generator) -> Dict[str, torch.Tensor]:
    """Random flips on the x / y axes and 90-degree rotations about z.

    Applied identically to channels, targets and mask.  Grid axis 0 is the
    vertical (``normalized_height`` and gravity), so it is never touched.
    Tensor dims: ``(B, C, z, y, x)`` for channels and ``(B, z, y, x)`` for
    the fields, so y is the second-to-last dim and x the last.
    """
    flip_y = bool(torch.rand((), generator=rng) < 0.5)
    flip_x = bool(torch.rand((), generator=rng) < 0.5)
    quarter_turns = int(torch.randint(0, 4, (), generator=rng))

    def _apply(t: torch.Tensor) -> torch.Tensor:
        y_dim, x_dim = t.ndim - 2, t.ndim - 1
        if flip_y:
            t = torch.flip(t, dims=(y_dim,))
        if flip_x:
            t = torch.flip(t, dims=(x_dim,))
        if quarter_turns:
            t = torch.rot90(t, quarter_turns, dims=(y_dim, x_dim))
        return t

    # Grids are (B, C, z, y, x) or (B, z, y, x); the conditioning vector (B, K)
    # has no spatial axes and passes through untouched.
    return {k: (_apply(v) if v.ndim >= 4 else v) for k, v in batch.items()}


class DeepDataset(Dataset):
    """Torch dataset over captured records, resampled to training resolution.

    Args:
        records: Output of ``core.surrogate.dataset.load_dataset``.
        resolution: Training grid resolution (default 64).
        split: One of ``"train"`` / ``"val"`` / ``"all"``.  Train / val
            membership is the geometry-grouped split from
            ``core.surrogate.split`` (``val_fraction`` 0.2, ``seed`` 0).
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        resolution: int = DEFAULT_TRAINING_RESOLUTION,
        split: str = "all",
        val_fraction: float = DEFAULT_VAL_FRACTION,
        seed: int = DEFAULT_SEED,
    ):
        if resolution not in (8, 16, 32, 64, 128):
            raise ValueError(
                f"Training resolution must be in {{8, 16, 32, 64, 128}}, got {resolution}"
            )
        if not records:
            raise ValueError("DeepDataset requires at least one record")
        if split not in ("train", "val", "all"):
            raise ValueError(f"Unknown split: {split}")

        self.resolution = int(resolution)
        indices = list(range(len(records)))
        if split in ("train", "val"):
            train_idx, val_idx = grouped_train_val_indices(records, val_fraction, seed)
            indices = train_idx if split == "train" else val_idx
        self._records = [records[i] for i in indices]
        if not self._records:
            raise ValueError(f"No records in split '{split}'")

        self._items: List[DeepItem] = []
        for record in self._records:
            src_res = int(record["channels"].shape[-1])
            channels, start, end, local = resample_record(
                record["channels"],
                np.asarray(record["start"], dtype=np.float32).reshape(src_res, src_res, src_res),
                np.asarray(record["end"], dtype=np.float32).reshape(src_res, src_res, src_res),
                np.asarray(record["local"], dtype=np.float32).reshape(src_res, src_res, src_res),
                self.resolution,
            )
            mask = evaluation_mask(channels, end)
            item = DeepItem(
                channels=torch.from_numpy(channels).float(),
                conditioning=torch.from_numpy(
                    np.asarray(record["conditioning"].vector, dtype=np.float32)
                ).float(),
                start=torch.from_numpy(start).float(),
                end=torch.from_numpy(end).float(),
                local=torch.from_numpy(local).float(),
                mask=torch.from_numpy(mask).bool(),
            )
            self._items.append(item)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> DeepItem:
        return self._items[idx]

    @property
    def records(self) -> List[Dict[str, Any]]:
        """The underlying record dicts in this split."""
        return self._records

    @property
    def conditioning_size(self) -> int:
        return int(self._items[0].conditioning.shape[0])

    @property
    def geometry_groups(self) -> int:
        """Number of distinct geometry groups among the records."""
        return len({geometry_group_key(r) for r in self._records})

    def target_normalization(self) -> Tuple[np.ndarray, np.ndarray]:
        """Mean / std of the ``TARGET_NAMES`` log1p targets over masked cells.

        Computed on this dataset only; callers pass the *train* split so the
        validation statistics never leak into the model.  A degenerate std
        (constant target) is floored at 1.0.
        """
        sums = np.zeros(len(TARGET_NAMES), dtype=np.float64)
        sq_sums = np.zeros(len(TARGET_NAMES), dtype=np.float64)
        count = 0
        for item in self._items:
            mask = item.mask
            if not bool(mask.any()):
                continue
            targets = log1p_targets(item.start[None], item.local[None])[0]  # (2, R, R, R)
            vals = targets[:, mask].double().numpy()  # (2, n)
            sums += vals.sum(axis=1)
            sq_sums += (vals ** 2).sum(axis=1)
            count += vals.shape[1]
        if count == 0:
            return np.zeros(len(TARGET_NAMES), dtype=np.float32), np.ones(len(TARGET_NAMES), dtype=np.float32)
        mean = sums / count
        var = np.maximum(sq_sums / count - mean ** 2, 0.0)
        std = np.sqrt(var)
        std = np.where(std < 1e-6, 1.0, std)
        return mean.astype(np.float32), std.astype(np.float32)

    @staticmethod
    def collate(batch: List[DeepItem]) -> Dict[str, torch.Tensor]:
        """Collate a batch of :class:`DeepItem` into stacked tensors."""
        return {
            "channels": torch.stack([it.channels for it in batch]),
            "conditioning": torch.stack([it.conditioning for it in batch]),
            "start": torch.stack([it.start for it in batch]),
            "end": torch.stack([it.end for it in batch]),
            "local": torch.stack([it.local for it in batch]),
            "mask": torch.stack([it.mask for it in batch]),
        }


def make_train_val_datasets(
    data_dir: str = DEFAULT_DATASET_DIR,
    resolution: int = DEFAULT_TRAINING_RESOLUTION,
    limit: Optional[int] = None,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
) -> Tuple[DeepDataset, Optional[DeepDataset]]:
    """Load records and split them by geometry group into train / validation.

    Returns:
        ``(train, val)``; ``val`` is None only when the grouped split holds
        nothing out (a single geometry group, or ``val_fraction`` 0).
    """
    records = load_dataset(data_dir, limit=limit)
    if not records:
        raise ValueError(f"No usable training records found in {data_dir}")
    train_idx, val_idx = grouped_train_val_indices(records, val_fraction, seed)
    train = DeepDataset(
        [records[i] for i in train_idx], resolution=resolution, split="all"
    )
    if not val_idx:
        return train, None
    val = DeepDataset([records[i] for i in val_idx], resolution=resolution, split="all")
    return train, val
