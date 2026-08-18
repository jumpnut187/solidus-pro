"""Torch dataset over captured surrogate records for deep training.

Reuses ``core.surrogate.dataset.load_dataset`` (schema v2 ``.npz`` records
from ``data/surrogate_dataset``) — no new capture code.  Records captured at
any resolution (16/32/64) are resampled to the training resolution (32^3
default) with block-average pooling (nearest-neighbour fallback).

Each item yields the channels, conditioning vector, target schedule fields
(start / end / local), and an occupancy mask marking cells to include in the
loss (occupancy > 0.5, chill cells excluded — same exclusion the surrogate
pipeline applies).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from core.surrogate.dataset import DEFAULT_DATASET_DIR, load_dataset

CHILL_PART_TYPE = 2  # encoded as part_type / 3.0 -> 2/3 in channel 1
CHILL_CHANNEL_TOL = 0.05


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
    """Loss mask: occupied cells, excluding chill cells."""
    occ = channels[0] > 0.5
    chill = np.abs(channels[1] - CHILL_PART_TYPE / 3.0) < CHILL_CHANNEL_TOL
    return occ & ~chill


class DeepDataset(Dataset):
    """Torch dataset over captured records, resampled to training resolution.

    Args:
        records: Output of ``core.surrogate.dataset.load_dataset``.
        resolution: Training grid resolution (default 32).
        split: One of ``"train"`` / ``"val"`` / ``"all"``.  The last 20% of
            records form validation (same convention as ``train_numpy``).
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        resolution: int = 32,
        split: str = "all",
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
            split_at = max(1, int(len(records) * 0.8))
            indices = indices[:split_at] if split == "train" else indices[split_at:]
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
            mask = _occupancy_mask(channels)
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
            item.mask &= item.end > 0.0
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
    resolution: int = 32,
    limit: Optional[int] = None,
) -> Tuple[DeepDataset, Optional[DeepDataset]]:
    """Load records and split into train / validation datasets.

    Returns:
        ``(train, val)``; ``val`` is None when fewer than 5 records exist (the
        whole dataset trains — same spirit as the numpy surrogate's split).
    """
    records = load_dataset(data_dir, limit=limit)
    if not records:
        raise ValueError(f"No usable training records found in {data_dir}")
    if len(records) < 5:
        return DeepDataset(records, resolution=resolution, split="all"), None
    train = DeepDataset(records, resolution=resolution, split="train")
    val = DeepDataset(records, resolution=resolution, split="val")
    return train, val
