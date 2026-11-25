from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object

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

CHILL_PART_TYPE = PART_TYPE_CHILL
CHILL_CHANNEL_TOL = 0.05
MIN_GEOMETRY_GROUPS = 2

def _block_mean(arr: np.ndarray, factor: int) -> np.ndarray:
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
    channels = np.asarray(channels, dtype=np.float32)
    src = channels.shape[-1]
    if src == target:
        occ = (channels[0] > 0.5).astype(np.float32)
        channels[0] = occ
        return channels, start, end, local

    def _scatter_like(arr: np.ndarray) -> np.ndarray:
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
    arr = np.asarray(arr, dtype=np.float32)
    src = arr.shape[-1]
    if src == target:
        return arr
    if src % target == 0:
        return _block_mean(arr, src // target).astype(np.float32)
    return _zoom(arr, target)

@dataclass
class DeepItem:

    channels: torch.Tensor
    conditioning: torch.Tensor
    start: torch.Tensor
    end: torch.Tensor
    local: torch.Tensor
    mask: torch.Tensor

def _occupancy_mask(channels: np.ndarray) -> np.ndarray:
    occ = channels[0] > 0.5
    chill = np.abs(channels[1] - CHILL_PART_TYPE / PART_TYPE_DIVISOR) < CHILL_CHANNEL_TOL
    return occ & ~chill

def evaluation_mask(channels: np.ndarray, ref_end: np.ndarray) -> np.ndarray:
    return _occupancy_mask(channels) & (np.asarray(ref_end) > 0.0)

def log1p_targets(start: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [torch.log1p(torch.clamp(start, min=0.0)), torch.log1p(torch.clamp(local, min=0.0))],
        dim=1,
    )

def augment_batch(batch: Dict[str, torch.Tensor], rng: torch.Generator) -> Dict[str, torch.Tensor]:
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

    return {k: (_apply(v) if v.ndim >= 4 else v) for k, v in batch.items()}

class DeepDataset(Dataset):

    def __init__(
        self,
        records: List[Dict[str, Any]],
        resolution: int = DEFAULT_TRAINING_RESOLUTION,
        split: str = "all",
        val_fraction: float = DEFAULT_VAL_FRACTION,
        seed: int = DEFAULT_SEED,
    ):
        if torch is None:
            raise ImportError(
                "PyTorch is not installed. Install the deep-learning extras with "
                "`pip install -r requirements.txt` (DEEP LEARNING MODULE section)."
            )
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
        return self._records

    @property
    def conditioning_size(self) -> int:
        return int(self._items[0].conditioning.shape[0])

    @property
    def geometry_groups(self) -> int:
        return len({geometry_group_key(r) for r in self._records})

    def target_normalization(self) -> Tuple[np.ndarray, np.ndarray]:
        sums = np.zeros(len(TARGET_NAMES), dtype=np.float64)
        sq_sums = np.zeros(len(TARGET_NAMES), dtype=np.float64)
        count = 0
        for item in self._items:
            mask = item.mask
            if not bool(mask.any()):
                continue
            targets = log1p_targets(item.start[None], item.local[None])[0]
            vals = targets[:, mask].double().numpy()
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
