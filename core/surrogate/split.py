from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

DEFAULT_VAL_FRACTION = 0.2
DEFAULT_SEED = 0

def geometry_group_key(record: Dict[str, Any]) -> str:
    channels = np.asarray(record["channels"])
    occ = (channels[0] > 0.5).astype(np.uint8)
    h = hashlib.sha1()
    h.update(np.asarray(occ.shape, dtype=np.int64).tobytes())
    h.update(np.packbits(occ.reshape(-1)).tobytes())
    return h.hexdigest()[:16]

def grouped_train_val_indices(
    records: Sequence[Dict[str, Any]],
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
) -> Tuple[List[int], List[int]]:
    n = len(records)
    if n == 0:
        return [], []
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")

    groups: Dict[str, List[int]] = {}
    for idx, record in enumerate(records):
        groups.setdefault(geometry_group_key(record), []).append(idx)

    if len(groups) < 2 or val_fraction <= 0.0:
        return list(range(n)), []

    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)

    target = val_fraction * n
    val: List[int] = []
    train: List[int] = []
    for key in keys:
        members = groups[key]
        if len(val) < target and len(train) + len(members) < n:
            val.extend(members)
        else:
            train.extend(members)
    if not train:
        train, val = val, []
    return sorted(train), sorted(val)

def split_records(
    records: Sequence[Dict[str, Any]],
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_idx, val_idx = grouped_train_val_indices(records, val_fraction, seed)
    return [records[i] for i in train_idx], [records[i] for i in val_idx]
