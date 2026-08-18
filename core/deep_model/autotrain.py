"""Continuous-training policy (pure logic, Qt-free).

The desktop UI watches ``data/surrogate_dataset/`` for new ``run_*.npz``
records and asks this module whether to retrain.  Keeping the policy pure
lets the skip/cooldown conditions be unit-tested without Qt or torch.

Policy (from the implementation plan):

1. Initial training: no deep model exists and dataset >= ``min_records``.
2. Retrain on capture: new records appeared since the last completed
   training *and* ``cooldown_seconds`` have elapsed since it finished.
3. Skip: torch not installed; a training already active; dataset unchanged
   since the last model; export in progress; fewer than ``min_records``.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

MIN_RECORDS_DEFAULT = 4
COOLDOWN_SECONDS_DEFAULT = 300
DEBOUNCE_SECONDS_DEFAULT = 10


@dataclass
class AutoTrainState:
    """Persisted state used for cooldown and change detection."""

    signature_count: int = 0
    signature_mtimes: Tuple[float, ...] = ()
    last_completed_at: float = 0.0
    model_exists: bool = False

    def to_dict(self) -> dict:
        return {
            "signature_count": int(self.signature_count),
            "signature_mtimes": list(self.signature_mtimes),
            "last_completed_at": float(self.last_completed_at),
            "model_exists": bool(self.model_exists),
        }

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "AutoTrainState":
        data = data or {}
        return cls(
            signature_count=int(data.get("signature_count", 0)),
            signature_mtimes=tuple(float(m) for m in data.get("signature_mtimes", [])),
            last_completed_at=float(data.get("last_completed_at", 0.0)),
            model_exists=bool(data.get("model_exists", False)),
        )


def dataset_signature(data_dir: str) -> Optional[Tuple[int, Tuple[float, ...]]]:
    """Snapshot the dataset: record count + sorted mtimes.

    Returns None when the directory does not exist or has no records.
    """
    if not os.path.isdir(data_dir):
        return None
    records = [
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if name.startswith("run_") and name.endswith(".npz")
    ]
    if not records:
        return None
    mtimes = sorted(os.path.getmtime(path) for path in records)
    return (len(records), tuple(mtimes))


AUTOTRAIN_STATE_FILENAME = ".autotrain_state.json"


def load_state(model_dir: str) -> AutoTrainState:
    """Load the persisted auto-train state (cooldown + change detection)."""
    path = os.path.join(model_dir, AUTOTRAIN_STATE_FILENAME)
    if not os.path.exists(path):
        return AutoTrainState()
    try:
        import json

        with open(path) as fh:
            return AutoTrainState.from_dict(json.load(fh))
    except Exception:
        return AutoTrainState()


def save_state(model_dir: str, state: AutoTrainState) -> str:
    """Persist the auto-train state next to the model artifacts."""
    import json

    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, AUTOTRAIN_STATE_FILENAME)
    with open(path, "w") as fh:
        json.dump(state.to_dict(), fh, indent=2)
    return path


def should_retrain(
    *,
    enabled: bool,
    torch_available: bool,
    training_active: bool,
    export_in_progress: bool,
    current_signature: Optional[Tuple[int, Tuple[float, ...]]],
    state: AutoTrainState,
    min_records: int = MIN_RECORDS_DEFAULT,
    cooldown_seconds: float = COOLDOWN_SECONDS_DEFAULT,
    now: Optional[float] = None,
) -> Tuple[bool, str]:
    """Decide whether a retrain should run right now.

    Returns:
        ``(should_run, reason)`` — reason is human-readable, used for UI logs.
    """
    now = time.time() if now is None else now

    if not enabled:
        return False, "Deep auto-train is disabled"
    if not torch_available:
        return False, "torch not installed — skipping deep auto-train"
    if training_active:
        return False, "A training run is already active"
    if export_in_progress:
        return False, "An export is in progress"
    if current_signature is None:
        return False, "No captured records in the dataset directory"
    count, mtimes = current_signature
    if count < min_records:
        return False, (
            f"Dataset has {count} record(s), need at least {min_records} — "
            "waiting for more captured runs"
        )

    if not state.model_exists:
        return True, (
            f"No deep model exists yet and the dataset has {count} records — "
            "running initial training"
        )

    if count == state.signature_count and tuple(mtimes) == tuple(state.signature_mtimes):
        return False, "Dataset unchanged since the last training run"

    elapsed = now - state.last_completed_at
    if elapsed < cooldown_seconds:
        return False, (
            f"Cooldown active ({elapsed:.0f}s / {cooldown_seconds:.0f}s) since "
            "the last completed training"
        )

    return True, (
        f"Dataset grew to {count} records — retraining on the entire "
        "accumulated dataset"
    )
