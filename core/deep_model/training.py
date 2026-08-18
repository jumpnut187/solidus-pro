"""Deep model training: loop, masked loss, metrics, checkpointing.

Shared by the desktop UI (Deep Model menu), the auto-train worker, and the
CLI (``tools/train_deep_model.py``).  Trains the :class:`UNet3D` on captured
records (``data/surrogate_dataset``) at the configured grid resolution and
saves artifacts to ``models/deep_model/trained/``:

* ``weights.pt`` — torch state dict.
* ``manifest.json`` — standard surrogate manifest (kind="torch_unet",
  backend="torch") so the shared ``load_model`` dispatch reads it unchanged.
* ``deep_manifest.json`` — deep-only metadata (targets, quantization).

Captured records are never modified; training refits on the entire
accumulated dataset (identical guarantee to the numpy surrogate).
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core.deep_model.dataset import DeepDataset, make_train_val_datasets
from core.deep_model.model import (
    DEFAULT_CONDITIONING_SIZE,
    SETTINGS_RESOLUTIONS,
    DeepModelManifest,
    UNet3D,
    levels_for_resolution,
    save_deep_manifest,
)
from core.surrogate.dataset import DEFAULT_DATASET_DIR
from core.surrogate.metrics import aggregate_schedule_metrics, compute_field_metrics
from core.surrogate.model import ModelManifest, _save_manifest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL_OUTPUT = os.path.join(PROJECT_ROOT, "models", "deep_model", "trained")

DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 1e-3
MIN_RECORDS = 4

ProgressCallback = Optional[Callable[[str], None]]
DetailedProgressCallback = Optional[Callable[[int, int], None]]
CancelCallback = Optional[Callable[[], bool]]

_SUPPORTED_ANALYSES = [
    "shrinkage",
    "porosity",
    "hotspot",
    "niyama",
    "sand_burn_in",
    "gas_porosity",
    "fraction_liquid",
    "chvorinov",
]


class DeepTrainingCancelledError(RuntimeError):
    """Raised when the user requests cancellation during deep training."""


def _check_cancel(cancel: CancelCallback) -> None:
    if cancel is not None and cancel():
        raise DeepTrainingCancelledError("Training cancelled")


def masked_schedule_loss(pred: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Masked MSE over start (linear), end (log space), local (linear).

    Cells excluded from the loss: inactive cells (occupancy < 0.5), chill
    cells (part_type 2), and cells with zero reference end time.  The end
    head predicts in log space (mirrors the linear surrogate's
    ``log(end_time)`` fit).
    """
    mask = batch["mask"]  # (B, R, R, R) bool
    if not bool(mask.any()):
        return torch.zeros((), device=pred.device)

    pred_start, pred_log_end, pred_local = pred[:, 0], pred[:, 1], pred[:, 2]
    ref_start, ref_end, ref_local = batch["start"], batch["end"], batch["local"]

    log_end_ref = torch.log(torch.clamp(ref_end, min=0.1))
    start_loss = F.mse_loss(pred_start[mask], ref_start[mask])
    end_loss = F.mse_loss(pred_log_end[mask], log_end_ref[mask])
    local_loss = F.mse_loss(pred_local[mask], ref_local[mask])
    return (start_loss + end_loss + local_loss) / 3.0


def _evaluate(
    model: UNet3D, dataset: Optional[DeepDataset], device: torch.device
) -> Dict[str, Any]:
    """Per-field metrics on the validation split (or None -> empty dict)."""
    if dataset is None or len(dataset) == 0:
        return {}
    model.eval()
    all_start_pred: List[np.ndarray] = []
    all_start_ref: List[np.ndarray] = []
    all_end_pred: List[np.ndarray] = []
    all_end_ref: List[np.ndarray] = []
    all_local_pred: List[np.ndarray] = []
    all_local_ref: List[np.ndarray] = []
    with torch.no_grad():
        for item in dataset:
            channels = item.channels.unsqueeze(0).to(device)
            cond = item.conditioning.unsqueeze(0).to(device)
            pred = model(channels, cond)[0].cpu()
            mask = item.mask.numpy()
            pred_start, pred_log_end, pred_local = pred[0].numpy(), pred[1].numpy(), pred[2].numpy()
            all_start_pred.append(pred_start[mask])
            all_end_pred.append(np.exp(pred_log_end[mask]))
            all_local_pred.append(pred_local[mask])
            all_start_ref.append(item.start.numpy()[mask])
            all_end_ref.append(item.end.numpy()[mask])
            all_local_ref.append(item.local.numpy()[mask])
    metric_dict = {
        "start_time": compute_field_metrics(
            np.concatenate(all_start_pred), np.concatenate(all_start_ref)
        ),
        "end_time": compute_field_metrics(
            np.concatenate(all_end_pred), np.concatenate(all_end_ref)
        ),
        "local_time": compute_field_metrics(
            np.concatenate(all_local_pred), np.concatenate(all_local_ref)
        ),
    }
    return metric_dict


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_deep_model(
    data_dir: str = DEFAULT_DATASET_DIR,
    output_dir: str = DEFAULT_MODEL_OUTPUT,
    grid: int = 32,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    name: str = "Trained Deep Surrogate (3D U-Net)",
    limit: Optional[int] = None,
    min_records: int = MIN_RECORDS,
    progress_cb: ProgressCallback = None,
    detailed_cb: DetailedProgressCallback = None,
    cancel_cb: CancelCallback = None,
) -> Dict[str, Any]:
    """Train the 3D U-Net from captured records and persist it.

    Returns:
        Dict with ``records``, ``train``, ``test``, ``headline``, ``saved_to``,
        ``alloys``, ``epochs``, ``device``.

    Raises:
        ValueError: When no usable training records are available.
        DeepTrainingCancelledError: When cancellation is requested.
    """
    grid = int(grid)
    if grid not in (8, 16, 32, 64, 128):
        raise ValueError(
            f"Grid resolution must be in {{8, 16, 32, 64, 128}}, got {grid}"
        )

    _check_cancel(cancel_cb)
    if progress_cb:
        progress_cb("Loading captured training data...")
    train_dataset, val_dataset = make_train_val_datasets(data_dir, resolution=grid, limit=limit)
    if len(train_dataset) < min_records:
        raise ValueError(
            f"Only {len(train_dataset)} usable training record(s) found in {data_dir}. "
            "Run more traditional simulations (with Surrogate training capture "
            "enabled), or use tools/train_deep_model.py once the dataset grows."
        )

    device = _select_device()
    device_name = "mps" if device.type == "mps" else "cpu"
    cond_size = train_dataset.conditioning_size
    model = UNet3D(
        conditioning_size=cond_size,
        levels=levels_for_resolution(grid),
    ).to(device)

    if progress_cb:
        progress_cb(
            f"Training U-Net on {len(train_dataset)} records at {grid}^3 "
            f"({device_name}) for {epochs} epochs..."
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        collate_fn=DeepDataset.collate,
    )

    best_loss = float("inf")
    first_loss = None
    last_loss = None
    for epoch in range(int(epochs)):
        _check_cancel(cancel_cb)
        model.train()
        epoch_losses = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            pred = model(batch["channels"], batch["conditioning"])
            loss = masked_schedule_loss(pred, batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        scheduler.step()
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        if first_loss is None:
            first_loss = mean_loss
        last_loss = mean_loss
        if mean_loss < best_loss:
            best_loss = mean_loss
        if detailed_cb:
            detailed_cb(epoch + 1, int(epochs))
        if progress_cb and (epoch % 10 == 0 or epoch == int(epochs) - 1):
            progress_cb(f"  epoch {epoch + 1}/{epochs} — loss {mean_loss:.4e}")

    _check_cancel(cancel_cb)
    if progress_cb:
        progress_cb("Evaluating on validation records...")
    metric_dict = _evaluate(model, val_dataset, device)
    headline = aggregate_schedule_metrics(metric_dict) if metric_dict else {}
    if not headline:
        headline = {
            "rmse_s": 0.0,
            "mae_s": 0.0,
            "max_error_s": 0.0,
            "relative_rmse_end": 0.0,
            "confidence": 0.0,
        }

    alloys = sorted({r["alloy_name"] for r in train_dataset.records})
    saved_to = save_trained(
        model=model,
        headline=headline,
        name=name,
        grid=grid,
        alloys=alloys,
        output_dir=output_dir,
        notes="PyTorch 3D U-Net + FiLM surrogate trained on captured runs. "
              "Retrains automatically on new captures; never erases captured data.",
        cond_size=cond_size,
        epochs=int(epochs),
        device_name=device_name,
        best_loss=best_loss,
        first_loss=first_loss,
        last_loss=last_loss,
    )
    if progress_cb:
        progress_cb(f"✓ Deep model saved to {saved_to}")
    return {
        "records": len(train_dataset.records) + (len(val_dataset.records) if val_dataset else 0),
        "train": len(train_dataset),
        "test": len(val_dataset) if val_dataset else 0,
        "headline": headline,
        "saved_to": saved_to,
        "alloys": alloys,
        "epochs": int(epochs),
        "device": device_name,
        "first_loss": first_loss,
        "last_loss": last_loss,
    }


def save_trained(
    model: UNet3D,
    headline: Dict[str, Any],
    name: str,
    grid: int,
    alloys: List[str],
    output_dir: str,
    notes: str,
    cond_size: int = DEFAULT_CONDITIONING_SIZE,
    epochs: int = 0,
    device_name: str = "",
    best_loss: float = 0.0,
    first_loss: Optional[float] = None,
    last_loss: Optional[float] = None,
) -> str:
    """Persist the trained U-Net weights + standard + deep manifests."""
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "weights.pt"))

    manifest = ModelManifest(
        name=name,
        version="1.0.0",
        kind="torch_unet",
        backend="torch",
        grid_resolution=int(grid),
        supported_analyses=list(_SUPPORTED_ANALYSES),
        alloy_coverage=list(alloys),
        accuracy={
            "rmse_s": round(float(headline.get("rmse_s", 0.0)), 2),
            "mae_s": round(float(headline.get("mae_s", 0.0)), 2),
            "max_error_s": round(float(headline.get("max_error_s", 0.0)), 2),
            "confidence": round(float(headline.get("confidence", 0.0)), 3),
        },
        trained_on=time.strftime("%Y-%m-%d"),
        notes=notes,
    )
    _save_manifest(manifest, output_dir)

    deep = DeepModelManifest(
        accuracy=manifest.accuracy,
        alloy_coverage=list(alloys),
        trained_on=manifest.trained_on,
        training_device=device_name,
        conditioning_size=int(cond_size),
        epochs=int(epochs),
        notes=f"best_val_mse={best_loss:.4e} "
              f"first_epoch_mse={first_loss:.4e} last_epoch_mse={last_loss:.4e}"
              if first_loss is not None else f"best_val_mse={best_loss:.4e}",
    )
    save_deep_manifest(deep, output_dir)
    return output_dir
