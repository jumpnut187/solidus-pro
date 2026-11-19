"""Deep model training: loop, masked loss, metrics, checkpointing.

Shared by the desktop UI (Deep Model menu), the auto-train worker, and the
CLI (``tools/train_deep_model.py``).  Trains the :class:`UNet3D` on captured
records (``data/surrogate_dataset``) at the configured grid resolution and
saves artifacts to ``models/deep_model/trained/``:

* ``weights.pt`` — torch state dict of the best-validation epoch (includes
  the ``target_mean`` / ``target_std`` normalization buffers).
* ``manifest.json`` — standard surrogate manifest (kind="torch_unet",
  backend="torch") so the shared ``load_model`` dispatch reads it unchanged.
* ``deep_manifest.json`` — deep-only metadata (targets, normalization
  statistics, training summary, quantization).

Targets are ``log1p(start)`` and ``log1p(local)``, normalized with the train
split mean / std; ``end`` is derived as ``start + local``.  Validation
metrics run the exact inference reconstruction
(``core.deep_model.inference.reconstruct_schedule``) on active, non-chill
cells with reference end > 0, and report a constant-mean baseline for end
time so the skill score means something.

Captured records are never modified; training refits on the entire
accumulated dataset (identical guarantee to the numpy surrogate).
"""

from __future__ import annotations

import copy
import math
import os
import shutil
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from core.deep_model.dataset import (
    MIN_GEOMETRY_GROUPS,
    DeepDataset,
    augment_batch,
    log1p_targets,
    make_train_val_datasets,
)
from core.deep_model.inference import max_simulation_time_from, reconstruct_schedule
from core.deep_model.model import (
    DEFAULT_BASE_CHANNELS,
    DEFAULT_CONDITIONING_SIZE,
    DEFAULT_TRAINING_RESOLUTION,
    TARGET_NAMES,
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
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_GRAD_CLIP_NORM = 1.0
DEFAULT_PATIENCE = 15
WARMUP_FRACTION = 0.05
MIN_LR_FRACTION = 0.01
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


def masked_normalized_loss(
    pred_normalized: torch.Tensor, batch: Dict[str, torch.Tensor], model: UNet3D
) -> torch.Tensor:
    """Masked MSE between the normalized heads and the normalized log1p targets.

    ``pred_normalized`` is ``UNet3D.forward_normalized`` output
    ``(B, 2, R, R, R)``.  Targets are ``log1p(start)``, ``log1p(local)``
    standardized with the model's stored ``target_mean`` / ``target_std``.
    Cells outside ``batch["mask"]`` (inactive, chill, or zero reference end)
    do not contribute.
    """
    mask = batch["mask"]  # (B, R, R, R) bool
    if not bool(mask.any()):
        return torch.zeros((), device=pred_normalized.device)
    targets = log1p_targets(batch["start"], batch["local"])  # (B, 2, R, R, R)
    mean = model.target_mean.view(1, -1, 1, 1, 1)
    std = model.target_std.view(1, -1, 1, 1, 1)
    targets = (targets - mean) / std
    mask2 = mask.unsqueeze(1).expand_as(targets)
    return F.mse_loss(pred_normalized[mask2], targets[mask2])


def _validation_loss(model: UNet3D, loader: Optional[DataLoader], device: torch.device) -> Optional[float]:
    if loader is None:
        return None
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred = model.forward_normalized(batch["channels"], batch["conditioning"])
            loss = masked_normalized_loss(pred, batch, model)
            total += float(loss.detach().cpu()) * int(batch["channels"].shape[0])
            count += int(batch["channels"].shape[0])
    return total / count if count else None


def _masked_end_mean(dataset: DeepDataset) -> float:
    """Mean reference end time over the loss-mask cells of ``dataset``."""
    total = 0.0
    count = 0
    for item in dataset:
        vals = item.end[item.mask]
        total += float(vals.sum())
        count += int(vals.numel())
    return total / count if count else 0.0


def _evaluate(
    model: UNet3D,
    dataset: Optional[DeepDataset],
    device: torch.device,
    baseline_end_mean: float,
) -> Dict[str, Any]:
    """Per-field metrics on ``dataset`` through the exact inference path.

    Each item goes through ``UNet3D.forward`` and
    ``reconstruct_schedule`` (active-cell zeroing and the
    ``max_simulation_time`` clamp), identical to
    ``DeepSurrogateModel.predict``.  Metrics use the item mask (active,
    non-chill, reference end > 0).  ``baseline_end_mean`` is the train-split
    constant predictor for end time; its RMSE on the same cells gives the
    skill score ``1 - rmse / baseline_rmse`` (clipped to [0, 1]).
    """
    if dataset is None or len(dataset) == 0:
        return {}
    model.eval()
    preds: Dict[str, List[np.ndarray]] = {"start": [], "end": [], "local": []}
    refs: Dict[str, List[np.ndarray]] = {"start": [], "end": [], "local": []}
    with torch.no_grad():
        for item, record in zip(dataset, dataset.records):
            channels_np = item.channels.numpy()
            channels = item.channels.unsqueeze(0).to(device)
            cond = item.conditioning.unsqueeze(0).to(device)
            network_output = model(channels, cond)[0].cpu().numpy()
            schedule = reconstruct_schedule(
                network_output, channels_np, max_simulation_time_from(record["conditioning"])
            )
            mask = item.mask.numpy()
            for i, name in enumerate(("start", "end", "local")):
                preds[name].append(schedule[i][mask])
            refs["start"].append(item.start.numpy()[mask])
            refs["end"].append(item.end.numpy()[mask])
            refs["local"].append(item.local.numpy()[mask])

    ref_end = np.concatenate(refs["end"])
    if ref_end.size == 0:
        return {}
    metric_dict = {
        f"{name}_time": compute_field_metrics(np.concatenate(preds[name]), np.concatenate(refs[name]))
        for name in ("start", "end", "local")
    }
    baseline_pred = np.full_like(ref_end, float(baseline_end_mean), dtype=np.float64)
    baseline_rmse = float(np.sqrt(np.mean((baseline_pred - ref_end) ** 2)))
    end_rmse = float(metric_dict["end_time"]["rmse"])
    skill = 1.0 - end_rmse / baseline_rmse if baseline_rmse > 1e-12 else 0.0
    metric_dict["baseline"] = {
        "end_mean_s": float(baseline_end_mean),
        "end_rmse": baseline_rmse,
        "skill_end": float(np.clip(skill, 0.0, 1.0)),
        "n_cells": int(ref_end.size),
    }
    return metric_dict


def _headline(metric_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Headline accuracy: per-field aggregate plus honest end-time skill."""
    if not metric_dict:
        return {
            "rmse_s": 0.0,
            "mae_s": 0.0,
            "max_error_s": 0.0,
            "relative_rmse_end": 0.0,
            "confidence": 0.0,
            "end_rmse_s": 0.0,
            "baseline_end_rmse_s": 0.0,
            "validated": False,
        }
    headline = aggregate_schedule_metrics(metric_dict)
    baseline = metric_dict.get("baseline", {})
    # One baseline definition everywhere: the constant fit on the TRAIN
    # split scored on val (same as core.surrogate.training.train_numpy).
    skill = float(round(baseline.get("skill_end", 0.0), 3))
    headline["end_rmse_s"] = float(metric_dict["end_time"]["rmse"])
    headline["baseline_end_rmse_s"] = float(baseline.get("end_rmse", 0.0))
    headline["baseline_rmse_s"] = headline["baseline_end_rmse_s"]
    headline["skill"] = skill
    headline["confidence"] = skill
    headline["validated"] = True
    return headline


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _warmup_cosine_lambda(total_steps: int) -> Callable[[int], float]:
    warmup = max(1, int(round(WARMUP_FRACTION * total_steps)))

    def _lr(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return MIN_LR_FRACTION + (1.0 - MIN_LR_FRACTION) * cosine

    return _lr


def train_deep_model(
    data_dir: str = DEFAULT_DATASET_DIR,
    output_dir: str = DEFAULT_MODEL_OUTPUT,
    grid: int = DEFAULT_TRAINING_RESOLUTION,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    name: str = "Trained Deep Surrogate (3D U-Net)",
    limit: Optional[int] = None,
    min_records: int = MIN_RECORDS,
    progress_cb: ProgressCallback = None,
    detailed_cb: DetailedProgressCallback = None,
    cancel_cb: CancelCallback = None,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    grad_clip_norm: float = DEFAULT_GRAD_CLIP_NORM,
    patience: int = DEFAULT_PATIENCE,
    augment: bool = True,
    base_channels: int = DEFAULT_BASE_CHANNELS,
    seed: int = 0,
) -> Dict[str, Any]:
    """Train the 3D U-Net from captured records and persist the best epoch.

    Returns:
        Dict with ``records``, ``train``, ``test``, ``headline``, ``saved_to``,
        ``alloys``, ``epochs`` (requested), ``epochs_run``, ``best_epoch``,
        ``device``, ``train_loss`` / ``val_loss`` (best epoch),
        ``first_loss`` / ``last_loss`` (train loss), ``early_stopped``.

    Raises:
        ValueError: When no usable training records are available, or the
            dataset has fewer than two geometry groups (no honest validation
            split exists).
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
    n_records = len(train_dataset) + (len(val_dataset) if val_dataset else 0)
    if n_records < min_records:
        raise ValueError(
            f"Only {n_records} usable training record(s) found in {data_dir}. "
            "Run more traditional simulations (with Surrogate training capture "
            "enabled), or use tools/train_deep_model.py once the dataset grows."
        )
    geometry_groups = train_dataset.geometry_groups + (
        val_dataset.geometry_groups if val_dataset else 0
    )
    if geometry_groups < MIN_GEOMETRY_GROUPS or val_dataset is None:
        raise ValueError(
            f"Dataset in {data_dir} has {n_records} record(s) but only "
            f"{geometry_groups} distinct geometry group(s); at least "
            f"{MIN_GEOMETRY_GROUPS} are required so a held-out geometry can "
            "validate the model. Capture runs on a different part geometry first."
        )

    torch.manual_seed(int(seed))
    device = _select_device()
    device_name = "mps" if device.type == "mps" else "cpu"
    cond_size = train_dataset.conditioning_size
    model = UNet3D(
        conditioning_size=cond_size,
        base_channels=int(base_channels),
        levels=levels_for_resolution(grid),
    )
    target_mean, target_std = train_dataset.target_normalization()
    model.set_target_normalization(target_mean, target_std)
    model = model.to(device)
    baseline_end_mean = _masked_end_mean(train_dataset)

    if progress_cb:
        progress_cb(
            f"Training U-Net on {len(train_dataset)} records "
            f"({len(val_dataset)} held-out, {geometry_groups} geometries) at {grid}^3 "
            f"({device_name}) for up to {epochs} epochs, patience {patience}..."
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        collate_fn=DeepDataset.collate,
        generator=torch.Generator().manual_seed(int(seed)),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=int(batch_size), shuffle=False, collate_fn=DeepDataset.collate
    )
    total_steps = max(1, int(epochs) * max(1, len(loader)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _warmup_cosine_lambda(total_steps))
    aug_rng = torch.Generator().manual_seed(int(seed) + 1)

    best_val = float("inf")
    best_train = None
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    first_loss = None
    last_loss = None
    last_val = None
    epochs_run = 0
    early_stopped = False
    for epoch in range(int(epochs)):
        _check_cancel(cancel_cb)
        model.train()
        epoch_losses = []
        for batch in loader:
            if augment:
                batch = augment_batch(batch, aug_rng)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            pred = model.forward_normalized(batch["channels"], batch["conditioning"])
            loss = masked_normalized_loss(pred, batch, model)
            loss.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            optimizer.step()
            scheduler.step()
            epoch_losses.append(float(loss.detach().cpu()))
        epochs_run = epoch + 1
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        if first_loss is None:
            first_loss = mean_loss
        last_loss = mean_loss
        last_val = _validation_loss(model, val_loader, device)
        if last_val is not None and not math.isfinite(last_val):
            raise RuntimeError(
                f"Validation loss is not finite at epoch {epochs_run} "
                f"(train loss {mean_loss:.4e}); training diverged. "
                "Lower the learning rate or check the dataset for degenerate records."
            )

        if last_val < best_val:
            best_val = last_val
            best_train = mean_loss
            best_epoch = epochs_run
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if detailed_cb:
            detailed_cb(epoch + 1, int(epochs))
        if progress_cb and (epoch % 10 == 0 or epoch == int(epochs) - 1):
            progress_cb(
                f"  epoch {epoch + 1}/{epochs} — train loss {mean_loss:.4e} "
                f"val loss {last_val:.4e} (best {best_val:.4e} @ {best_epoch})"
            )
        if patience and epochs_without_improvement >= int(patience):
            early_stopped = True
            if progress_cb:
                progress_cb(
                    f"  early stop at epoch {epoch + 1}: no val improvement for "
                    f"{patience} epochs (best {best_val:.4e} @ {best_epoch})"
                )
            break

    _check_cancel(cancel_cb)
    model.load_state_dict(best_state)
    if progress_cb:
        progress_cb("Evaluating best epoch on held-out geometries...")
    metric_dict = _evaluate(model, val_dataset, device, baseline_end_mean)
    headline = _headline(metric_dict)
    if progress_cb and metric_dict:
        progress_cb(
            f"  val end RMSE {headline['end_rmse_s']:.1f}s vs constant-mean "
            f"baseline {headline['baseline_end_rmse_s']:.1f}s "
            f"(skill {headline['confidence']:.2f})"
        )

    alloys = sorted({r["alloy_name"] for r in train_dataset.records})
    training_summary = {
        "optimizer": "adamw",
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "grad_clip_norm": float(grad_clip_norm),
        "schedule": "warmup_cosine",
        "augment": bool(augment),
        "patience": int(patience),
        "epochs_requested": int(epochs),
        "epochs_run": int(epochs_run),
        "best_epoch": int(best_epoch),
        "early_stopped": bool(early_stopped),
        "train_loss_first": first_loss,
        "train_loss_last": last_loss,
        "train_loss_best_epoch": best_train,
        "val_loss_best": best_val if math.isfinite(best_val) else None,
        "val_loss_last": last_val,
        "train_records": len(train_dataset),
        "val_records": len(val_dataset),
        "geometry_groups": int(geometry_groups),
        "val_metrics": metric_dict,
    }
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
        epochs=int(epochs_run),
        device_name=device_name,
        training_summary=training_summary,
    )
    if progress_cb:
        progress_cb(f"✓ Deep model saved to {saved_to}")
    return {
        "records": n_records,
        "train": len(train_dataset),
        "test": len(val_dataset),
        "headline": headline,
        "saved_to": saved_to,
        "alloys": alloys,
        "epochs": int(epochs),
        "epochs_run": int(epochs_run),
        "best_epoch": int(best_epoch),
        "early_stopped": bool(early_stopped),
        "device": device_name,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "train_loss": best_train,
        "val_loss": best_val if math.isfinite(best_val) else None,
        "metrics": metric_dict,
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
    training_summary: Optional[Dict[str, Any]] = None,
) -> str:
    """Persist the U-Net weights (current state) + standard + deep manifests."""
    os.makedirs(output_dir, exist_ok=True)
    # Exported artifacts (Core ML / ONNX / TFLite) were built from the previous
    # weights.  Leaving them in place lets backend="auto" prefer a stale
    # .mlpackage over the fresh torch weights, so drop them; export is
    # reproducible from weights.pt.
    stale_export = os.path.join(output_dir, "export")
    if os.path.isdir(stale_export):
        shutil.rmtree(stale_export, ignore_errors=True)
    for name in ("export_log.json",):
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            os.remove(path)
    torch.save(model.state_dict(), os.path.join(output_dir, "weights.pt"))
    training_summary = dict(training_summary or {})

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
            "end_rmse_s": round(float(headline.get("end_rmse_s", 0.0)), 2),
            "baseline_rmse_s": round(float(headline.get("baseline_rmse_s", 0.0)), 2),
            "skill": round(float(headline.get("skill", 0.0)), 3),
        },
        trained_on=time.strftime("%Y-%m-%d"),
        notes=notes,
    )
    _save_manifest(manifest, output_dir)

    accuracy = dict(manifest.accuracy)
    accuracy["end_rmse_s"] = round(float(headline.get("end_rmse_s", 0.0)), 2)
    accuracy["baseline_end_rmse_s"] = round(float(headline.get("baseline_end_rmse_s", 0.0)), 2)
    accuracy["metric_cells"] = "active, non-chill, reference end > 0; inference reconstruction"
    val_loss = training_summary.get("val_loss_best")
    train_loss = training_summary.get("train_loss_best_epoch")
    deep = DeepModelManifest(
        accuracy=accuracy,
        alloy_coverage=list(alloys),
        trained_on=manifest.trained_on,
        training_device=device_name,
        conditioning_size=int(cond_size),
        base_channels=int(model.base_channels),
        epochs=int(epochs),
        target_names=list(TARGET_NAMES),
        target_mean=[float(v) for v in model.target_mean.detach().cpu().tolist()],
        target_std=[float(v) for v in model.target_std.detach().cpu().tolist()],
        training=_json_safe(training_summary),
        notes=(
            f"best epoch {training_summary.get('best_epoch', 0)}: "
            f"train_loss={_fmt(train_loss)} val_loss={_fmt(val_loss)} "
            "(masked MSE on normalized log1p targets); "
            "confidence = 1 - end_rmse / constant-mean baseline rmse on held-out geometries"
        ),
    )
    save_deep_manifest(deep, output_dir)
    return output_dir


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{float(value):.4e}"


def _json_safe(value: Any) -> Any:
    """Convert numpy scalars / non-finite floats so ``json.dump`` succeeds."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value
