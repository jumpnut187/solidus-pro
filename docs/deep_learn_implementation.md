# Deep Learning Module Implementation Plan

Solidus Pro — cross-platform deep learning solidification surrogate.

Module: `core/deep_model/` — a PyTorch 3D U-Net that predicts solidification schedule fields (start/end/local times) and deploys to three NPU targets: Apple ANE, Google Tensor, Snapdragon.

This module is **separate** from `core/surrogate/` (the lightweight NumPy linear-log model). It reuses the surrogate's data capture, field encoding, conditioning vectors, and pipeline interface so both modules coexist.

---

## 1. Goals

1. Predict solidification start/end/local times from geometry channels + alloy/config conditioning at 32³ grid resolution.
2. Feed predictions through the existing post-analysis stack (shrinkage, porosity, hotspot, Niyama, etc.) via the `SurrogateModel` interface — zero engine changes.
3. Export one trained model to all three NPU formats:
   - Apple ANE → Core ML (`.mlpackage`)
   - Google Tensor → TFLite (int8 quantized, NNAPI)
   - Snapdragon → ONNX (fp16, ONNX Runtime QNN EP)
4. Retrain automatically in the background as new simulation runs are captured — no manual intervention.
5. Stay an optional dependency (torch not required to launch the app).

## 2. Design decisions

| Decision | Choice | Reason |
|---|---|---|
| Framework | PyTorch | Only framework that exports to all three NPU targets; MPS backend trains on Apple GPU |
| Training device | MPS (Apple GPU), CPU fallback | This is a macOS desktop app |
| Grid resolution | 32³ default, configurable | Matches user requirement; 64³ is memory-heavy and slow to generate data for |
| Capture reuse | `core/surrogate.dataset` schema v2 `.npz` records | Zero new capture code; existing runs feed both modules |
| Resampling | Records downsampled to 32³ at load time | Existing 64³ (and 16³) captures work unchanged |
| Model interface | `DeepSurrogateModel(SurrogateModel)` ABC | Drops into `run_surrogate_pipeline` and UI without changes |
| Loader | `torch_unet` kind added to `core/surrogate/model.py:load_model` (lazy import) | One small edit; UI load path untouched |
| Auto-train | Background `QThread` + `QFileSystemWatcher` on the dataset dir | Continuous training on captured runs |
| Dependency policy | torch + export libs optional in `requirements.txt` | Matches repo's GPU-extras pattern; app starts without torch |

## 3. Package layout (`core/deep_model/`)

| File | Responsibility |
|---|---|
| `__init__.py` | Public API: `train_deep_model`, `load_deep_model`, `export_for_target`, `detect_deep_backend`. Lazy imports — app must start without torch |
| `model.py` | `UNet3D` (torch `nn.Module`) + FiLM conditioning + log-space end-time head |
| `dataset.py` | `DeepDataset` (torch `Dataset`) over `load_dataset()` records; 32³ resampling; occupancy masking; train/val split |
| `training.py` | Train/val loop, masked loss, metrics via `core.surrogate.metrics`, checkpointing, `save_trained` → `models/deep_model/trained/` |
| `inference.py` | `DeepSurrogateModel` (ABC impl) + `load_deep_model()`; backend-aware predict (Core ML → torch MPS → torch CPU) |
| `backend.py` | Deep backend ladder + probes (mirrors `core/surrogate/backend.py` pattern) |
| `export.py` | Export pipeline: torch → Core ML, torch → ONNX, torch → TFLite; quantization; per-target validation |
| `autotrain.py` | Continuous-training controller: dataset change detection, debounce/cooldown, retrain trigger logic |

### 3.1 Public API (`__init__.py`)

- `train_deep_model(data_dir, output_dir, ...)` — full train from captured records, returns summary dict
- `load_deep_model(path)` — returns `DeepSurrogateModel` or error result
- `export_for_target(model_path, target, out_dir, ...)` — target in `{"coreml", "onnx", "tflite"}`
- `detect_deep_backend()` — ladder: Core ML → Torch MPS → Torch CPU
- `DeepModelManifest` — json manifest: `kind="torch_unet"`, target list, accuracy, alloy coverage, trained_on

## 4. Model architecture (UNet3D + FiLM)

Input: `(7, R, R, R)` channels — occupancy, part_type, surface_distance, local_modulus, normalized_height, chill_boost, riser_slowdown (same channel order as `core.surrogate.fields.CHANNELS`).

Conditioning: normalized alloy + config vector from `core.surrogate.conditioning.build_conditioning_vector` (size ≥ 20).

Network:

1. Encoder — 4 levels, each: Conv3D + GroupNorm + SiLU, stride-2 downsampling. Base channels 32, doubling per level (32 → 64 → 128 → 256).
2. Bottleneck — Conv3D block at lowest resolution.
3. Decoder — 4 levels: transposed-conv upsample, skip-concat from encoder, Conv3D + GroupNorm + SiLU.
4. FiLM — at each decoder level, conditioning vector → MLP → per-channel scale/shift applied to feature maps (modulates physics per alloy/config).
5. Head — 1×1×1 Conv3D → `(3, R, R, R)` = start_time, end_time, local_time. End-time head predicts in log space (mirrors linear surrogate's `log(end_time)` fit), output clamped to `[0.1, max_simulation_time]`.

Loss: masked MSE. Inactive cells (occupancy < 0.5) and chill cells excluded. End time in log space; start/local in linear space.

## 5. Data pipeline

1. Data source: `data/surrogate_dataset/run_*.npz` + `index.csv` (schema v2, appended per captured simulation — never erased).
2. `DeepDataset` loads records via `core.surrogate.dataset.load_dataset()`.
3. Each record resampled to training resolution (32³ default) via average pooling / nearest-neighbor — handles records captured at any resolution (16/32/64).
4. Split: 80/20 train/val (last 20% as validation, matching `train_numpy` convention).
5. Targets: solidification_start_times, solidification_end_times, local_solidification_time grids from the record.
6. No new capture code. Future runs captured by existing `capture_run()` flow (Surrogate auto-capture setting) feed both modules automatically.

## 6. Continuous background training (auto-train)

The core requirement: training on existing captured runs now, and on future runs automatically.

### 6.1 Trigger

- `QFileSystemWatcher` watches `data/surrogate_dataset/` (desktop UI only).
- New `run_*.npz` files appear → schedule a retrain after debounce.

### 6.2 Policy

1. **Initial training:** on first app launch with deep training enabled — if no deep model exists and dataset ≥ `min_records` (default 4), train on all existing runs.
2. **Retrain on capture:** new records → wait `cooldown_seconds` (default 300) since last completed training → retrain on the **entire accumulated dataset** (refit semantics, same as surrogate — never erases data).
3. **Skip conditions:** torch not installed; a training already active; dataset unchanged since last model (record count + file mtimes); export in progress.

### 6.3 Execution

- `DeepAutoTrainWorker` (QThread) in `ui/workers.py` — same pattern as `SurrogateTrainWorker` (progress signals, cancellation).
- Manual "Train Deep Model…" and auto-train share the worker with mutual exclusion.
- New model saved to `models/deep_model/trained/` (weights `.pt` + manifest.json + exported artifacts).
- Existing captured data is never deleted or modified (identical guarantee to the surrogate module).

### 6.4 Settings (`DeepModelSettings` in `config/settings.py`)

| Setting | Default | Env var |
|---|---|---|
| `enabled` | false | `CASTSIM_DEEP_ENABLED` |
| `min_records` | 4 | `CASTSIM_DEEP_MIN_RECORDS` |
| `cooldown_seconds` | 300 | `CASTSIM_DEEP_COOLDOWN_SECONDS` |
| `epochs` | 100 | `CASTSIM_DEEP_EPOCHS` |
| `batch_size` | 4 | `CASTSIM_DEEP_BATCH_SIZE` |
| `learning_rate` | 1e-3 | `CASTSIM_DEEP_LEARNING_RATE` |
| `grid_resolution` | 32 | `CASTSIM_DEEP_GRID_RESOLUTION` |
| `model_dir` | models/deep_model | `CASTSIM_DEEP_MODEL_DIR` |
| `export_dir` | models/deep_model/export | `CASTSIM_DEEP_EXPORT_DIR` |

Sections also added to `castsim.yaml` / `castsim.toml` defaults.

## 7. NPU export matrix

| Target | Artifact | On-device runtime | Conversion path | Desktop validation |
|---|---|---|---|---|
| Apple ANE | `.mlpackage` | Core ML, compute_units=all (ANE/GPU/CPU) | torch trace → coremltools | Direct on macOS (ANE active if shapes fit ANE tiles) |
| Google Tensor | `.tflite` (int8 PTQ) | Android NNAPI (Edge TPU delegate) | torch → ONNX → onnx2tf | tflite-runtime interpreter (CPU) |
| Snapdragon | `.onnx` (fp16) | ONNX Runtime QNN EP (Hexagon/HTP) | torch → ONNX | onnxruntime CPU EP (graph validation) |

Notes:

- All three artifacts produced by `export.py` from the same trained weights.
- Quantization: int8 post-training quantization for TFLite; fp16 for ONNX.
- Export records target + quantization info in the deep manifest.
- Actual on-device execution happens in future mobile apps — this repo produces and validates the artifacts.

## 8. Integration points

1. `core/surrogate/model.py` — add `torch_unet` kind dispatch in `load_model` (lazy import of `core.deep_model.inference`). Only edit to the surrogate module.
2. `config/settings.py` — `DeepModelSettings` dataclass, env var mapping, yaml/toml defaults.
3. `ui/workers.py` — `DeepTrainWorker` (manual), `DeepAutoTrainWorker` (watcher-driven).
4. `ui/main_window.py` — "Deep Model" menu: Train Deep Model…, Load Deep Model…, Export for NPU…, Auto-Train toggle. Inference reuses `SurrogateWorker` unchanged (model implements `SurrogateModel` ABC).
5. `ui/deep_model_dialog.py` — train dialog (epochs/lr/resolution summary) + export dialog (target picker: ANE / Tensor / Snapdragon, with per-target notes).
6. `tools/train_deep_model.py` — CLI wrapper (mirrors `tools/train_surrogate_model.py`).
7. `tools/export_deep_model.py` — CLI export wrapper (`--target coreml|onnx|tflite`).
8. `requirements.txt` — optional DEEP LEARNING section (see below).
9. `AGENTS.md` — document the module, auto-train behavior, and the never-erases-data guarantee.

## 8.1 Confirmed wiring decisions

Reviewed against the current codebase; the following decisions are locked in.

1. **`Backend.TORCH` enum value** (`core/surrogate/backend.py`). `SurrogateModel.backend` constructs `Backend(self.manifest.backend)`, so a deep manifest with backend `"torch"` raises `ValueError` today. Add `TORCH = "torch"` plus a `_probe_torch()` entry (import torch, report MPS/CPU) and a `backend_device_name` entry. **Not** added to `_PREFERENCE_ORDER` — the numpy-surrogate auto ladder is untouched; deep backend selection lives in `deep_model/backend.py`.
2. **Two edits to `core/surrogate/backend.py`** — this is the second shared-module edit beyond the `model.py` dispatch listed above.
3. **Manifest layout** — deep training writes the *standard* `manifest.json` (`kind="torch_unet"`, `backend="torch"`) so the existing `load_model` dispatch reads it unchanged, plus a separate `deep_manifest.json` for export targets and quantization info. No custom parsing in the surrogate module.
4. **UI placement** — separate `&Deep Model` menu after the Surrogate menu. Loading a deep model sets `self.surrogate_model`, so the existing "Run Surrogate Simulation" → `SurrogateWorker` → `run_surrogate_pipeline` path runs it with zero changes.
5. **Autoload** — `_autoload_surrogate_model` gains `models/deep_model/trained` as a candidate when `deep.enabled`.
6. **`load_model` dispatch** — the `torch_unet` branch replaces the dead `coreml`/`mlx` branch (that path is superseded by this module).
7. **Grid resolution restriction** — the 4-level stride-2 UNet requires R divisible by 16; the setting is restricted to {16, 32, 64, 128}.
8. **Auto-train does not swap the loaded model** — refits and saves to `models/deep_model/trained/`, logs to the UI, but leaves the active model untouched.

## 9. Tests & verification

`tests/test_deep_model.py` (unittest pattern, mirrors `tests/test_surrogate.py`):

1. Architecture — forward pass on fake channels returns `(3, R, R, R)`, finite values.
2. Tiny overfit — train on 2 synthetic records at 8³, assert loss decreases.
3. ABC contract — `DeepSurrogateModel` runs `run_surrogate_pipeline` end-to-end on a box mesh.
4. Export smoke — ONNX export + onnxruntime CPU validation; TFLite export + tflite-runtime interpreter validation; Core ML export guarded by `skipIf` (macOS + coremltools required).
5. Auto-train logic — cooldown/skip conditions unit-tested without Qt (pure policy function in `autotrain.py`).

`tests/verify_deep_benchmarks.py` (standalone script, mirrors `verify_thermal_benchmarks.py`):

- Deep model vs full solver on held-out geometries (boxes, plates, bars).
- Report RMSE/MAE/relative error for start/end/local times.

Verify commands (from AGENTS.md):

```bash
source venv/bin/activate
python -m unittest discover tests
python -m compileall config core physics ui web
```

## 10. Dependencies (all optional)

```text
# DEEP LEARNING MODULE (optional; core/deep_model/)
# Install with: pip install -r requirements.txt --extra-index-url ... (or per-platform)
torch>=2.2.0               # Training + inference (MPS on Apple Silicon)
coremltools>=7.0           # Apple ANE export (.mlpackage)
onnx>=1.16.0               # Snapdragon/ONNX export + graph validation
onnxruntime>=1.18.0        # ONNX CPU EP validation
onnx2tf>=1.22.0            # Google Tensor TFLite export (torch → ONNX → tflite)
tflite-runtime>=2.14.0     # TFLite interpreter validation
```

- Torch only imported lazily inside `core/deep_model/` — app startup must not require it.
- Core ML / onnx2tf / tflite imports are guarded per-export-target.

## 11. Implementation phases

| Phase | Work | Done when |
|---|---|---|
| 0 | Install deps: `venv/bin/pip install torch coremltools` (MPS on Apple Silicon; torch>=2.6, coremltools>=8.3 for Python 3.13). ONNX/TFLite toolchain deferred to phase 5 | `torch.backends.mps.is_available()` is true |
| 1 | `model.py` — UNet3D + FiLM + head | Forward pass shape correct; tiny fake-input test passes |
| 2 | `dataset.py` — DeepDataset, resampling, split | Loads real captured records at 32³; masks inactive cells |
| 3 | `training.py` — loop, loss, metrics, checkpoint, save_trained | Tiny overfit test passes; artifacts written to `models/deep_model/trained/` |
| 4 | `inference.py` + loader dispatch | `load_model()` loads deep model; `run_surrogate_pipeline` end-to-end test passes |
| 5 | `export.py` — Core ML / ONNX / TFLite | All three export smoke tests pass (guarded skips where deps absent) |
| 6 | CLI tools — train + export | Both CLIs run end-to-end |
| 7 | Settings + env vars | `CASTSIM_DEEP_*` parsed; yaml/toml defaults present |
| 8 | UI — workers, menu, dialogs, auto-train watcher | Auto-train retrains after new capture (verified via UI logs) |
| 9 | Tests + verify script | Full `unittest` suite + `verify_deep_benchmarks.py` pass |
| 10 | requirements.txt + AGENTS.md | Deps documented; module + auto-train behavior documented |

## 12. Risks & caveats

1. **Data volume is the dominant quality risk.** A 3D U-Net needs far more data than the linear surrogate. Initial training on a handful of runs will be weak. Mitigation: auto-retrain accumulates data continuously; CLI `--generate` flag (reuse `generate_dataset`-style solver sweep) available for bulk seeding later.
2. **ANE tile constraints.** Core ML may fall back to GPU/CPU when tensor shapes don't fit ANE tiles. Export must report the actual compute unit used.
3. **Training cost.** Retraining on the full dataset each time grows slower as data accumulates. Mitigation: MPS training is fast at 32³; cooldown prevents thrashing; cancellation supported.
4. **Heavy optional dependency.** Torch must never be imported at app startup. Lazy imports enforced in `__init__.py`.
5. **Mobile NPUs are out of repo scope.** Artifacts are produced and validated here; Android/iOS apps consuming them are future work.
6. **mypy.** `core/deep_model/` left out of `mypy.ini` allowlist initially; add files later if type-checking is wanted.
7. **Python 3.13 wheels.** This repo runs Python 3.13.14 / arm64. torch>=2.6 and coremltools>=8.3 support it; `onnx2tf` may lag — TFLite export must fail with a clear message rather than crash.

## 13. Definition of done

1. All 11 phases complete (0–10).
2. Full regression suite passes (`python -m unittest discover tests`).
3. Deep model trains on existing captured runs and auto-retrains on new captures.
4. Deep model runs a simulation via the UI with all post analyses working.
5. Exports to all three NPU targets produce validated artifacts.
6. AGENTS.md documents the module and the never-erases-data guarantee.
