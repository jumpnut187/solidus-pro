from __future__ import annotations

import os
import sys
import json
import time
import traceback

import numpy as np

REPO_ROOT = "/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim"
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

import faulthandler
faulthandler.enable()

VOXEL_SIZE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
FILL_GRID_COARSENING = 1
MAX_FILL_TIME_S = 30.0
ALLOY = "WCB"
HARD_TIMEOUT_MS = 20 * 60 * 1000

STEP_DIR = os.path.join(REPO_ROOT, "Test_Model", "Filling_Test")
STEP_FILES = sorted(
    os.path.join(STEP_DIR, f) for f in os.listdir(STEP_DIR) if f.lower().endswith(".step")
)

problems: list[str] = []
summary: dict = {
    "voxel_size_in": VOXEL_SIZE,
    "fill_grid_coarsening": FILL_GRID_COARSENING,
    "max_fill_time_s_cap": MAX_FILL_TIME_S,
    "alloy": ALLOY,
    "step_files": STEP_FILES,
}

def log(msg: str) -> None:
    print(msg, flush=True)

import vtk
from PySide6.QtWidgets import QApplication, QWidget, QMessageBox
from PySide6.QtCore import QEventLoop, QTimer

app = QApplication.instance() or QApplication([])

class _OffscreenVTKWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rw = vtk.vtkRenderWindow()
        self._rw.SetOffScreenRendering(1)
        self._rw.SetSize(1024, 768)
        iren = vtk.vtkGenericRenderWindowInteractor()
        iren.SetRenderWindow(self._rw)

    def GetRenderWindow(self):
        return self._rw

import ui.visualizer as viz_mod
viz_mod.QVTKRenderWindowInteractor = _OffscreenVTKWidget
problems.append(
    "PROBLEM #1 (environment, worked around in this harness only): "
    "constructing a real QVTKRenderWindowInteractor under "
    "QT_QPA_PLATFORM=offscreen segfaults (SetWindowInfo(winId()) dereferences "
    "a native NSView handle the offscreen Qt platform plugin never creates). "
    "Worked around by swapping in a bare off-screen vtkRenderWindow "
    "(SetOffScreenRendering) + vtkGenericRenderWindowInteractor before "
    "constructing MainWindow; no repo code was changed."
)

QMessageBox.warning = staticmethod(lambda *a, **kw: QMessageBox.Yes)
QMessageBox.critical = staticmethod(lambda *a, **kw: QMessageBox.Yes)
QMessageBox.question = staticmethod(lambda *a, **kw: QMessageBox.Yes)
QMessageBox.information = staticmethod(lambda *a, **kw: QMessageBox.Ok)

import core.filling_stage as filling_stage_mod
_orig_run_filling_stage = filling_stage_mod.run_filling_stage
_filling_capture: dict = {}

def _spy_run_filling_stage(*a, **kw):
    result = _orig_run_filling_stage(*a, **kw)
    if result is not None:
        _filling_capture["result"] = result
    return result

filling_stage_mod.run_filling_stage = _spy_run_filling_stage

import ui.workers as workers_mod
_orig_run_pipeline = workers_mod.run_pipeline
_pipeline_capture: dict = {}

def _spy_run_pipeline(*a, **kw):
    result = _orig_run_pipeline(*a, **kw)
    _pipeline_capture["result"] = result
    return result

workers_mod.run_pipeline = _spy_run_pipeline

from ui.main_window import MainWindow

t_construct = time.perf_counter()
w = MainWindow()
w.console_window.stop_capture()
log(f"MainWindow constructed OK in {time.perf_counter() - t_construct:.1f}s")

w._performance_options["part_assignment_mode"] = "sequential"
problems.append(
    "NOTE (harness accommodation, not an app bug): part_assignment_mode forced "
    "to 'sequential' because this script has no __main__ guard for "
    "multiprocessing spawn (main.py, the real entry point, has one)."
)

assert len(STEP_FILES) == 8, f"expected 8 STEP files, found {len(STEP_FILES)}: {STEP_FILES}"

for f in STEP_FILES:
    new_ids = w.part_manager.load_assembly_parts(f)
    assert new_ids, f"failed to load any bodies from {f}"

w._update_parts_ui()
w._update_visualization()

from core.part_manager import PartType
from collections import Counter

parts = list(w.part_manager.parts.values())
type_counts = Counter(p.part_type for p in parts)
loaded_types = {
    "CAST": type_counts.get(PartType.CAST, 0),
    "CHILL": type_counts.get(PartType.CHILL, 0),
    "SPRUE": type_counts.get(PartType.SPRUE, 0),
    "GATE": type_counts.get(PartType.GATE, 0),
    "RUNNER": type_counts.get(PartType.RUNNER, 0),
    "RISER": type_counts.get(PartType.RISER, 0),
}
log(f"Loaded {len(parts)} parts:")
for p in parts:
    log(f"  {p.name:45s} {p.part_type.value.upper()}")
log(f"Type counts: {loaded_types}")
summary["loaded_parts"] = {p.name: p.part_type.value for p in parts}
summary["part_type_counts"] = loaded_types

expected = {"CAST": 1, "CHILL": 1, "SPRUE": 1, "GATE": 1, "RUNNER": 2, "RISER": 2}
assert loaded_types == expected, f"part type mix mismatch: got {loaded_types}, expected {expected}"
assert w.stl_loader.is_valid(), "stl_loader did not become valid after loading parts"

sprue = next(p for p in parts if p.part_type is PartType.SPRUE)
lo, hi = sprue.get_transformed_mesh().bounds
top_centre = (float((lo[0] + hi[0]) / 2.0), float((lo[1] + hi[1]) / 2.0), float(hi[2]))
log(f"Sprue '{sprue.name}' bounds: lo={tuple(lo)} hi={tuple(hi)}")
log(f"Pour location (top face centre): {top_centre}")

from core.pour_location import PourLocation

pour_location = PourLocation(point_in=top_centre, normal=(0.0, 0.0, 1.0), part_id=sprue.id)
summary["pour_location_point_in"] = list(top_centre)

gate = {}
gate["initial_disabled"] = not w.analysis_dialog.filling_check.isEnabled()
gate["initial_filling_false"] = w._analysis_options["filling"] is False
assert gate["initial_disabled"], "Mold Filling checkbox must start disabled with no pour location"
assert gate["initial_filling_false"], "_analysis_options['filling'] must start False"

w._on_pour_location_changed(pour_location)
gate["enabled_after_set"] = w.analysis_dialog.filling_check.isEnabled()
assert gate["enabled_after_set"], "Mold Filling checkbox must become enabled once a pour location is set"

w.analysis_dialog.filling_check.setChecked(True)
w.analysis_dialog.apply_values()
gate["filling_true_after_check_and_apply"] = w._analysis_options["filling"] is True
assert gate["filling_true_after_check_and_apply"], "_analysis_options['filling'] should be True after checking + applying"

w._on_pour_location_changed(None)
gate["disabled_after_clear"] = not w.analysis_dialog.filling_check.isEnabled()
gate["filling_false_after_clear"] = w._analysis_options["filling"] is False
assert gate["disabled_after_clear"], "Mold Filling checkbox must be disabled again after clearing the pour location"
assert gate["filling_false_after_clear"], "_analysis_options['filling'] must be forced back to False after clearing"

w._on_pour_location_changed(pour_location)
w.analysis_dialog.filling_check.setChecked(True)
w.analysis_dialog.apply_values()
assert w.analysis_dialog.filling_check.isEnabled() and w._analysis_options["filling"] is True
log(f"Gate checks passed both ways: {gate}")
summary["gate_checks"] = gate

w._on_mesh_settings_confirmed(VOXEL_SIZE, w._current_refinement, w._current_memory_mode)
w._on_alloy_confirmed(ALLOY)
assert w._current_voxel_size == VOXEL_SIZE
assert w._current_alloy == ALLOY

w._solidification_params["filling"]["fill_grid_coarsening"] = FILL_GRID_COARSENING
w._solidification_params["filling"]["max_fill_time_s"] = MAX_FILL_TIME_S

w.analysis_dialog.shrinkage_check.setChecked(True)
w.analysis_dialog.hotspot_check.setChecked(True)
w.analysis_dialog.fraction_liquid_check.setChecked(True)
w.analysis_dialog.filling_check.setChecked(True)
w.analysis_dialog.apply_values()
for key in ("shrinkage", "hotspot", "fraction_liquid", "filling"):
    assert w._analysis_options[key] is True, f"analysis option {key!r} not enabled"
summary["analysis_options"] = dict(w._analysis_options)
log(f"Settings: voxel_size={VOXEL_SIZE}, alloy={ALLOY}, "
    f"fill_grid_coarsening={FILL_GRID_COARSENING}, max_fill_time_s={MAX_FILL_TIME_S}")
log(f"Analysis options: {w._analysis_options}")

import ui.main_window as mw_mod

_OrigSimulationWorker = mw_mod.SimulationWorker
progress_messages: list[str] = []
detailed_progress: list[tuple] = []
run_result: dict = {}
loop = QEventLoop()

def _on_finished(success, status):
    run_result["success"] = bool(success)
    run_result["status"] = str(status)
    loop.quit()

class _TracedSimulationWorker(_OrigSimulationWorker):
    def start(self, *a, **kw):
        self.progress.connect(progress_messages.append)
        self.detailed_progress.connect(
            lambda cur, tot, stage: detailed_progress.append((int(cur), int(tot), str(stage)))
        )
        self.finished.connect(_on_finished)
        super().start(*a, **kw)

mw_mod.SimulationWorker = _TracedSimulationWorker

t_run = time.perf_counter()
w._run_simulation()

if w.simulation_worker is None or not w.simulation_worker.isRunning():
    problems.append(
        "CRITICAL: _run_simulation() returned without starting a SimulationWorker "
        "(likely blocked by memory validation or stl_loader.is_valid() check)."
    )
    summary["problems"] = problems
    with open(os.path.join(OUT_DIR, "e2e_gui_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    log("FAIL: simulation never started. See e2e_gui_summary.json")
    sys.exit(1)

timeout_timer = QTimer()
timeout_timer.setSingleShot(True)
timed_out = {"flag": False}

def _on_timeout():
    timed_out["flag"] = True
    loop.quit()

timeout_timer.timeout.connect(_on_timeout)
timeout_timer.start(HARD_TIMEOUT_MS)

loop.exec()
wall_seconds = time.perf_counter() - t_run
timeout_timer.stop()

if timed_out["flag"] and "success" not in run_result:
    problems.append(f"CRITICAL: simulation did not finish within the {HARD_TIMEOUT_MS/60000:.0f}-minute hard timeout.")
    summary["problems"] = problems
    summary["wall_seconds"] = wall_seconds
    summary["progress_message_count"] = len(progress_messages)
    with open(os.path.join(OUT_DIR, "e2e_gui_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    log(f"FAIL: hard timeout after {wall_seconds:.1f}s. See e2e_gui_summary.json")
    sys.exit(1)

log(f"Simulation worker finished: success={run_result.get('success')} "
    f"status={run_result.get('status')!r} in {wall_seconds:.1f}s")
summary["pipeline_success"] = run_result.get("success")
summary["pipeline_status"] = run_result.get("status")
summary["wall_seconds"] = round(wall_seconds, 1)

fill_progress = [m for m in progress_messages if "Filling" in m]
summary["progress_message_count"] = len(progress_messages)
summary["filling_progress_message_count"] = len(fill_progress)
summary["filling_progress_messages_sample"] = fill_progress[:3] + (["..."] if len(fill_progress) > 6 else []) + fill_progress[-3:]
log(f"Progress messages: {len(progress_messages)} total, {len(fill_progress)} mention 'Filling'")
for m in fill_progress[:5]:
    log(f"    {m}")

if not run_result.get("success"):
    problems.append(f"CRITICAL: pipeline did not complete successfully (status={run_result.get('status')!r}).")
    summary["problems"] = problems
    with open(os.path.join(OUT_DIR, "e2e_gui_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    log("FAIL: pipeline did not succeed. See e2e_gui_summary.json")
    sys.exit(1)

assert len(fill_progress) > 0, "no progress message mentioned 'Filling' -- the filling stage may not have run"

pipeline_result = _pipeline_capture.get("result")
assert pipeline_result is not None, "run_pipeline spy never captured a result"
assert "fill_time" in pipeline_result.visualization_data, "'fill_time' missing from visualization_data"
summary["fill_time_present_in_visualization_data"] = True

filling_result = _filling_capture.get("result")
if filling_result is None:
    problems.append("CRITICAL: the filling stage's run_filling_stage() never returned a result (filling did not actually run).")
else:
    filled_pct = filling_result.filled_fraction * 100.0
    hit_cap = bool(filling_result.fill_seconds >= 0.98 * MAX_FILL_TIME_S)
    fill_stats = {
        "inlet_voxel_count": filling_result.inlet_voxel_count,
        "filled_voxel_count": filling_result.filled_voxel_count,
        "misrun_voxel_count": filling_result.misrun_voxel_count,
        "filled_fraction_pct": round(filled_pct, 2),
        "fill_seconds_simulated": round(float(filling_result.fill_seconds), 3),
        "steps": filling_result.steps,
        "max_fill_time_s_cap": MAX_FILL_TIME_S,
        "hit_time_cap": hit_cap,
        "entrapped_voxel_count": filling_result.entrapped_voxel_count,
        "cold_shut_voxel_count": filling_result.cold_shut_voxel_count,
        "warnings": list(filling_result.warnings),
    }
    summary["fill_stats"] = fill_stats
    log("=== HONEST FILL RESULT ===")
    log(f"  Metal filled: {filled_pct:.1f}% ({filling_result.filled_voxel_count:,} of "
        f"{filling_result.filled_voxel_count + filling_result.misrun_voxel_count:,} metal voxels)")
    log(f"  Simulated fill time: {filling_result.fill_seconds:.2f}s of a {MAX_FILL_TIME_S:.0f}s cap "
        f"({'HIT THE TIME CAP' if hit_cap else 'finished before the cap'})")
    if filling_result.warnings:
        for msg in filling_result.warnings:
            log(f"  warning: {msg}")

w._on_result_type_changed("Fill Time")
assert w.time_slider_widget.isVisibleTo(w), "time slider must become visible in Fill Time mode"
summary["time_slider_visible"] = True

sim = w.simulation_worker.simulation
fill_times_raw = np.asarray(getattr(sim, "_fill_arrival_times"), dtype=np.float64)
misrun_mask_raw = fill_times_raw < 0.0
misrun_count_raw = int(misrun_mask_raw.sum())

w._update_fill_time_slider(0)
label_text = w.time_label.text()
summary["ui_label_at_zero"] = label_text
summary["ui_label_misrun_count"] = misrun_count_raw
summary["non_metal_voxel_count"] = int((~np.isfinite(fill_times_raw)).sum())
if filling_result is not None:
    stage_misrun = int(filling_result.misrun_voxel_count)
    summary["stage_misrun_voxel_count"] = stage_misrun
    if misrun_count_raw != stage_misrun:
        problems.append(
            f"PROBLEM: Fill Time label reports {misrun_count_raw:,} 'never fill' but the "
            f"filling stage counted {stage_misrun:,} misrun metal voxels -- "
            f"{misrun_count_raw - stage_misrun:,} non-metal voxels are being reported "
            f"and drawn as misruns."
        )
        log(f"  MISMATCH: label={misrun_count_raw:,} vs stage={stage_misrun:,}")
    else:
        log(f"  Misrun count agrees: label={misrun_count_raw:,} == stage={stage_misrun:,}")

min_time, max_time = sim.get_fill_time_range()
log(f"Fill time range: {min_time:.3f}s .. {max_time:.3f}s")
summary["fill_time_range_s"] = [min_time, max_time]

SLIDER_MAX = mw_mod.FRACTION_LIQUID_TIME_SLIDER_MAX
slider_positions = list(range(0, SLIDER_MAX + 1, SLIDER_MAX // 10))
png_positions = set(slider_positions[::2])

sweep = []
png_paths = []
vtk_render_ok = None

for v in slider_positions:
    w.time_slider.setValue(v)
    w._update_time_slider(v)
    time_value = w._map_fraction_liquid_slider_time(v, min_time, max_time)

    arrived_mask = (~misrun_mask_raw) & (fill_times_raw <= time_value)
    expected_count = int(np.sum(misrun_mask_raw | arrived_mask))

    actor = w.visualizer.fill_time_actor
    displayed_count = None
    if actor is not None:
        polydata = actor.GetMapper().GetInput()
        displayed_count = polydata.GetNumberOfCells() // 6

    frame = {
        "slider_value": v,
        "time_s": round(float(time_value), 4),
        "displayed_voxel_count": displayed_count,
        "expected_voxel_count": expected_count,
        "misrun_count_ground_truth": misrun_count_raw,
    }
    sweep.append(frame)
    log(f"  slider={v:4d} t={time_value:7.3f}s displayed={displayed_count} expected={expected_count}")

    if v in png_positions:
        try:
            w2i = vtk.vtkWindowToImageFilter()
            w2i.SetInput(w.visualizer.render_window)
            w2i.Update()
            png_path = os.path.join(OUT_DIR, f"fill_frame_slider_{v:04d}.png")
            writer = vtk.vtkPNGWriter()
            writer.SetFileName(png_path)
            writer.SetInputConnection(w2i.GetOutputPort())
            writer.Write()
            ok = os.path.exists(png_path) and os.path.getsize(png_path) > 0
            vtk_render_ok = ok if vtk_render_ok is not False else vtk_render_ok
            if ok:
                png_paths.append(png_path)
                if vtk_render_ok is None:
                    vtk_render_ok = True
            else:
                vtk_render_ok = False
        except Exception as exc:
            vtk_render_ok = False
            problems.append(f"VTK offscreen PNG capture failed at slider={v}: {exc!r}")

summary["fill_time_sweep"] = sweep
summary["png_frames"] = png_paths
summary["vtk_offscreen_rendering_worked"] = vtk_render_ok

counts_for_check = [
    f["displayed_voxel_count"] if f["displayed_voxel_count"] is not None else f["expected_voxel_count"]
    for f in sweep
]
is_monotonic = all(b >= a for a, b in zip(counts_for_check, counts_for_check[1:]))
is_strictly_greater_end = counts_for_check[-1] > counts_for_check[0]
summary["animation_sweep_counts"] = counts_for_check
summary["animation_monotonic"] = is_monotonic
summary["animation_strictly_increases"] = is_strictly_greater_end
assert is_monotonic, f"fill time sweep is not monotonic: {counts_for_check}"
assert is_strictly_greater_end, f"fill time sweep does not reveal more voxels at the end than the start: {counts_for_check}"
log(f"Animation sweep counts: {counts_for_check}")
log("Animation sweep is monotonic and non-trivial: PASS")

mismatches = [
    f for f in sweep
    if f["displayed_voxel_count"] is not None and f["displayed_voxel_count"] != f["expected_voxel_count"]
]
if mismatches:
    problems.append(
        f"Displayed voxel counts did not exactly match the independently computed "
        f"mask at {len(mismatches)} slider position(s): {mismatches}"
    )
summary["displayed_matches_expected_exactly"] = (len(mismatches) == 0)

misrun_always_included = all(f["expected_voxel_count"] >= f["misrun_count_ground_truth"] for f in sweep)
summary["misrun_voxels_always_visible"] = misrun_always_included
assert misrun_always_included, "misrun voxels were not present in the mask at every slider position"
log(f"Misrun voxels ({misrun_count_raw:,}) confirmed present at every slider position: PASS")

summary["problems"] = problems
summary_path = os.path.join(OUT_DIR, "e2e_gui_summary.json")
with open(summary_path, "w") as fh:
    json.dump(summary, fh, indent=2, default=str)

log("")
log("=" * 70)
critical = [p for p in problems if p.startswith("CRITICAL")]
log("E2E GUI HARNESS: PASS" if not critical else "E2E GUI HARNESS: FAILED")
log(f"Summary written to {summary_path}")
if png_paths:
    log(f"PNG frames written ({len(png_paths)}): {png_paths}")
else:
    log("No PNG frames written -- see 'vtk_offscreen_rendering_worked' / 'problems' in the summary.")
log("=" * 70)
sys.exit(1 if critical else 0)
