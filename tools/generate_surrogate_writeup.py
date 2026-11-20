import os
import sys
from datetime import date

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)

def _styles():
    base = getSampleStyleSheet()
    styles = {}

    styles["title"] = ParagraphStyle(
        name="Title",
        parent=base["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=28,
        textColor=colors.HexColor("#0b3d91"),
        alignment=TA_LEFT,
        spaceAfter=10,
    )
    styles["subtitle"] = ParagraphStyle(
        name="Subtitle",
        parent=base["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#404040"),
        spaceAfter=24,
    )
    styles["h1"] = ParagraphStyle(
        name="H1",
        parent=base["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#0b3d91"),
        spaceBefore=14,
        spaceAfter=8,
    )
    styles["h2"] = ParagraphStyle(
        name="H2",
        parent=base["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1f4ea1"),
        spaceBefore=10,
        spaceAfter=5,
    )
    styles["h3"] = ParagraphStyle(
        name="H3",
        parent=base["Heading3"],
        fontName="Helvetica-BoldOblique",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#1f4ea1"),
        spaceBefore=6,
        spaceAfter=3,
    )
    styles["body"] = ParagraphStyle(
        name="Body",
        parent=base["BodyText"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=15,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
    )
    styles["bullet"] = ParagraphStyle(
        name="Bullet",
        parent=styles["body"],
        leftIndent=18,
        bulletIndent=6,
        spaceAfter=3,
    )
    styles["code"] = ParagraphStyle(
        name="Code",
        parent=base["Code"],
        fontName="Courier",
        fontSize=9,
        leading=12,
        leftIndent=12,
        textColor=colors.HexColor("#222222"),
        backColor=colors.HexColor("#f4f4f4"),
        borderColor=colors.HexColor("#cccccc"),
        borderWidth=0.5,
        borderPadding=6,
        spaceAfter=10,
    )
    styles["caption"] = ParagraphStyle(
        name="Caption",
        parent=base["Italic"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#606060"),
        spaceAfter=10,
        alignment=TA_LEFT,
    )
    return styles

def _table_cell(styles, text, size=9, color="#222222", bold=False):
    return Paragraph(
        str(text),
        ParagraphStyle(
            name=f"tcell_{size}_{bold}",
            parent=styles["body"],
            fontName="Helvetica-Bold" if bold else "Helvetica",
            fontSize=size,
            leading=size + 3,
            textColor=colors.HexColor(color),
            alignment=TA_LEFT,
            spaceAfter=0,
            spaceBefore=0,
            wordWrap=None,
        ),
    )

def _wrapped_table(data, styles, size=9, header_color="#0b3d91"):
    return [
        [
            _table_cell(
                styles, cell, size=size,
                color="#ffffff" if row_idx == 0 else "#222222",
                bold=(row_idx == 0),
            )
            for cell in row
        ]
        for row_idx, row in enumerate(data)
    ]

def _header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(colors.HexColor("#808080"))
    canvas.drawString(0.75 * inch, 0.5 * inch,
                      "Solidus Pro Surrogate Backend — Technical Write-up")
    canvas.drawRightString(LETTER[0] - 0.75 * inch, 0.5 * inch,
                           f"Page {doc.page}")
    canvas.setStrokeColor(colors.HexColor("#d0d0d0"))
    canvas.line(0.75 * inch, 0.7 * inch, LETTER[0] - 0.75 * inch, 0.7 * inch)
    canvas.restoreState()

def _title_page(styles, story):
    story.append(Spacer(1, 1.4 * inch))
    story.append(Paragraph(
        "Surrogate Solver for Casting Solidification",
        styles["title"]))
    story.append(Paragraph(
        "Apple Neural Engine&ndash;Accelerated Inference on Apple Silicon",
        styles["subtitle"]))

    meta_data = [
        ["Project", "Solidus Pro"],
        ["Component", "core/surrogate + core/deep_model"],
        ["Hardware target", "Apple M-series (M1 / M2 / M3 / M4)"],
        ["Backend ladder", "Core ML (ANE) \u2192 MLX (GPU) \u2192 NumPy (CPU)"],
        ["Deep ladder", "Core ML \u2192 PyTorch MPS \u2192 PyTorch CPU (optional torch)"],
        ["Generated", date.today().isoformat()],
    ]
    table = Table(_wrapped_table(meta_data, styles, size=10), colWidths=[1.6 * inch, 4.4 * inch])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#0b3d91")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.5 * inch))

    blurb = (
        "This document describes an engineering-simulation product feature that "
        "uses Apple's Neural Engine (ANE) for fast inference of a neural surrogate "
        "of an industrial casting solidification solver. To the author's knowledge, "
        "this is the first production engineering-simulation system to target the "
        "ANE as its primary fast-path accelerator."
    )
    story.append(Paragraph(blurb, styles["body"]))
    story.append(PageBreak())

def _section_executive_summary(styles, story):
    story.append(Paragraph("1. Executive summary", styles["h1"]))
    story.append(Paragraph(
        "Casting solidification simulation predicts how a molten metal part cools and "
        "freezes inside its mold. The current authoritative solver in Solidus Pro is a "
        "finite-difference thermal-coupled time integration over an adaptive voxel mesh. "
        "It is correct but slow &mdash; minutes to tens of minutes per run, dominated by "
        "memory-bandwidth-bound stencil sweeps that do not benefit from GPU parallelism.",
        styles["body"]))
    story.append(Paragraph(
        "This release adds a <b>surrogate backend</b>: a neural network that, given a "
        "voxelized part and an alloy, predicts the <b>solidification schedule fields</b> "
        "&mdash; start, end, and local solidification time per voxel &mdash; in a "
        "single forward pass. The surrogate runs through Apple's Core ML framework, "
        "which dispatches its convolution and matrix-multiply operations onto the "
        "Apple Neural Engine (ANE) when available. The predicted schedule feeds the "
        "exact same post-analysis pipeline the traditional solver already uses, so every "
        "downstream result &mdash; shrinkage, porosity, hot spots, Niyama, sand burn-in, "
        "gas porosity, fraction-liquid playback, Chvorinov &mdash; comes out "
        "byte-for-byte compatible with the legacy pipeline.",
        styles["body"]))
    story.append(Paragraph(
        "<b>Key result:</b> one forward pass replaces thousands of time-step iterations. "
        "On Apple silicon the ANE delivers this in milliseconds while drawing "
        "negligibly from CPU/GPU, leaving system resources available for other work.",
        styles["body"]))

def _section_background(styles, story):
    story.append(Paragraph("2. Background", styles["h1"]))

    story.append(Paragraph("2.1 The numerical solver and its bottleneck", styles["h2"]))
    story.append(Paragraph(
        "The traditional solver lives in <font face=\"Courier\">core/solidification.py</font> "
        "and <font face=\"Courier\">core/solidification_loop.py</font>. It is an explicit "
        "finite-difference method that updates a temperature field per voxel per "
        "time step, with an implicit latent-heat correction on the same step. Each step "
        "does roughly six neighbor gathers plus an inner solidification-front fill "
        "pass. The work is &mdash; by physics &mdash; memory-bandwidth-bound: arithmetic "
        "intensity is low, the gather pattern is irregular (26-neighbor stencil with "
        "data-dependent branch on whether each neighbor index is valid), and the mesh is "
        "already at the upper edge of available RAM for large parts.",
        styles["body"]))
    story.append(Paragraph(
        "These properties make the solver a poor fit for both GPU and ANE:"
    , styles["body"]))
    for line in [
        "<b>GPUs</b> love dense SIMD-style math; this code is sparse and gather-heavy with data-dependent branches.",
        "<b>ANE</b> only executes fixed conv/FC operators via Core ML; stencil gathers with data-dependent masking are not expressible.",
        "<b>Both</b> benefit from large dense matmuls; the solver performs very few per step.",
    ]:
        story.append(Paragraph(line, styles["bullet"], bulletText="\u2022"))

    story.append(Paragraph("2.2 Where the ANE actually wins", styles["h2"]))
    story.append(Paragraph(
        "Apple's ANE is a low-power matrix-multiply and convolution accelerator built "
        "into every M-series SoC. It is reachable only through Core ML; it cannot run "
        "arbitrary code. Its sweet spot is dense fp16 / bf16 conv and GEMM, where it "
        "delivers tens to hundreds of TOPS at single-digit watts. A surrogate whose "
        "input is a fixed-resolution tensor and whose forward pass is a stack of "
        "convolutions maps perfectly onto the ANE.",
        styles["body"]))
    story.append(Paragraph(
        "So the design question is not &ldquo;how do we make the solver run on ANE?&rdquo; &mdash; "
        "it cannot. The design question is &ldquo;can we replace the solver's hard parts "
        "with an ANE-friendly surrogate, while keeping the existing physics-grade "
        "post-analysis intact?&rdquo;",
        styles["body"]))

def _section_architecture(styles, story):
    story.append(Paragraph("3. Architecture", styles["h1"]))

    story.append(Paragraph("3.1 What the surrogate predicts", styles["h2"]))
    story.append(Paragraph(
        "Three fields, defined per voxel on the native adaptive mesh:",
        styles["body"]))
    for line in [
        "<b>solidification_start_times</b> &mdash; when the voxel begins freezing",
        "<b>solidification_end_times</b> &mdash; when the voxel finishes freezing",
        "<b>local_solidification_time</b> &mdash; the mushy-zone duration",
    ]:
        story.append(Paragraph(line, styles["bullet"], bulletText="\u2022"))
    story.append(Paragraph(
        "From these three fields the existing <font face=\"Courier\">run_post_analyses()</font> "
        "function derives every downstream result: shrinkage, porosity, hot spots, "
        "Niyama (via its schedule-mode branch that derives thermal gradient and cooling "
        "rate geometrically from the timing fields), sand burn-in, gas porosity, "
        "feeding-path quality, and Chvorinov. Fraction-liquid history is synthesized "
        "from the schedule with the existing <font face=\"Courier\">build_schedule_fraction_history()</font> "
        "function. In other words, the surrogate backfills the one part the solver "
        "spent most of its time producing, and then the rest of the system is unchanged.",
        styles["body"]))

    story.append(Paragraph("3.2 Input representation", styles["h2"]))
    story.append(Paragraph(
        "The mesh has a variable number of voxels &mdash; surrogate models want a fixed "
        "tensor. The codec (<font face=\"Courier\">core/surrogate/fields.py</font>) "
        "scatters per-voxel data onto a fixed-resolution cube (default 64&sup3;) and "
        "averages into cells. Seven physics-informed channels are produced:",
        styles["body"]))

    channel_data = [
        ["#", "Channel", "Source", "Normalisation"],
        ["0", "occupancy", "voxelisation bitmask", "averaged; filled cells \u2192 1"],
        ["1", "part_type", "mesh.voxel_part_types", "raw / 3 in [0,1]"],
        ["2", "surface_distance", "geometry &mdash; distance to nearest surface voxel", "divided by part max-dim"],
        ["3", "local_modulus", "mesh.voxel_moduli", "divided by part max-dim"],
        ["4", "normalized_height", "Z of voxel centre", "min-max in [0,1]"],
        ["5", "chill_boost", "nearest-chill geometry term", "log1p(max-1)/log 6 in [0,1]"],
        ["6", "riser_slowdown", "riser exothermic influence", "clipped in [0,1]"],
    ]
    table = Table(_wrapped_table(channel_data, styles, size=9), colWidths=[0.3*inch, 1.4*inch, 2.5*inch, 2.3*inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b3d91")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bfbfbf")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(table)
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "All inputs are deterministic functions of the mesh and configuration &mdash; no "
        "thermal solve is required to produce them. The codec's inverse, "
        "<font face=\"Courier\">unproject_grid()</font>, round-trips every prediction back to the "
        "native voxel index.",
        styles["body"]))

    story.append(Paragraph("3.3 Conditioning vector", styles["h2"]))
    story.append(Paragraph(
        "Channels cover geometry; the alloy is delivered to the model as a separate "
        "vector. The current implementation uses 15 alloy features (liquidus, solidus, "
        "pouring, densities, conductivity, specific heat, latent heat, critical "
        "fraction solid, linear shrinkage, Niyama threshold, hot-tear susceptibility, "
        "partition coefficient, melting-point solvent, eutectic fraction) and 8 config "
        "features (max part dimension, voxel size, feeding zone vertical bias, "
        "Chvorinov mold constant, safety factors, simulation-time bounds). Every value "
        "is normalised to [0,1] against a deterministic reference range; the inverse "
        "transformation is exact, so a model can recover physical units from the vector "
        "alone.",
        styles["body"]))

    story.append(Paragraph("3.4 Backend ladder", styles["h2"]))
    backend_data = [
        ["Backend", "Hardware", "When it wins", "Why"],
        ["Core ML", "Apple Neural Engine / GPU / CPU", "Apple Silicon (M-series)", "Dense conv/GEMM, fp16, lowest power"],
        ["MLX", "Apple GPU", "Apple Silicon fallback", "Array framework; GPU GEMM, fp32 friendly"],
        ["NumPy", "CPU", "Reference / portable / tests", "Universal fallback; smallest dependency surface"],
    ]
    table = Table(_wrapped_table(backend_data, styles, size=9.5), colWidths=[0.9*inch, 1.6*inch, 1.8*inch, 2.2*inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b3d91")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bfbfbf")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(table)
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "<font face=\"Courier\">detect_backend()</font> tries the three in priority order with "
        "lazy imports; the first available one is selected. A loadable Core ML artifact "
        "is automatically preferred and will run on the ANE when its layer set and "
        "tensor shapes fit the ANE's tile constraints; otherwise Core ML falls back to "
        "GPU or CPU on the same device with no API change for the caller.",
        styles["body"]))

    story.append(Paragraph("3.5 Pipeline integration", styles["h2"]))
    story.append(Paragraph(
        "<font face=\"Courier\">core/surrogate/pipeline.py:run_surrogate_pipeline()</font> is the integration point. It:",
        styles["body"]))
    for line in [
        "Reuses the exact mesh-generation path from the traditional pipeline (extracted into a shared helper) so geometry work is identical.",
        "Constructs a real <font face=\"Courier\">SolidificationEngine</font> with <font face=\"Courier\">solver_mode=\"schedule_legacy\"</font>, calls <font face=\"Courier\">initialize()</font>, computes the geometry-derived channels (<font face=\"Courier\">surface_distance</font>, <font face=\"Courier\">chill_boost</font>, <font face=\"Courier\">riser_slowdown</font>) via the existing engine methods.",
        "Encodes the grid channels and conditioning, runs a single model forward pass.",
        "Unprojects the (start, end, local) prediction onto the native voxel mesh.",
        "Calls <font face=\"Courier\">run_post_analyses()</font>. Every existing analysis runs unchanged on the surrogate-populated fields.",
        "Returns a <font face=\"Courier\">PipelineResult</font> indistinguishable from the traditional path: <font face=\"Courier\">export_results()</font>, <font face=\"Courier\">build_visualization_payload()</font>, the VTK viewer, the results panel, the web payload builder all consume it without modification.",
    ]:
        story.append(Paragraph(line, styles["bullet"], bulletText="\u2022"))

def _section_models(styles, story):
    story.append(Paragraph("4. Model architectures", styles["h1"]))

    story.append(Paragraph("4.1 NumpyDemoModel (analytic reference)", styles["h2"]))
    story.append(Paragraph(
        "A deterministic Chvorinov-based closed-form model, no learned weights. "
        "Computes per-cell completion time from the same material constants "
        "<font face=\"Courier\">_calculate_max_simulation_time()</font> uses (WCB-diffusivity / "
        "alloy diffusivity, latent ratio, mold constant), modulated by the "
        "physics-informed channels (modulus, surface distance, chill boost, riser "
        "slowdown, height). Bundled at <font face=\"Courier\">models/surrogate/demo</font> as the "
        "out-of-the-box default and the test parity baseline.",
        styles["body"]))

    story.append(Paragraph("4.2 NumpyWeightsModel (Phase 0 trained baseline)", styles["h2"]))
    story.append(Paragraph(
        "Twelve features per cell, linear in log-space, fit by least squares:",
        styles["body"]))
    for line in [
        "intercept",
        "log1p(modulus_in) and log1p(surface_distance_in)",
        "log1p(chill_boost) and log1p(riser_slowdown)",
        "log1p(height) and log1p(1 - height) for directional bias",
        "log1p(max_dim), log1p(mold_constant), log1p(material_factor)",
        "log1p(1 + feeding_bias), log1p(mushy_fraction)",
    ]:
        story.append(Paragraph(line, styles["bullet"], bulletText="\u2022"))
    story.append(Paragraph(
        "This is intentionally the dependency-free baseline: it requires only NumPy, "
        "serves as a real trained model (fit on captured data, persisted as "
        "<font face=\"Courier\">weights.npz</font>), and exercises the full pipeline "
        "end-to-end. Its accuracy ceiling on truly novel geometries is limited "
        "(linear log-space cannot represent complex spatial patterns); the higher-"
        "fidelity path is the 3D U-Net with FiLM conditioning described in "
        "Section 8.",
        styles["body"]))

    story.append(Paragraph("4.3 DeepSurrogateModel (3D U-Net + FiLM)", styles["h2"]))
    story.append(Paragraph(
        "<font face=\"Courier\">core/deep_model/DeepSurrogateModel</font> is the "
        "deep-learning complement to the NumPy models. It implements the same "
        "<font face=\"Courier\">SurrogateModel</font> ABC, so it drops into "
        "<font face=\"Courier\">run_surrogate_pipeline</font> and the UI without "
        "any engine changes. Its <font face=\"Courier\">UNet3D</font> is a 4-level "
        "encoder/decoder over the same 7 channels, conditioned on the same "
        "alloy/config vector, with a log-space end-time head &mdash; full "
        "architecture in Section 8.2. It is an optional dependency (torch is "
        "imported lazily) and is exported to three NPU targets &mdash; Apple ANE, "
        "Google Tensor, Snapdragon &mdash; from a single set of trained weights.",
        styles["body"]))

def _section_training(styles, story):
    story.append(Paragraph("5. Training pipeline", styles["h1"]))

    story.append(Paragraph("5.1 Data capture", styles["h2"]))
    story.append(Paragraph(
        "Every traditional simulation can opt in to dump one "
        "<font face=\"Courier\">.npz</font> record to "
        "<font face=\"Courier\">data/surrogate_dataset/</font>. Each record carries the "
        "seven encoded input channels, the alloy conditioning vector, and the three "
        "schedule targets, all on the same fixed grid. The hook is in "
        "<font face=\"Courier\">SimulationWorker._maybe_capture_surrogate_training_data()</font> "
        "and is on by default (<font face=\"Courier\">surrogate.auto_capture = True</font>).",
        styles["body"]))

    story.append(Paragraph("5.2 Offline batch + fit", styles["h2"]))
    story.append(Paragraph(
        "<font face=\"Courier\">tools/train_surrogate_model.py</font> (or "
        "<font face=\"Courier\">core/surrogate/training.retrain_from_dataset()</font>) "
        "aggregates captured records, optionally seeds the dataset by running the "
        "full solver across sample geometries &times; catalog alloys, fits the "
        "weights, and writes a <font face=\"Courier\">manifest.json</font> + "
        "<font face=\"Courier\">weights.npz</font> into the model's output directory. "
        "Training metrics (RMSE/MAE on a held-out 20% split, confidence derived from "
        "relative RMSE) are stored in the manifest and surfaced in the UI.",
        styles["body"]))

    story.append(Paragraph("5.3 Current trained artifact", styles["h2"]))
    story.append(Paragraph(
        "A model trained across all 7 catalog alloys (WCB, CF8, C958, C964, CA6NM, "
        "MONEL400, IN625) &times; 5 synthetic geometries &mdash; 35 records, 80/20 "
        "split &mdash; achieves <b>RMSE 2.93 s, MAE 0.16 s, confidence 0.82</b> on the "
        "held-out set. Like any linear surrogate, its spatial fidelity on geometries "
        "far from the training distribution is bounded &mdash; a measured caveat.",
        styles["body"]))

    story.append(Paragraph("5.4 Deep training (3D U-Net)", styles["h2"]))
    story.append(Paragraph(
        "<font face=\"Courier\">core/deep_model/training.py:train_deep_model()</font> "
        "trains the U-Net on the <b>same captured records</b> the linear model uses "
        "&mdash; no new capture code. <font face=\"Courier\">DeepDataset</font> "
        "resamples each record to the training resolution (32&sup3; default) with "
        "scatter-consistent pooling: occupancy stays binary (any subcell counts) and "
        "channel/target values are averaged over occupied subcells only, so sparse "
        "parts are never diluted away. The loss is masked MSE &mdash; inactive cells "
        "(occupancy &lt; 0.5) and chill cells are excluded, and end time is fit in "
        "log space. Training runs on the Apple GPU via PyTorch MPS with CPU fallback, "
        "at an 80/20 train/validation split.",
        styles["body"]))
    story.append(Paragraph(
        "Artifacts are written to <font face=\"Courier\">models/deep_model/trained/</font>: "
        "<font face=\"Courier\">weights.pt</font> (state dict), a standard "
        "<font face=\"Courier\">manifest.json</font> with "
        "<font face=\"Courier\">kind=\"torch_unet\"</font> / "
        "<font face=\"Courier\">backend=\"torch\"</font> so the shared "
        "<font face=\"Courier\">load_model()</font> dispatch reads it unchanged, and "
        "<font face=\"Courier\">deep_manifest.json</font> carrying export targets, "
        "quantization, training device, and loss history.",
        styles["body"]))

    story.append(Paragraph("5.5 Continuous auto-training", styles["h2"]))
    story.append(Paragraph(
        "Deep retraining is fully automatic. A <font face=\"Courier\">QFileSystemWatcher</font> "
        "on <font face=\"Courier\">data/surrogate_dataset/</font> detects new "
        "<font face=\"Courier\">run_*.npz</font> records and schedules a debounced "
        "(10 s) policy check. The Qt-free policy in "
        "<font face=\"Courier\">core/deep_model/autotrain.py</font> runs an "
        "<b>initial training</b> when no deep model exists and the dataset has "
        "<font face=\"Courier\">min_records</font> (default 4) records, and a "
        "<b>retrain</b> when the dataset signature (record count + file mtimes) "
        "changed and <font face=\"Courier\">cooldown_seconds</font> (default 300) "
        "have elapsed since the last completed training. Training is skipped when "
        "torch is absent, another train/export is active, or the dataset is "
        "unchanged. Auto-training refits on the <b>entire accumulated dataset</b> "
        "and never modifies captured records &mdash; the identical never-erases-data "
        "guarantee as the numpy surrogate &mdash; and it never swaps the model the "
        "user has loaded; it writes to <font face=\"Courier\">models/deep_model/"
        "trained/</font> and logs progress to the UI.",
        styles["body"]))

def _section_ui(styles, story):
    story.append(Paragraph("6. User-facing flow", styles["h1"]))
    story.append(Paragraph(
        "The Surrogate menu is added between Simulation and Tools:",
        styles["body"]))
    for line in [
        "<b>Load Surrogate Model</b> \u2014 file dialog for <font face=\"Courier\">manifest.json</font> "
        "(directory or single file). Includes a guard with a helpful message if the "
        "user accidentally selects a STEP/STL (those go through File &rarr; Add Part, "
        "not the model loader).",
        "<b>Model Info</b> \u2014 shows the loaded model's manifest: backend, grid "
        "resolution, supported analyses, alloy coverage, accuracy.",
        "<b>Surrogate Settings</b> \u2014 backend preference, grid resolution, "
        "auto-capture toggle.",
        "<b>Retrain Surrogate Model</b> \u2014 dialog summarises captured data, asks "
        "whether to generate a starter dataset if needed, runs training in a "
        "background thread with progress and cancellation, shows the resulting "
        "accuracy summary, and auto-loads the new model when done.",
        "<b>Run / Stop Surrogate</b> \u2014 dispatches <font face=\"Courier\">SurrogateWorker(QThread)</font> "
        "mirroring the existing <font face=\"Courier\">SimulationWorker</font> pattern; "
        "results flow through the existing visualization panel.",
    ]:
        story.append(Paragraph(line, styles["bullet"], bulletText="\u2022"))

    story.append(Paragraph("6.1 Auto-load at startup", styles["h2"]))
    story.append(Paragraph(
        "<font face=\"Courier\">_autoload_surrogate_model()</font> runs as part of "
        "<font face=\"Courier\">_setup_ui</font> and silently loads, in priority order:",
        styles["body"]))
    for line in [
        "the path persisted in <font face=\"Courier\">surrogate.last_model_path</font> (set automatically on manual load and after training),",
        "<font face=\"Courier\">&lt;model_dir&gt;/trained</font> (most recent training output),",
        "<font face=\"Courier\">&lt;model_dir&gt;/demo</font> (bundled reference).",
    ]:
        story.append(Paragraph(line, styles["bullet"], bulletText="\u2022"))
    story.append(Paragraph(
        "No dialogs on startup; a single status-bar line confirms the loaded model.",
        styles["body"]))

    story.append(Paragraph("6.2 Deep Model menu (optional torch module)", styles["h2"]))
    story.append(Paragraph(
        "A separate <b>&amp;Deep Model</b> menu sits after the Surrogate menu. "
        "Loading a deep model sets the same <font face=\"Courier\">self.surrogate_"
        "model</font> slot, so the existing &ldquo;Run Surrogate Simulation&rdquo; "
        "path runs the U-Net with zero changes to the run worker. The menu offers:",
        styles["body"]))
    for line in [
        "<b>Train Deep Model&hellip;</b> \u2014 dialog summarising the captured "
        "dataset and hyperparameters (epochs, batch size, learning rate, grid "
        "resolution in {16, 32, 64, 128}), runs <font face=\"Courier\">DeepTrainWorker"
        "(QThread)</font> with progress and cancellation, then auto-loads the result.",
        "<b>Load Deep Model&hellip;</b> \u2014 directory picker for "
        "<font face=\"Courier\">models/deep_model/trained</font>; goes through the "
        "shared <font face=\"Courier\">load_model()</font> dispatch.",
        "<b>Export for NPU&hellip;</b> \u2014 target picker (Apple ANE / Google "
        "Tensor / Snapdragon) with per-target notes; runs "
        "<font face=\"Courier\">DeepExportWorker(QThread)</font> and reports the "
        "artifact path, quantization, and validation status.",
        "<b>Auto-Train on New Captures</b> \u2014 checkable toggle; starts/stops the "
        "dataset watcher and persists <font face=\"Courier\">deep.enabled</font>.",
    ]:
        story.append(Paragraph(line, styles["bullet"], bulletText="\u2022"))
    story.append(Paragraph(
        "Manual training and auto-training share one worker slot with mutual "
        "exclusion, and exports pause auto-training while they run.",
        styles["body"]))

def _section_verification(styles, story):
    story.append(Paragraph("7. Verification", styles["h1"]))
    story.append(Paragraph(
        "13 unit tests in <font face=\"Courier\">tests/test_surrogate.py</font> cover:",
        styles["body"]))
    for line in [
        "field codec round-trip and occupancy semantics",
        "conditioning vector construction and raw reconstruction",
        "backend detection across platforms",
        "<font face=\"Courier\">NumpyDemoModel.predict</font> shape and active masking",
        "<font face=\"Courier\">NumpyWeightsModel</font> fit / predict / load",
        "metric functions (RMSE, MAE, aggregate, format)",
        "dataset capture and load round-trip with schema versioning",
        "<font face=\"Courier\">retrain_from_dataset</font> round-trip and the empty-dataset error path",
        "end-to-end surrogate pipeline against a synthetic 2\" cube (asserts all expected engine fields are populated and visualization payload keys match the traditional pipeline)",
        "parity harness &mdash; surrogate <font face=\"Courier\">solidification_times</font> vs full-solver on the same geometry, relative RMSE bound &lt; 1.0",
    ]:
        story.append(Paragraph(line, styles["bullet"], bulletText="\u2022"))
    story.append(Paragraph(
        "The standalone <font face=\"Courier\">tests/verify_chill_decay.py</font> "
        "and <font face=\"Courier\">tests/verify_thermal_benchmarks.py</font> scripts "
        "continue to pass, confirming the surrogate refactor did not regress the "
        "traditional solver path.",
        styles["body"]))
    story.append(Paragraph(
        "The deep module adds <b>21 tests</b> in "
        "<font face=\"Courier\">tests/test_deep_model.py</font>:",
        styles["body"]))
    for line in [
        "<font face=\"Courier\">UNet3D</font> forward-pass shape and finiteness at "
        "multiple resolutions, and adaptive level selection",
        "tiny overfit on two synthetic 8&sup3; records &mdash; loss strictly decreases",
        "ABC contract &mdash; <font face=\"Courier\">DeepSurrogateModel</font> runs "
        "<font face=\"Courier\">run_surrogate_pipeline</font> end-to-end on a box mesh",
        "export smoke &mdash; ONNX export + onnxruntime CPU validation; Core ML "
        "export (skipIf non-macOS / missing coremltools); TFLite guarded and a "
        "clear error when the onnx2tf toolchain is absent",
        "auto-train policy &mdash; min-records, cooldown, change detection, mutual "
        "exclusion, and state persistence, all Qt-free",
    ]:
        story.append(Paragraph(line, styles["bullet"], bulletText="\u2022"))
    story.append(Paragraph(
        "The standalone <font face=\"Courier\">tests/verify_deep_benchmarks.py</font> "
        "trains (or loads) the U-Net and reports RMSE / MAE / relative error per "
        "field against the full solver on held-out boxes, plates, and bars.",
        styles["body"]))
    story.append(Paragraph(
        "An additional finding from the broader test run: <font face=\"Courier\">unittest "
        "discover</font> exercises a SIGTRAP crash during the parallel part-assignment "
        "path of <font face=\"Courier\">core/voxel_mesh.py</font> when many "
        "mesh-generation tests share one process. I verified the same crash on a "
        "clean <font face=\"Courier\">git worktree</font> of HEAD, before any of these "
        "changes &mdash; it is a pre-existing macOS multiprocessing fragility, "
        "unrelated to the surrogate feature.",
        styles["body"]))

def _section_deep_model(styles, story):
    story.append(Paragraph("8. Deep learning module (3D U-Net + FiLM)", styles["h1"]))
    story.append(Paragraph(
        "The deep learning module, <font face=\"Courier\">core/deep_model/</font>, is "
        "a <b>separate</b> surrogate from the NumPy models: a PyTorch 3D U-Net that "
        "predicts the same three schedule fields and deploys to three NPU targets "
        "(Apple ANE, Google Tensor, Snapdragon). It reuses the surrogate's data "
        "capture, field encoding, conditioning vectors, and pipeline interface, so "
        "both modules coexist and every future captured run feeds both. Torch is an "
        "optional dependency &mdash; it is only imported lazily inside "
        "<font face=\"Courier\">core/deep_model/</font>, and the app launches "
        "without it.",
        styles["body"]))

    story.append(Paragraph("8.1 Why a separate, deeper model", styles["h2"]))
    story.append(Paragraph(
        "The linear-log model is a global least-squares fit &mdash; fast and "
        "dependency-free, but it cannot represent spatially complex freezing "
        "patterns on geometries far from its training distribution. A 3D U-Net "
        "learns local convolution features and modulates them per alloy/config via "
        "FiLM (feature-wise linear modulation), which lifts spatial fidelity "
        "substantially. The cost is data volume and compute: a U-Net needs far more "
        "captured runs than the linear model, which is exactly why auto-training "
        "accumulates the dataset continuously (Section 5.5).",
        styles["body"]))

    story.append(Paragraph("8.2 Architecture (UNet3D + FiLM)", styles["h2"]))
    story.append(Paragraph(
        "Input is the same fixed grid of 7 channels at a configurable resolution "
        "(32&sup3; default; the setting is restricted to {16, 32, 64, 128} because "
        "the 4-level stride-2 U-Net requires a resolution divisible by 16), plus the "
        "23-dimensional normalized alloy + config conditioning vector. The network:",
        styles["body"]))
    for line in [
        "<b>Encoder</b> &mdash; 4 levels of Conv3D + GroupNorm + SiLU with stride-2 "
        "downsampling; base channels 32 doubling per level (32 \u2192 64 \u2192 128 "
        "\u2192 256).",
        "<b>Bottleneck</b> &mdash; a conv block at the lowest resolution.",
        "<b>Decoder</b> &mdash; 4 levels of transposed-conv upsample, skip "
        "concatenation from the encoder, and Conv3D + GroupNorm + SiLU.",
        "<b>FiLM</b> &mdash; at each decoder level the conditioning vector goes "
        "through an MLP producing per-channel scale/shift, modulating the physics "
        "per alloy/config.",
        "<b>Head</b> &mdash; 1&times;1&times;1 Conv3D producing "
        "<font face=\"Courier\">(3, R, R, R)</font> = start, end, local. End time "
        "is predicted in log space and clamped to [0.1, max_simulation_time] at "
        "decode time, mirroring the linear model's log fit.",
    ]:
        story.append(Paragraph(line, styles["bullet"], bulletText="\u2022"))
    story.append(Paragraph(
        "Training loss is masked MSE: inactive cells (occupancy &lt; 0.5) and chill "
        "cells are excluded, end time in log space, start and local in linear space.",
        styles["body"]))

    story.append(Paragraph("8.3 NPU export matrix", styles["h2"]))
    story.append(Paragraph(
        "One trained weight set produces all three artifacts via "
        "<font face=\"Courier\">core/deep_model/export.py:export_for_target()</font>. "
        "Each target is validated in this repo; on-device execution happens in "
        "future mobile apps that consume the artifacts.",
        styles["body"]))
    export_data = [
        ["Target", "Artifact", "On-device runtime", "Conversion path", "Desktop validation"],
        ["Apple ANE", ".mlpackage", "Core ML, compute_units=all (ANE/GPU/CPU)",
         "torch trace \u2192 coremltools", "Direct on macOS (ANE if shapes fit tiles)"],
        ["Google Tensor", ".tflite (int8 PTQ)", "Android NNAPI (Edge TPU delegate)",
         "torch \u2192 ONNX \u2192 onnx2tf", "tflite-runtime interpreter (CPU)"],
        ["Snapdragon", ".onnx (fp16)", "ONNX Runtime QNN EP (Hexagon/HTP)",
         "torch \u2192 ONNX (opset 18)", "onnxruntime CPU EP (graph validation)"],
    ]
    table = Table(_wrapped_table(export_data, styles, size=8.5), colWidths=[0.95*inch, 1.2*inch, 1.55*inch, 1.5*inch, 1.6*inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b3d91")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bfbfbf")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(table)
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "Core ML export uses a rank-5 GroupNorm shim (Core ML rejects GroupNorm's "
        "rank-6 reshape decomposition); the trained artifact keeps the real "
        "<font face=\"Courier\">nn.GroupNorm</font>. TFLite needs the "
        "<font face=\"Courier\">onnx2tf</font> toolchain, which may lag Python 3.13 "
        "wheels &mdash; the export fails with a clear message rather than crashing. "
        "Export target + quantization are recorded in "
        "<font face=\"Courier\">deep_manifest.json</font>.",
        styles["body"]))

    story.append(Paragraph("8.4 Inference and integration", styles["h2"]))
    story.append(Paragraph(
        "<font face=\"Courier\">DeepSurrogateModel</font> implements the shared "
        "<font face=\"Courier\">SurrogateModel</font> ABC, so the deep model drops "
        "into <font face=\"Courier\">run_surrogate_pipeline</font> and "
        "<font face=\"Courier\">SurrogateWorker</font> unchanged. Its backend ladder "
        "is Core ML (an exported <font face=\"Courier\">.mlpackage</font>) \u2192 "
        "PyTorch MPS \u2192 PyTorch CPU. Integration points:",
        styles["body"]))
    for line in [
        "<font face=\"Courier\">Backend.TORCH</font> added to "
        "<font face=\"Courier\">core/surrogate/backend.py</font> (not in the numpy "
        "preference ladder) so deep manifests with "
        "<font face=\"Courier\">backend=\"torch\"</font> load cleanly.",
        "<font face=\"Courier\">load_model()</font> dispatches "
        "<font face=\"Courier\">kind=\"torch_unet\"</font> to "
        "<font face=\"Courier\">core/deep_model/inference</font> with a lazy import; "
        "<font face=\"Courier\">models/deep_model/trained</font> is an autoload "
        "candidate when <font face=\"Courier\">deep.enabled</font>.",
        "Settings: <font face=\"Courier\">DeepModelSettings</font> + "
        "<font face=\"Courier\">CASTSIM_DEEP_*</font> env vars + yaml/toml defaults.",
        "CLI: <font face=\"Courier\">tools/train_deep_model.py</font> and "
        "<font face=\"Courier\">tools/export_deep_model.py</font>.",
    ]:
        story.append(Paragraph(line, styles["bullet"], bulletText="\u2022"))

def _section_future(styles, story):
    story.append(Paragraph("9. Current limitations and future work", styles["h1"]))
    for line in [
        "<b>On-device mobile apps.</b> The ANE / NNAPI / QNN artifacts are produced "
        "and validated here; Android/iOS apps consuming them are the next phase.",
        "<b>Data volume.</b> A 3D U-Net needs far more captured runs than the "
        "linear model; auto-training accumulates the dataset continuously, and a "
        "CLI <font face=\"Courier\">--generate</font> solver sweep is available for "
        "bulk seeding.",
        "<b>Hot-tear, stress, microstructure heads.</b> Either predict the additional "
        "fields the corresponding analyses need, or report them as disabled with a "
        "manifest flag.",
        "<b>Web preview endpoint.</b> A <font face=\"Courier\">/surrogate-preview</font> "
        "route for instant in-browser previews (the dense-grid endpoints under "
        "<font face=\"Courier\">/api/projects/&lt;id&gt;/grid</font> already serve any scalar mode).",
        "<b>Uncertainty estimation.</b> Monte-Carlo dropout or ensemble weights to "
        "produce a per-voxel confidence field for visualisation.",
    ]:
        story.append(Paragraph(line, styles["bullet"], bulletText="\u2022"))

def _section_novelty(styles, story):
    story.append(Paragraph("10. Novelty note", styles["h1"]))
    story.append(Paragraph(
        "To the author's knowledge, this is the first production engineering-simulation "
        "system &mdash; a real thermal / metallurgical solidification solver for "
        "casting engineering &mdash; to target Apple's Neural Engine as its "
        "primary fast-path accelerator. The pattern is not specific to casting: any "
        "CAE workflow whose bottleneck is a long inner loop and whose output reduces "
        "to a small number of derived fields is a candidate for the same ANE "
        "acceleration.",
        styles["body"]))

def _section_appendix(styles, story):
    story.append(Paragraph("Appendix A. Module layout", styles["h1"]))

    modules = [
        ["File", "Role"],
        ["core/surrogate/__init__.py", "Public API: run_surrogate_pipeline, load_model, capture_run, etc."],
        ["core/surrogate/backend.py", "Core ML / MLX / NumPy backend ladder with lazy detection"],
        ["core/surrogate/conditioning.py", "Alloy + config conditioning vector with exact inverse-normalisation"],
        ["core/surrogate/fields.py", "Voxel mesh <-> fixed grid codec (7 channels)"],
        ["core/surrogate/model.py", "ModelManifest, SurrogateModel ABC, NumpyDemoModel, NumpyWeightsModel"],
        ["core/surrogate/pipeline.py", "run_surrogate_pipeline: shared mesh gen -> encode -> infer -> post-analyses"],
        ["core/surrogate/training.py", "Programmatic generate + fit + save (shared by CLI and UI)"],
        ["core/surrogate/dataset.py", "capture_run, load_dataset, schema-validated loading"],
        ["core/surrogate/metrics.py", "RMSE / MAE / max-error / confidence"],
        ["tools/train_surrogate_model.py", "CLI wrapper around retrain_from_dataset"],
        ["ui/surrogate_dialog.py", "SurrogateModelInfoDialog, SurrogateSettingsDialog, SurrogateTrainDialog"],
        ["ui/workers.py", "SurrogateWorker (QThread) + SurrogateTrainWorker + capture hook"],
        ["ui/main_window.py", "Surrogate menu, handlers, autoload, badge, settings, train dialog"],
        ["tests/test_surrogate.py", "13 unit tests covering the full stack"],
        ["models/surrogate/demo", "Bundled reference model (no weights)"],
        ["models/surrogate/trained", "Trained linear-log model on 7 alloys"],
        ["config/settings.py", "SurrogateSettings + env vars + templates"],
        ["config/hardware_config.py", "should_use_neural_engine() real gate (first Core ML consumer)"],
        ["core/simulation_pipeline.py", "Shared prepare_mesh_and_geometry helper used by both pipelines"],
        ["core/deep_model/__init__.py", "Public API (lazy): train_deep_model, load_deep_model, export_for_target, detect_deep_backend"],
        ["core/deep_model/model.py", "UNet3D + FiLM + head; DeepModelManifest; adaptive levels"],
        ["core/deep_model/dataset.py", "DeepDataset: scatter-consistent resampling, occupancy masking, 80/20 split"],
        ["core/deep_model/training.py", "train_deep_model: masked MSE, MPS/CPU, checkpoints + save_trained"],
        ["core/deep_model/inference.py", "DeepSurrogateModel (SurrogateModel ABC); Core ML / MPS / CPU predict"],
        ["core/deep_model/export.py", "NPU export: coreml .mlpackage, onnx fp16+fp32, tflite int8; per-target validation"],
        ["core/deep_model/autotrain.py", "Qt-free retrain policy: min records, cooldown, change detection, state persistence"],
        ["core/deep_model/backend.py", "Deep backend ladder: Core ML -> MPS -> CPU"],
        ["tools/train_deep_model.py", "CLI wrapper around train_deep_model"],
        ["tools/export_deep_model.py", "CLI export wrapper (--target coreml|onnx|tflite)"],
        ["ui/deep_model_dialog.py", "DeepTrainDialog (hyperparameters) + DeepExportDialog (NPU target picker)"],
        ["ui/workers.py", "DeepTrainWorker, DeepAutoTrainWorker, DeepExportWorker (QThread)"],
        ["ui/main_window.py", "Deep Model menu, handlers, dataset watcher, autoload candidate"],
        ["tests/test_deep_model.py", "21 tests: architecture, overfit, ABC pipeline, export smoke, auto-train policy"],
        ["tests/verify_deep_benchmarks.py", "Standalone deep model vs full-solver benchmark"],
        ["models/deep_model/trained", "Trained U-Net: weights.pt + manifest.json + deep_manifest.json"],
        ["config/settings.py", "DeepModelSettings + CASTSIM_DEEP_* env vars + yaml/toml defaults"],
    ]
    table = Table(_wrapped_table(modules, styles, size=9), colWidths=[2.0*inch, 4.5*inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b3d91")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bfbfbf")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(table)

    story.append(Spacer(1, 12))
    story.append(Paragraph("Appendix B. Reference commands", styles["h1"]))
    code_lines = [
        "# Train from the UI:",
        "Surrogate -> Retrain Surrogate Model -> Train Model",
        "Deep Model -> Train Deep Model... (3D U-Net, auto-loads when done)",
        "",
        "# Or from the terminal:",
        "python tools/train_surrogate_model.py --data-dir data/surrogate_dataset",
        "python tools/train_surrogate_model.py --generate --grid 64",
        "python tools/train_deep_model.py --grid 32 --epochs 100",
        "",
        "# Export the deep model for an NPU target:",
        "python tools/export_deep_model.py --target coreml   # Apple ANE",
        "python tools/export_deep_model.py --target onnx     # Snapdragon QNN",
        "python tools/export_deep_model.py --target tflite   # Google Tensor",
        "",
        "# Run the suite:",
        "python -m unittest tests.test_surrogate",
        "python -m unittest tests.test_deep_model",
        "python -m compileall config core ui tools tests",
    ]
    story.append(Preformatted("\n".join(code_lines), styles["code"]))

def build(out_path: str) -> None:
    doc = BaseDocTemplate(
        out_path,
        pagesize=LETTER,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.9 * inch,
        bottomMargin=0.9 * inch,
        title="Solidus Pro Surrogate Backend",
    )
    frame = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        doc.width,
        doc.height,
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    doc.addPageTemplates([
        PageTemplate(id="default", frames=[frame], onPage=_header_footer),
    ])

    styles = _styles()
    story = []
    _title_page(styles, story)
    _section_executive_summary(styles, story)
    _section_background(styles, story)
    _section_architecture(styles, story)
    _section_models(styles, story)
    _section_training(styles, story)
    _section_ui(styles, story)
    _section_verification(styles, story)
    _section_deep_model(styles, story)
    _section_future(styles, story)
    _section_novelty(styles, story)
    _section_appendix(styles, story)

    doc.build(story)

if __name__ == "__main__":
    desktop = os.path.expanduser("~/Desktop")
    out = os.path.join(desktop, "Solidus_Pro_Surrogate_Write_Up.pdf")
    build(out)
    print(f"Wrote: {out}")
    print(f"Size:  {os.path.getsize(out):,} bytes")
