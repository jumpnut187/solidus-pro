"""
Mesh-to-lattice seam for the mold-filling stage.

:mod:`physics.filling` solves on a dense ``(nx, ny, nz)`` lattice and knows
nothing about voxel meshes; :class:`~core.voxel_mesh.AdaptiveVoxelMesh` stores a
flat list of occupied voxel centres. This module owns the translation, plus the
filter geometry and drag-coefficient construction that needs both the mesh and
the material data.

Layering: ``core/`` may import ``physics/``; the reverse is forbidden and
enforced by ``tests/test_architecture_boundaries.py``. Keeping the mapping here
rather than in ``physics/`` is what lets the solver be tested on synthetic
lattices with no mesh at all.

Two verified facts drive the design
-----------------------------------
**The dense shape must be derived from the voxel centres, not from
``mesh.grid_shape``.** ``grid_shape`` is a *pre-voxelization* bounding-box
estimate, ``ceil((max - min) / dx)``, assigned before any voxel exists. Measured
disagreements: ``thick_thin_test.stl`` at dx=0.15 gives centres spanning
(21, 21, 34) against a ``grid_shape`` of (21, 21, 35), and the
``Filling_Test`` gating assembly at dx=0.3 spans (57, 33, 39) against (58, 34,
40). Sizing a lattice from ``grid_shape`` would leave a phantom empty layer on
two faces.

**The mapping ``ijk = round((centre - min) / dx)`` is bijective.** Verified on
``l_bracket``, ``block_2inch``, ``cylinder`` and ``thick_thin_test``, on both the
default and chunked voxelization paths, with ``unique(ijk) == n_voxels`` in every
case and a maximum lattice residual of 3.8e-6. That is what makes a dense array
representation viable at all.

Coarsening
----------
``fill_grid_coarsening`` is implemented by **box-downsampling the dense
part-type grid**, never by regenerating the mesh at a larger ``base_voxel_size``.
Re-meshing is not merely slower: ``AdaptiveVoxelMesh.generate`` segfaults
reproducibly at certain voxel sizes (see :data:`COARSENING_NOTE`), so a stage
that re-meshes would crash on geometry that the fine mesh handled.

Coarse cells resolve by **priority, never by majority vote**. A filter is
typically one or two cells thick, so a majority rule would vote it out of
existence and silently remove the restriction the user placed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, cast

import numpy as np

from core.part_manager import (
    PART_TYPE_CAST,
    PART_TYPE_CHILL,
    PART_TYPE_CORE,
    PART_TYPE_FILTER,
    PART_TYPE_GATE,
    PART_TYPE_POUR_CUP,
    PART_TYPE_RISER,
    PART_TYPE_RISER_POST,
    PART_TYPE_RUNNER,
    PART_TYPE_SPRUE,
    is_metal,
)
from core.filter_database import (
    POUR_MODE_BOTTOM,
    POUR_MODE_LIP,
    flow_rate_kg_s_for_area,
    steel_class_for_alloy,
)
from physics.filling import PorousZone
from physics.fluid_flow import (
    CS2 as LATTICE_CS2,
    SMAGORINSKY_CS,
    ceramic_foam_permeability,
    forchheimer_inverse_length,
    porous_drag_coefficients,
    capillary_entry_density_offset,
    capillary_entry_pressure,
)
from physics.units import inches_to_meters

#: Why coarsening downsamples rather than re-meshes. Kept as a named constant
#: so the reason travels with the code rather than living only in a commit.
COARSENING_NOTE = (
    "AdaptiveVoxelMesh.generate() segfaults at certain voxel sizes, so the "
    "fill grid is produced by downsampling the fine dense grid rather than by "
    "regenerating the mesh at a larger base_voxel_size."
)

#: Sentinel for a dense cell that lies outside the meshed geometry.
EMPTY_CELL = np.int8(-1)

#: Resolution order when several fine part types fall into one coarse cell.
#:
#: Highest priority first. The filter leads deliberately: it is often one or
#: two cells thick, and any majority or averaging rule would erase it. After
#: that, mold materials outrank metal (a chill that vanishes changes the
#: thermal answer), and within metal the flow path outranks the casting so the
#: gating system stays connected at coarse resolution.
COARSENING_PRIORITY: Tuple[int, ...] = (
    PART_TYPE_FILTER,
    PART_TYPE_CHILL,
    PART_TYPE_CORE,
    PART_TYPE_GATE,
    PART_TYPE_RUNNER,
    PART_TYPE_SPRUE,
    PART_TYPE_POUR_CUP,
    PART_TYPE_RISER_POST,
    PART_TYPE_RISER,
    PART_TYPE_CAST,
)

#: Frames captured across a fill for playback. The solver overwrites its whole
#: state every step, so without this the only surviving record of the pour is
#: one write-once arrival time per cell -- enough to answer "when did this fill?"
#: but not to show metal moving. 60 frames is smooth playback at a size that
#: stays trivial next to the mesh itself (two float32 per voxel per frame:
#: ~16 MB at 33k voxels).
FILL_ANIMATION_FRAMES = 60

#: How a filling run ended. See ``FillingResult.status``.
STATUS_COMPLETE = "complete"
STATUS_TIME_CAPPED = "time_capped"
STATUS_STALLED = "stalled"
STATUS_DIVERGED = "diverged"

#: Cell thickness required at the narrowest gating section before the fill can
#: be believed. Halfway bounce-back puts the walls at +-0.5 cells, so a channel
#: one cell thick has no interior at all and seals shut; two is the bare
#: minimum that admits any flow, and the field notes ask for three to four.
#:
#: This is a *thickness*, not a cross-sectional area. The first version of this
#: check compared an area against 4 and so passed a wide, thin gate -- exactly
#: the geometry that seals -- while looking like it had checked something.
#:
#: **Calibrated at Lambda = 3/16, which the solver no longer runs at.** The
#: +-0.5-cell wall position this threshold's whole argument rests on holds only
#: at the exact magic parameter, and
#: :data:`~physics.fluid_flow.TAU_MINUS_MAX` gave that up to damp an
#: undamped momentum mode. Solved exactly for steady Stokes-TRT Poiseuille,
#: ``(H_eff/2)^2 = (H/2)^2 + (4/3)(Lambda - 3/16)``, so the clamp costs a flat
#: 0.25 cell^2 of flow-carrying width at any channel width -- **-25 % of the
#: flow through a 2-cell channel** and -11 % through a 3-cell one, against
#: exactly 0 before. The effective viscosity is unchanged; the entire cost is at
#: the wall.
#:
#: The threshold is deliberately left at 2.0 rather than raised. What changed is
#: how much flow a marginal channel carries, not where the channel stops having
#: an interior, and that is what this number is about: at one cell across,
#: halfway bounce-back seals the channel outright whatever Lambda is. The 25 %
#: is why the warning text says a channel at this thickness cannot be trusted
#: rather than merely apologising for it.
MIN_CHOKE_THICKNESS_CELLS = 2.0

#: Thickness at which a channel is resolved well enough to stop apologising for.
#:
#: Three is the observed floor, not a round number. ``Test_Model/Filling_Test``
#: on a 0.20 in fill grid puts three cells across its 0.5 in gate, and it is the
#: thinnest channel this solver has ever carried a complete fill through: it
#: now completes 4,357/4,357 cells at 18.0 s, 97 % by mass -- but its peak
#: lattice velocity reaches 0.446, about nine times the 0.05 the timestep was
#: sized for, and the highest of the four validated assemblies. Complete is not
#: the same as trustworthy at three cells.
#:
#: That run used to time-cap at 57 % of cells, and this docstring and the
#: warning text both cited the figure. It was not the gate: a 20-column periodic
#: wrap put the runner floor opposite the sprue mouth, so the pour boundary was
#: being fed back into itself, and the fill completed the first time it ran with
#: a mold pad in place. Measurements taken through a torus are not measurements
#: -- see :data:`MOLD_PAD_CELLS` and ``docs/filling_field_notes.md``.
#:
#: **And it is now optimistic by 11 %.** These thresholds, and the complete
#: fills that justify them, were all measured at ``Lambda = 3/16``.
#: :data:`~physics.fluid_flow.TAU_MINUS_MAX` costs a flat 0.25 cell^2 of
#: flow-carrying width -- ``(H_eff/2)^2 = (H/2)^2 + (4/3)(Lambda - 3/16)``,
#: solved exactly -- which on a 3-cell channel is **-11.1 % of the flow for the
#: same head**. So a gate that measures 3.0 cells now passes about a ninth less
#: metal than the runs behind this number did.
#:
#: The value stays at 3.0 anyway, for two reasons. It is a *resolution*
#: threshold, and the resolution has not changed -- a 3-cell channel still has
#: exactly one interior node. And the warning it fires already tells the user to
#: treat the flow through such a channel as marginal rather than as a result,
#: which is the correct response to an 11 % boundary bias as much as to a
#: coarse-grid one. Raising it would refuse geometries the solver does complete.
#:
#: Above the hard minimum, below this, the fill is marginal rather than wrong --
#: worth saying, in a softer voice.
MARGINAL_CHOKE_THICKNESS_CELLS = 3.0

#: Parts the metal has to flow *through* on its way to the casting.
#:
#: The filter is included, unlike in :func:`_min_flow_slice_cells`. Thickness is
#: measured on the connected channel, and a filter sitting between the runner
#: and the gate is part of that channel -- leaving it out would put a wall
#: across the flow path and report the runner and gate as pinched at a junction
#: that is actually open. It is also a body worth measuring in its own right: a
#: filter thinner than the cells that carry the metal across it is not a
#: restriction the solver can represent.
_METAL_PATH_CODES: Tuple[int, ...] = (
    PART_TYPE_SPRUE,
    PART_TYPE_RUNNER,
    PART_TYPE_FILTER,
    PART_TYPE_GATE,
)

#: How the metal-path parts are named in warnings, in flow order.
_METAL_PATH_LABELS: Dict[int, str] = {
    PART_TYPE_SPRUE: "sprue",
    PART_TYPE_RUNNER: "runner",
    PART_TYPE_FILTER: "filter",
    PART_TYPE_GATE: "gate",
}

#: Largest Smagorinsky constant the stage will accept from settings.
#:
#: The eddy viscosity goes as Cs^2, so the tolerated band is not a matter of
#: taste: at 0.5 the subgrid viscosity is already ten times what the literature
#: values (0.1-0.2, 0.16 for homogeneous isotropic turbulence) produce, and
#: past that the closure stops behaving like a subgrid model and starts
#: smearing the fill front into a diffusion problem -- while still returning a
#: perfectly stable, perfectly wrong pour. Zero is legal and means exactly what
#: it says: tau_eff collapses to tau and the collision is plain TRT.
MAX_SMAGORINSKY_CS = 0.5

#: Total wet mass, as a multiple of the wet-cell count, above which the run is
#: declared diverged. Every fluid cell holds mass ~= rho * porosity ~= 1 and
#: every interface cell less, so the physical ceiling is the wet-cell count
#: itself; 1.5x is margin, not a threshold to tune. Crossing it does not mean
#: "slightly inaccurate", it means the free-surface bookkeeping has broken.
DIVERGENCE_MASS_FACTOR = 1.5

#: Steps between progress-bar updates. Frequent enough that the bar moves
#: smoothly on a run that takes minutes -- the text reports are twenty times
#: sparser, which is right for a log and far too slow for a progress bar. A
#: fill used to drive no bar at all, so the whole stage looked like a hang.
PROGRESS_UPDATE_INTERVAL = 200

#: Steps between mass-conservation checks. Cheap next to a lattice step.
DIVERGENCE_CHECK_INTERVAL = 200

#: Consecutive progress reports with no advance -- in either measure -- before
#: the run is called stalled. The front genuinely pauses while a cavity backs
#: up, so a single flat report proves nothing.
STALL_REPORTS = 5

#: Blend weight for the inlet's driven velocity, ``u = (1-b) u_prev + b u_new``.
#:
#: A numerical filter on the boundary's response, not a physical term. The
#: head inlet sets its entry velocity from the density difference across the
#: mouth every step, and that density is read off a handful of cells whose
#: population noise would otherwise feed straight back into what the boundary
#: writes -- through a square root, whose slope near zero is unbounded.
#: Measured on a 2x2x40 column, unfiltered: peak |u| in the two cells below
#: the inlet reached 0.183 against a 0.05 Mach guard, swinging 0.05 <-> 0.18
#: report to report and still rising at the end of the run. At b = 0.2 the
#: same run peaks at 0.077 and fills in 2,060 steps; at b = 0.05 it peaks at
#: 0.053 but takes 3,296. It damps the boundary's response, it does not bias
#: it: the filter has unit gain, so a steady back-pressure settles on exactly
#: the velocity that back-pressure calls for.
INLET_VELOCITY_BETA = 0.2

#: Extra fraction of the metal's full mass that must arrive between reports for
#: the fill to count as advancing.
#:
#: The filled-cell high-water mark alone is not enough. A cell only flips to
#: FLUID at the top of the eps hysteresis band, so a front creeping through
#: interface cells takes on mass every report while flipping no flags and reads
#: as stopped. Measured on Test_Model/Filling_Test: 428 casting cells held at
#: eps ~ 0.5, advancing, with the cell count flat. A stall now requires both
#: measures to be flat.
STALL_MASS_FRACTION_EPSILON = 0.001

#: Discharge coefficient for metal leaving a pouring basin into the sprue.
#:
#: Torricelli's ``sqrt(2 g h)`` is the ideal efflux velocity of an inviscid jet
#: from a sharp-edged orifice; the real one is lower, because the stream
#: contracts (vena contracta) and loses energy at the entrance. Foundry gating
#: practice puts the combined coefficient for a basin-and-sprue entrance at
#: 0.7-0.9 depending on how well the sprue is radiused, and Campbell's
#: *Castings* uses the same band. 0.8 is the middle of it, and it is a
#: coefficient on the *target*: the solver still meters delivery against
#: back-pressure, so getting this slightly wrong shifts how hard the basin
#: pushes, not whether the fill is conservative.
INLET_DISCHARGE_COEFFICIENT = 0.8

#: How far below the inlet mouth metered delivery may place metal, in cells.
#:
#: The pressure pin holds the inlet cells themselves full; this is the depth of
#: the *throat* just below them that the basin is allowed to top up when the
#: pin alone under-delivers. It has to be more than one cell -- a single layer
#: fills and converts to FLUID within a step or two of the front leaving it,
#: which throttles delivery to zero while the sprue below is still empty -- and
#: it must stay shallow, because metal placed here arrives without having
#: fallen. Three cells is one lattice-step of travel at the Mach guard plus
#: margin, and on both test rigs the metered top-up is exhausted in the first
#: layer or two whenever the sprue is running.
INLET_DELIVERY_DEPTH_CELLS = 3

#: Integrator gain on the head inlet's delivery-rate controller.
#:
#: Per step, the imposed entry velocity moves by ``gain x relative rate error
#: x u_target``, clipped to what the head can drive. The scale factor makes
#: the gain dimensionless: at a 100 % shortfall the velocity walks a fiftieth
#: of the ideal entry velocity per step, so it crosses the two-to-three times
#: the ideal that the lid's reflection actually needs inside about 150 steps
#: -- fast against the tens of thousands a pour takes, slow against the ~5
#: steps the low-pass on the back-pressure measurement settles in, so the loop
#: cannot chase its own filter.
INLET_RATE_GAIN = 0.02

#: Checks kept in the negative-population precursor window, and the share of
#: them that must show negatives before the run is aborted.
#:
#: A negative population is not survivable in the long run -- see the measured
#: cascade in ``free_surface_kernel`` -- but an *isolated* one is routine and
#: self-heals, so the signal is prevalence, not presence. Calibrated against
#: two instrumented runs, counting wet cells holding at least one negative
#: population at every :data:`DIVERGENCE_CHECK_INTERVAL`:
#:
#:   * ``Full_Gating_Test`` at shipped defaults, a healthy 43,323-step pour
#:     that completes: **zero** cells with a negative population at every one
#:     of 86 samples across the whole run. Peak prevalence 0.000.
#:   * ``Test 2`` at coarsening 2, which dies non-finite at step 63,708:
#:     negatives first appear at step 11,500 and are then present in 125 of
#:     211 samples (59 %), in bursts of 1-20 checks separated by clean runs.
#:
#: So the healthy run sits at zero and the sick one at 59 %; the threshold is
#: put at 50 % of a 40-check (8,000-step) window, which is unreachable by
#: intermittent self-healing negatives and fires on ``Test 2`` at step 35,750
#: -- 27,958 steps, 44 % of the run, before the field goes non-finite.
NEGATIVE_POPULATION_WINDOW = 40
NEGATIVE_POPULATION_PREVALENCE = 0.5

#: Immediate-abort share of wet cells holding a negative population.
#:
#: The prevalence window above needs 8,000 steps of history, which a fast
#: cascade does not grant: ``Test 2`` went from 5 affected cells to 3,681 --
#: 93 % of its wet cells -- inside a single 200-step interval. This catches
#: that shape. It is deliberately far above anything either instrumented run
#: showed while still integrating (the sick run's slow phase peaked at 20
#: cells, 0.6 % of wet), so it can only fire on a collapse in progress.
NEGATIVE_POPULATION_CASCADE_FRACTION = 0.02

#: Cells that must be affected before the cascade fraction is consulted at all,
#: so a handful of wet cells early in a pour cannot trip it on rounding.
NEGATIVE_POPULATION_CASCADE_MIN_CELLS = 8

#: Solve cells a metal-path body needs before it may drive the grid decision.
#:
#: A body of a handful of cells reads a thickness of **1.0 at every coarsening**
#: -- measured: a single cell, a 1-cell plate and a 2-cell plate all come back
#: 1.0 -- so it trips the hard minimum, forces the refinement loop down to
#: coarsening 1, and is *still* 1.0 there. The run then proceeds at 8x the cells
#: and a proportionally shorter timestep under a warning that no setting can
#: clear, for a voxelisation fragment. Before bodies were separated the pooled
#: median diluted it; now it would dictate the grid for the whole assembly.
#:
#: 8 = a 2x2x2 block: below that a body cannot contain a cube of cells in any
#: orientation, so it has no cross-section to carry flow through and is a
#: meshing artefact, not a channel. The bound is deliberately this low rather
#: than the 27 of a 3x3x3 block, because real thin gates land between the two --
#: a 3x3x1 plate is 9 cells and a 5x5x1 plate is 25, and both are genuine sealed
#: cross-sections that *should* force a refinement. Fragments are reported (in
#: ``choke_thickness_bodies_by_part`` and in a warning of their own) rather than
#: hidden; they simply do not vote on the grid.
MIN_CHOKE_BODY_CELLS = 8

#: Face-only (6-neighbour) connectivity, for labelling metal-path bodies and
#: comparing the metal path's connectivity across a coarsening.
#:
#: Two cells that touch only along an edge or at a corner share zero boundary
#: area in the 3-D geometry the lattice discretises, so no flow crosses there --
#: the same argument ``physics.filling.find_entrapped_gas`` makes for gas
#: pockets, and the same connectivity ``build_filter_zone_specs`` uses for
#: filter inserts. Built once at import; ``ndimage.generate_binary_structure``
#: is not used because this module must stay importable without scipy at module
#: scope.
_SIX_CONNECTED: np.ndarray = np.zeros((3, 3, 3), dtype=bool)
_SIX_CONNECTED[1, 1, :] = True
_SIX_CONNECTED[1, :, 1] = True
_SIX_CONNECTED[:, 1, 1] = True

#: Consecutive divergence checks the cascade condition must hold before the run
#: is aborted.
#:
#: The floor of 8 cells was chosen against models with tens of thousands of wet
#: cells, where 8 is negligible. On a *small* model it is not: at 400 wet cells
#: the 2 % fraction is 8 cells too, so both halves of the condition are met by
#: 8 cells -- and the first divergence check lands at step 200, when a splat rim
#: routinely carries that many transient negatives. One sample was enough to
#: abort a run that was about to be fine. Nothing in the calibration justified
#: acting on one sample; the two instrumented runs it came from were read over
#: hundreds of them.
#:
#: Requiring two consecutive checks, the same shape as
#: :data:`FLUID_FILL_FRACTION_CHECKS`, rather than a warm-up window. A warm-up
#: would blind the guard for its first N steps, and a genuine cascade in the
#: first few hundred steps is exactly as fatal as a late one -- the run should
#: still die, 200 steps later, rather than not at all. Two checks costs 200
#: steps of lead and buys immunity to a single transient.
#:
#: It keeps the recorded event: the round-2 ``Test 2`` cascade went from 5 to
#: **3,681** negative cells -- 93 % of its wet cells -- inside one 200-step
#: interval, and fired **27,958 steps** before the field went non-finite. A
#: field that far gone does not recover in 200 steps, so the second check sees
#: it too and the abort moves from step 35,750 to 35,950 out of 27,958 steps of
#: warning. And if a cascade ever *did* clear after one sample, the prevalence
#: window is still watching -- that is what it is for.
NEGATIVE_POPULATION_CASCADE_CHECKS = 2

#: Fill fraction band a FLUID cell's mass bookkeeping must stay inside.
#:
#: ``eps = m / (rho * porosity)`` is a *geometric* quantity: a full cell holds
#: exactly its pore volume of metal, so eps outside [0, 1] is not a physical
#: state at any pressure. Back-pressure belongs in ``rho``, and does go there.
#:
#: **What this actually measures on a FLUID cell.** ``state.rho`` is the
#: *start-of-step* density (``physics/filling.py``'s ``_collide_stream_numba``)
#: while ``state.mass`` is the end-of-step value, and under Koerner Table 1 a
#: FLUID cell's mass increment is exactly its own ``drho``. So
#: ``eps ~ 1 + drho/rho`` and the band reads as "no FLUID cell changed density
#: by more than 10 % over one step". That is a narrower statement than the
#: geometric one, and it is why the abort needs
#: :data:`FLUID_FILL_FRACTION_CHECKS` rather than firing on a single check: a
#: front slamming into a closed gate is a real water hammer, and a correct run
#: is entitled to one 200-step window of it.
#:
#: The flag scans cannot see a breach. ``transition_scan_kernel``,
#: ``apply_transitions`` and ``heal_submerged_interface`` all select on
#: FLAG_INTERFACE, so a FLUID cell's mass is written by the Koerner exchange
#: and by nothing else -- once it is over full there is no corrective path, and
#: because the exchange conserves mass exactly, every unit of phantom fill is a
#: unit missing from a real cell. Measured on ``Test 2`` with the old
#: eps-weighted porous exchange: eps_max in the filter zone went 1.046 ->
#: 3.711 -> 7.76 while eps_min fell to -1.17 and then -9.5, and the run
#: surfaced as a NaN at step 16,493 -- 8,800 steps after the first breach at
#: 7,650. A FLUID cell over full is a bookkeeping failure, and reporting it
#: where it happens is the difference between naming the mechanism and naming
#: the wreckage.
#:
#: **What it does not cover, measured.** On a mass-side amplifier -- the failure
#: it was written against -- the band gives thousands of steps of warning. On a
#: momentum-side blow-up it gives almost none: validated on Test 2, an
#: exponential ``rho``/``u`` runaway at a filter inlet e-folded every 64 steps,
#: and because ``rho`` grew in the denominator at the same time as ``mass`` grew
#: in the numerator, ``eps`` sat at 1.008 until 32 steps before the field went
#: non-finite -- inside this check's own 200-step cadence, so it never fired.
#: Tightening the band does not fix that: 1.05 would have flagged that run 2,772
#: steps early, but the same assembly reached 1.064 on a healthy stretch and
#: carried on for 2,700 more steps. The missing precursor is a *velocity* one,
#: and none exists yet.
#:
#: The band is wide on purpose. Two documented, harmless excursions sit just
#: outside [0, 1]: the one-step ``drho/rho`` lag above -- measured at 1-2 % on
#: several hundred healthy FLUID cells of a quiescent Test 2 field, plateauing,
#: never growing -- and ``heal_submerged_interface``, which flags a cell FLUID
#: and then has ``_redistribute`` strand its deficit back on it, leaving it
#: *below* full. 10 % clears both by a factor of five while still catching the
#: divergence 8,800 steps early.
FLUID_FILL_FRACTION_MIN = -0.1
FLUID_FILL_FRACTION_MAX = 1.1

#: Consecutive divergence checks that must find the band breached before the
#: run is aborted.
#:
#: Presence, not prevalence -- unlike the negative-population precursor above,
#: which needed a 40-check window because an isolated negative self-heals. A
#: FLUID cell outside the band does not self-heal: nothing in the solver writes
#: a FLUID cell's mass except the exchange, so the *only* way a breach clears is
#: for the cell to be demoted to INTERFACE and capped. Two checks is therefore
#: the smallest window that distinguishes the two shapes this guard has to tell
#: apart: a one-window hydraulic transient at a sealing gate (see
#: :data:`FLUID_FILL_FRACTION_MIN` on what eps measures), against the real
#: defect, which the diagnosis measured climbing monotonically from step ~7,650
#: to a NaN at 16,493 -- 44 consecutive checks, so it still trips on the second.
FLUID_FILL_FRACTION_CHECKS = 2

#: Cells of mold laid on every face of the solve grid before it is solved.
#:
#: ``build_grid_mapping`` sizes the dense grid exactly to the voxel centres'
#: bounding box, so a feature flush with the geometry's extent lies *on* an
#: array face with nothing beyond it. Both step paths index neighbours modulo
#: the array -- Numba ``% n``, NumPy ``np.roll`` -- and both document that they
#: rely on an obstacle shell to make that wrap unreachable
#: (``physics.fluid_flow.stream_kernel``: "real domains are closed by obstacle
#: cells, so the wrap never fires"). Without a margin the shell does not exist
#: and the lattice is a torus.
#:
#: Measured on ``Test 2``: 302 metal-to-metal wrap pairs across z, of which 298
#: put a CAST cell on ``z = 0`` directly opposite a RISER cell on ``z = 93``,
#: plus 117 across x and 55 across y. At step 12,156 -- the step the front
#: first occupied ``z = 0`` -- ``apply_transitions``' promote dilation created
#: five INTERFACE cells at ``z = 93``, every wet donor a wrapped cast-floor
#: cell 92 empty planes away. The phantom pool then grew *downward* from the
#: riser top for 3,594 steps and the run went non-finite with 1,263 negative
#: riser cells. Both self-repair paths were blinded by the same wrap:
#: ``reset_detached_interface_cells`` needs no wet neighbour and the wrap gave
#: the cell nine; ``heal_submerged_interface`` needs no gas neighbour and the
#: cell had plenty of real riser gas above it.
#:
#: One cell is enough, because ``_obstacle_mask`` makes ``EMPTY_CELL`` solid
#: and every one of the ~40 wrapped accesses already handles an obstacle
#: neighbour correctly. Measured cost on Test 2: 1,713,150 -> 1,801,824 cells
#: (+5.2 %), all 339,824 metal cells preserved, all 2,247 riser-top vents
#: preserved, and zero spurious upper-riser cells over 1,200 steps of the
#: padded miniature against 140 on the unpadded one, on both step paths.
#:
#: Clamping the ~40 neighbour accesses instead would be 40 individually
#: correct edits and 40 parity re-verifications, against one change here that
#: touches no solver code at all -- which is why ``tests/test_filling_parity``
#: is still valid after this fix.
MOLD_PAD_CELLS = 1

#: Extra cells above the original ``+z`` face, opened as atmosphere wherever
#: metal reaches that face.
#:
#: :data:`MOLD_PAD_CELLS` on its own caps an open riser with bounce-back one
#: cell above its top plane. The riser still fills -- the gas phase is a void,
#: not a compressible fluid -- but it can no longer overflow, and a metered
#: over-pour would pressurise the top cell instead of spilling out of it. These
#: cells are ``EMPTY_CELL``-typed and simply not made solid, so they are GAS to
#: the solver, excluded from ``_metal_mask`` and therefore from the fill
#: fraction's numerator and denominator alike, and included in the vent term as
#: the atmosphere they represent. The outermost plane stays obstacle, so the
#: array face is still sealed and :data:`MOLD_PAD_CELLS`' guarantee holds.
#:
#: Not opened above the inlet mouth, or within one cell of it in x or y. The
#: inlet is a Dirichlet source whose stream halo is kept wet by
#: ``ensure_interface_skin`` every step, so headspace in the halo's reach would
#: grow an interface skin *above* the geometry and let metal pool outside the
#: metal mask, where the fill ledger cannot see it. The halo is an 18-direction
#: dilation, so its reach on the plane above the mouth is the mouth's full
#: 8-neighbourhood in (x, y) -- see the exclusion in ``run_filling_stage`` for
#: the counts measured on Test 2. A pour mouth is already at atmosphere by
#: definition; it needs no headspace.
HEADSPACE_CELLS = 2

#: Steps that must elapse before a stall can be declared at all. Reports are
#: spaced at ``max_steps / 20``, so on a short step budget five of them can
#: pass within a handful of steps -- before the front has left the inlet. That
#: is a run too young to judge, not a stalled one.
STALL_MIN_STEPS = 2000


@dataclass(frozen=True)
class GridMapping:
    """
    Bijection between the mesh's voxel list and a dense lattice.

    Attributes:
        shape: Dense grid shape, derived from the voxel centres.
        dx_in: Voxel pitch in inches.
        origin_in: Minimum voxel centre, inches, shape ``(3,)``.
        voxel_to_grid: ``(n_voxels, 3)`` int32 lattice index per voxel.
        grid_to_voxel: ``(nx, ny, nz)`` int32 voxel index per cell, ``-1``
            where the lattice cell is outside the meshed geometry.
    """

    shape: Tuple[int, int, int]
    dx_in: float
    origin_in: np.ndarray
    voxel_to_grid: np.ndarray
    grid_to_voxel: np.ndarray

    @property
    def n_voxels(self) -> int:
        return int(self.voxel_to_grid.shape[0])

    @property
    def occupancy(self) -> float:
        """
        Fraction of the dense box that is real geometry.

        Worth watching: a gating assembly is mostly empty bounding box. The
        ``Filling_Test`` rigging measures 0.147, well below the 0.3-0.8 the
        design doc anticipated, so the dense lattice is ~7x the occupied voxel
        count and the memory estimate must be made per dense cell.
        """
        return self.n_voxels / float(np.prod(self.shape))

    @property
    def dx_m(self) -> float:
        return float(inches_to_meters(self.dx_in))


def build_grid_mapping(
    voxel_centers: np.ndarray, voxel_size_in: float
) -> GridMapping:
    """
    Map occupied voxel centres onto a dense lattice.

    Args:
        voxel_centers: ``(n, 3)`` voxel centres in inches.
        voxel_size_in: Uniform voxel pitch in inches.

    Returns:
        A :class:`GridMapping`.

    Raises:
        ValueError: If the centres are empty, malformed, or do not sit on a
            uniform lattice of the given pitch. The last case is checked rather
            than assumed: the whole dense representation rests on it, and a
            silent failure would produce a mapping that loses voxels.
    """
    centers = np.asarray(voxel_centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError(f"voxel_centers must be (n, 3), got {centers.shape}")
    if centers.shape[0] == 0:
        raise ValueError("voxel_centers is empty")
    if voxel_size_in <= 0:
        raise ValueError(f"voxel_size_in must be positive, got {voxel_size_in}")

    origin = centers.min(axis=0)
    scaled = (centers - origin) / float(voxel_size_in)
    ijk = np.round(scaled).astype(np.int64)

    residual = float(np.abs(scaled - ijk).max())
    if residual > 1e-3:
        raise ValueError(
            f"voxel centres do not lie on a uniform lattice of pitch "
            f"{voxel_size_in}: maximum residual {residual:.3e} cells. The dense "
            f"grid representation requires a uniform mesh."
        )

    shape = tuple(int(v) for v in (ijk.max(axis=0) + 1))
    grid_to_voxel = np.full(shape, -1, dtype=np.int32)
    grid_to_voxel[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = np.arange(
        centers.shape[0], dtype=np.int32
    )

    mapped = int(np.count_nonzero(grid_to_voxel >= 0))
    if mapped != centers.shape[0]:
        raise ValueError(
            f"voxel-to-grid mapping is not bijective: {centers.shape[0]} voxels "
            f"collapsed onto {mapped} lattice cells. Two voxels share a cell, so "
            f"the mesh is not uniform."
        )

    return GridMapping(
        shape=(shape[0], shape[1], shape[2]),
        dx_in=float(voxel_size_in),
        origin_in=origin,
        voxel_to_grid=ijk.astype(np.int32),
        grid_to_voxel=grid_to_voxel,
    )


def dense_part_types(
    mapping: GridMapping, voxel_part_types: np.ndarray
) -> np.ndarray:
    """
    Scatter the mesh's per-voxel part codes onto the dense lattice.

    Cells outside the geometry take :data:`EMPTY_CELL`, which is distinct from
    ``PART_TYPE_CAST`` (0). Conflating "no geometry here" with "casting metal
    here" would fill the entire bounding box with metal.
    """
    codes = np.asarray(voxel_part_types)
    if codes.shape[0] != mapping.n_voxels:
        raise ValueError(
            f"voxel_part_types has {codes.shape[0]} entries but the mapping "
            f"covers {mapping.n_voxels} voxels"
        )
    dense = np.full(mapping.shape, EMPTY_CELL, dtype=np.int8)
    ijk = mapping.voxel_to_grid
    dense[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = codes.astype(np.int8)
    return dense


def scatter_to_grid(
    mapping: GridMapping, values: np.ndarray, fill: float = 0.0
) -> np.ndarray:
    """Scatter a per-voxel field onto the dense lattice."""
    values = np.asarray(values)
    if values.shape[0] != mapping.n_voxels:
        raise ValueError(
            f"values has {values.shape[0]} entries but the mapping covers "
            f"{mapping.n_voxels} voxels"
        )
    dense = np.full(mapping.shape, fill, dtype=np.float64)
    ijk = mapping.voxel_to_grid
    dense[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = values.astype(np.float64)
    return dense


def gather_to_voxels(mapping: GridMapping, dense: np.ndarray) -> np.ndarray:
    """
    Gather a dense field back onto the voxel list.

    The inverse of :func:`scatter_to_grid`. The result has shape
    ``(n_voxels,)`` -- which is exactly what
    ``SolidificationEngine.set_initial_temperature_field`` requires, and is the
    occupied-voxel count, not the dense cell count.
    """
    dense = np.asarray(dense)
    if tuple(dense.shape) != tuple(mapping.shape):
        raise ValueError(
            f"dense field has shape {dense.shape}, expected {mapping.shape}"
        )
    ijk = mapping.voxel_to_grid
    gathered: np.ndarray = dense[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    return gathered


def coarsen_part_types(
    dense: np.ndarray, factor: int, priority: Sequence[int] = COARSENING_PRIORITY
) -> np.ndarray:
    """
    Box-downsample a dense part-type grid.

    Each coarse cell takes the highest-priority code present among the fine
    cells it covers, so thin features survive. A coarse cell covering only
    empty fine cells stays empty.

    Args:
        dense: ``(nx, ny, nz)`` int8 part-type grid, :data:`EMPTY_CELL` outside.
        factor: Downsampling factor. 1 returns a copy.
        priority: Codes highest-priority first.

    Returns:
        The coarsened grid, shape ``ceil(shape / factor)``.

    Raises:
        ValueError: If ``factor`` is not a positive integer.
    """
    dense = np.asarray(dense, dtype=np.int8)
    if factor < 1 or int(factor) != factor:
        raise ValueError(f"factor must be a positive integer, got {factor}")
    factor = int(factor)
    if factor == 1:
        return cast("np.ndarray", dense.copy())

    nx, ny, nz = dense.shape
    pad = (
        (0, (-nx) % factor),
        (0, (-ny) % factor),
        (0, (-nz) % factor),
    )
    padded = np.pad(dense, pad, mode="constant", constant_values=EMPTY_CELL)
    cx, cy, cz = (s // factor for s in padded.shape)
    blocks = padded.reshape(cx, factor, cy, factor, cz, factor)
    blocks = blocks.transpose(0, 2, 4, 1, 3, 5).reshape(cx, cy, cz, -1)

    out = np.full((cx, cy, cz), EMPTY_CELL, dtype=np.int8)
    # Fill from lowest priority upward so higher priority overwrites.
    for code in reversed(list(priority)):
        present = np.any(blocks == code, axis=-1)
        out[present] = np.int8(code)
    return out


@dataclass(frozen=True)
class FilterZoneSpec:
    """
    One ceramic filter body, resolved to lattice cells and drag coefficients.

    Attributes:
        zone_id: Index of this filter among those found.
        part_name: Name of the originating part, for reporting.
        cell_count: Number of lattice cells the filter occupies.
        front_face_area_m2: Face area presented to the flow.
        thickness_m: Thickness along the flow.
        porosity: Void fraction used for the drag closure.
        pore_diameter_m: Mean cell diameter.
        k1_m2: Darcy permeability.
        inv_k2_per_m: Forchheimer inertial coefficient.
        cells_along_flow: Effective cell count through the thickness.
        thickness_correction: Factor keeping the integrated pressure drop
            invariant to how many cells resolve the thickness.
        entry_pressure_pa: Capillary pressure required to prime.
        priming_head_m: The same, expressed as a metallostatic head.
    """

    zone_id: int
    part_name: str
    cell_count: int
    front_face_area_m2: float
    thickness_m: float
    porosity: float
    pore_diameter_m: float
    k1_m2: float
    inv_k2_per_m: float
    cells_along_flow: float
    thickness_correction: float
    entry_pressure_pa: float
    priming_head_m: float


def filter_geometry_from_mesh(part_mesh) -> Tuple[float, float]:
    """
    Front-face area and thickness of a filter body, in metres.

    Thickness is the **smallest oriented-bounding-box extent**, which is
    correct for every catalog size: 20 < 50, 25 < 55, 25 < 75, 25 < 100,
    30 < 125 and 30 < 150 mm.

    Area is ``volume / thickness``, which is exact for a prism of *any*
    cross-section. Multiplying the two larger bbox extents instead
    over-estimates a round filter by 4/pi, i.e. 27 %, and the catalog has seven
    round sizes.

    Args:
        part_mesh: A ``trimesh.Trimesh`` in inches.

    Returns:
        ``(front_face_area_m2, thickness_m)``.

    Raises:
        ValueError: If the body has no volume.
    """
    extents = np.asarray(part_mesh.bounding_box_oriented.extents, dtype=np.float64)
    thickness_in = float(extents.min())
    volume_in3 = float(abs(part_mesh.volume))
    if thickness_in <= 0.0 or volume_in3 <= 0.0:
        raise ValueError(
            "filter body has no measurable volume; it may be an open or "
            "degenerate mesh"
        )
    area_in2 = volume_in3 / thickness_in
    thickness_m = float(inches_to_meters(thickness_in))
    area_m2 = area_in2 * (0.0254 ** 2)
    return area_m2, thickness_m


def thickness_correction(
    thickness_m: float,
    cells_along_flow: float,
    dx_m: float,
    *,
    minimum: float = 0.25,
    maximum: float = 4.0,
) -> float:
    """
    Keep the integrated pressure drop invariant to cell count.

    The solver applies drag per cell, so a filter resolved by 2 cells and the
    same filter resolved by 8 produce different total drops unless the
    coefficients are scaled. This returns the factor by which the resolved
    thickness over- or under-states the true one; divide ``k1`` by it and
    multiply ``1/k2`` by it.

    Clamped, because a filter resolved by a fraction of a cell would otherwise
    demand an unbounded correction.
    """
    resolved = max(cells_along_flow * dx_m, dx_m)
    return float(np.clip(thickness_m / resolved, minimum, maximum))


def cells_along_flow(cell_count: int, front_face_area_m2: float, dx_m: float) -> float:
    """
    Effective number of cells through a filter's thickness.

    Derived as ``volume / (area * dx)`` rather than by casting a single ray,
    so ragged voxelization averages out instead of depending on where the ray
    happens to land.
    """
    if front_face_area_m2 <= 0 or dx_m <= 0:
        raise ValueError("front_face_area_m2 and dx_m must be positive")
    volume = cell_count * dx_m ** 3
    return float(volume / (front_face_area_m2 * dx_m))


def build_filter_zone_specs(
    dense_types: np.ndarray,
    *,
    dx_m: float,
    part_meshes: Optional[Dict[int, object]] = None,
    porosity: float = 0.85,
    pore_diameter_m: float = 2.54e-3,
    ergun_prefactor: float = 2.2,
    forchheimer_c_f: float = 0.55,
    surface_tension_n_m: float = 1.8,
    contact_angle_deg: float = 120.0,
    window_fraction: float = 0.45,
    density_kg_m3: float = 7200.0,
) -> List[FilterZoneSpec]:
    """
    Find connected filter regions and derive their drag coefficients.

    Each connected component of FILTER cells becomes one zone, so an assembly
    with two filters gets two independently reported zones.

    Geometry is taken from the lattice by default: face area is the component's
    largest axis-aligned cross-section and thickness follows from the volume.
    When the originating ``trimesh`` bodies are available they give a better
    answer via :func:`filter_geometry_from_mesh`.

    Args:
        dense_types: Dense part-type grid.
        dx_m: Cell size in metres.
        part_meshes: Optional ``{zone_id: trimesh}`` for exact geometry.
        porosity: Foam void fraction.
        pore_diameter_m: Mean cell diameter.
        ergun_prefactor: Fitted Ergun constant.
        forchheimer_c_f: Ward inertial coefficient.
        surface_tension_n_m: Melt surface tension.
        contact_angle_deg: Melt-ceramic contact angle.
        window_fraction: Window-to-cell diameter ratio.
        density_kg_m3: Melt density.

    Returns:
        One spec per filter zone, empty when the assembly has no filter.
    """
    from scipy import ndimage

    mask = np.asarray(dense_types) == PART_TYPE_FILTER
    if not np.any(mask):
        return []

    labels, count = ndimage.label(mask)
    specs: List[FilterZoneSpec] = []

    k1 = ceramic_foam_permeability(
        porosity, pore_diameter_m, ergun_prefactor=ergun_prefactor
    )
    inv_k2 = forchheimer_inverse_length(k1, c_f=forchheimer_c_f)
    entry_pressure = capillary_entry_pressure(
        surface_tension_n_m, contact_angle_deg, pore_diameter_m, window_fraction
    )
    head_m = entry_pressure / (density_kg_m3 * 9.81)

    for zone_id in range(1, count + 1):
        component = labels == zone_id
        cell_count = int(np.count_nonzero(component))

        supplied = (part_meshes or {}).get(zone_id)
        if supplied is not None:
            area_m2, thickness_m = filter_geometry_from_mesh(supplied)
        else:
            area_m2, thickness_m = _lattice_filter_geometry(component, dx_m)

        along = cells_along_flow(cell_count, area_m2, dx_m)
        correction = thickness_correction(thickness_m, along, dx_m)

        specs.append(
            FilterZoneSpec(
                zone_id=zone_id,
                part_name=f"filter_{zone_id}",
                cell_count=cell_count,
                front_face_area_m2=area_m2,
                thickness_m=thickness_m,
                porosity=porosity,
                pore_diameter_m=pore_diameter_m,
                k1_m2=k1 / correction,
                inv_k2_per_m=inv_k2 * correction,
                cells_along_flow=along,
                thickness_correction=correction,
                entry_pressure_pa=entry_pressure,
                priming_head_m=head_m,
            )
        )
    return specs


def _lattice_filter_geometry(
    component: np.ndarray, dx_m: float
) -> Tuple[float, float]:
    """
    Face area and thickness of a filter region, measured from the lattice.

    The flow axis is taken as the one with the smallest extent -- a filter is a
    thin slab -- and the face area is the volume divided by that thickness,
    matching :func:`filter_geometry_from_mesh`'s convention so the two routes
    agree.
    """
    occupied = np.argwhere(component)
    extents = occupied.max(axis=0) - occupied.min(axis=0) + 1
    thickness_cells = int(extents.min())
    thickness_m = thickness_cells * dx_m
    volume_m3 = int(component.sum()) * dx_m ** 3
    return volume_m3 / thickness_m, thickness_m


def build_porous_zone(
    dense_types: np.ndarray,
    specs: Sequence[FilterZoneSpec],
    *,
    dx_m: float,
    dt_s: float,
    nu_lattice: float,
    density_kg_m3: float = 7200.0,
) -> PorousZone:
    """
    Assemble the per-cell drag arrays the solver consumes.

    Cells outside every filter keep the inert values ``(porosity 1, c0 0.5,
    c1 0)``, which make the Guo-Zhao solve reduce to ``u = v`` exactly.

    Args:
        dense_types: Dense part-type grid.
        specs: Filter zones from :func:`build_filter_zone_specs`.
        dx_m: Cell size in metres.
        dt_s: Timestep in seconds.
        nu_lattice: Kinematic viscosity in lattice units.
        density_kg_m3: Melt density, for the capillary offset.

    Returns:
        A populated :class:`~physics.filling.PorousZone`.
    """
    from scipy import ndimage

    dense_types = np.asarray(dense_types)
    shape = (
        int(dense_types.shape[0]),
        int(dense_types.shape[1]),
        int(dense_types.shape[2]),
    )
    zone = PorousZone.inert(shape)
    if not specs:
        return zone

    mask = dense_types == PART_TYPE_FILTER
    labels, _ = ndimage.label(mask)

    for spec in specs:
        cells = labels == spec.zone_id
        if not np.any(cells):
            continue
        c0, c1 = porous_drag_coefficients(
            spec.porosity,
            spec.k1_m2,
            spec.inv_k2_per_m,
            nu_lattice=nu_lattice,
            dx_m=dx_m,
        )
        offset = capillary_entry_density_offset(
            spec.entry_pressure_pa,
            density_kg_m3=density_kg_m3,
            dt_s=dt_s,
            dx_m=dx_m,
        )
        zone.porosity[cells] = np.float32(spec.porosity)
        zone.drag_c0[cells] = np.float32(c0)
        zone.drag_c1[cells] = np.float32(c1)
        zone.entry_offset[cells] = np.float32(offset)

    return zone


def _static_vent_mask(
    dense_types: np.ndarray, inlet: np.ndarray, vent_risers_enabled: bool
) -> np.ndarray:
    """
    Vent-eligible cells whose eligibility does not depend on the fill's
    progress: open riser tops (when enabled) and the inlet.

    "Open riser top" is a static geometric fact -- the cell one step above a
    RISER cell is either off the top of the domain or has no geometry at all
    (:data:`EMPTY_CELL`), which is exactly section 6.3's "vents/risers open
    to atmosphere". Whether that riser cell currently *holds* gas is not
    this function's concern -- :func:`~physics.filling.find_entrapped_gas`
    intersects the mask with the live GAS set itself, so a riser that has
    already filled simply contributes nothing further, harmlessly.

    Inlet cells are included per the design doc even though they hold FLUID
    for essentially the whole run and so rarely intersect a gas set -- the
    one moment they might (before the very first fill step) is exactly the
    moment a well-formed vent mask should already cover them.
    """
    from core.part_manager import PART_TYPE_RISER

    shape = dense_types.shape
    vent = np.zeros(shape, dtype=bool)

    if vent_risers_enabled:
        riser = dense_types == PART_TYPE_RISER
        above = np.full(shape, EMPTY_CELL, dtype=dense_types.dtype)
        above[:, :, :-1] = dense_types[:, :, 1:]
        vent |= riser & (above == EMPTY_CELL)

    if inlet.shape[0]:
        vent[inlet[:, 0], inlet[:, 1], inlet[:, 2]] = True

    return vent


def _open_air_mask(shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Cells on the one face of the meshed bounding box that is open to air.

    The top, ``+z``, and nothing else.

    This used to be all six faces, on the reading that ``build_grid_mapping``
    sizes the grid exactly to the voxel centres' bounding box (module
    docstring, "the dense shape must be derived from the voxel centres"), so a
    cell at index 0 or ``shape - 1`` along any axis is at the true edge of the
    meshed geometry -- nothing modelled beyond it and no wall asserted there, so
    a GAS cell found there was called open to atmosphere by omission. That was
    the only defensible reading while nothing was modelled beyond the edge.

    :data:`MOLD_PAD_CELLS` settled the question the other way for five of the
    six faces, and settled it correctly: the mold does not stop where the
    meshed geometry stops. The bounding box cuts through sand below the
    casting's floor and beside its sides, which is why the pad is mold rather
    than air, and why the solver bounce-backs off it. Gravity picks out the
    remaining face: ``+z`` is the plane the pour cup and the risers break
    through, and it is the only face the pad opens a headspace above.

    Calling the other five faces vents was actively wrong for the thing the
    mask feeds. A gas pocket pinned against the casting floor is the classic
    gas-porosity site, and it read as **vented** -- the detector's answer
    inverted at exactly the geometry it exists to flag.

    **Pass the unpadded shape.** ``run_filling_stage`` solves on a padded grid
    where every array face is obstacle and therefore never GAS, so this mask
    built on the padded shape would be identically empty. The caller builds it
    on the original shape and embeds it at the pad offset, which is where the
    top of the meshed geometry still is.

    Nearly, but not entirely, redundant with the headspace the pad opens above
    top-face metal, and the remainder is why it stays. The headspace covers
    every such column except the ring within one cell of the pour mouth, which
    is excluded from it so the inlet's interface skin cannot pool metal above
    the geometry (:data:`HEADSPACE_CELLS`); ``_static_vent_mask`` covers riser
    tops and the mouth itself, by type. What is left over is that ring, where it
    is neither riser nor inlet. Measured on a casting plate flush with the top
    of its own bounding box with a 4-cell sprue punched through it: of 64
    top-face cells, ``_static_vent_mask`` covers 4 and the headspace above
    covers 48, and this term is the only thing covering the remaining **12** --
    the ring of casting-top columns around the mouth. They are the top of the
    mold immediately beside an open pour cup, so air is the right answer for
    them.
    """
    open_air = np.zeros(shape, dtype=bool)
    open_air[:, :, -1] = True
    return open_air


def _entrapment_warning_if_no_vent(vent_mask: np.ndarray) -> Optional[str]:
    """
    Message to warn with when the assembled vent mask has no open cell.

    Section 6.3 of the design doc: without at least one vent the cavity
    cannot fill at all and air compresses unphysically; every gas pocket
    then reads as entrapped by construction, which is worth telling the user
    rather than letting them discover a 100%-entrapped result and assume the
    detector is broken.
    """
    if np.any(vent_mask):
        return None
    return (
        "no vent path found for air-entrapment checking (vent_risers is "
        "off, no riser is present, and no gas cell currently touches the "
        "inlet or the open top of the mold): every gas pocket will read as "
        "entrapped until a vent opens"
    )


def measure_choke_area_cm2(dense_types: np.ndarray, dx_m: float) -> float:
    """
    Smallest cross-section anywhere in the gating system, in cm^2.

    The choke is measured rather than asked for. Foseco's sizing rules are
    expressed as a ratio of filter face area to choke area, and the choke is a
    property of the geometry the user already drew -- asking them to type it in
    invites a number that disagrees with their own model.

    Scanned as the minimum per-slice count across all three axes, over the
    sprue, runner and gate. The pour cup is excluded: it is a reservoir, not a
    metering section, and its cross-section is irrelevant to the choke.

    **Measure this on the grid that is actually solved.** A cell count times a
    cell area is a property of a particular lattice, not of the CAD, and the
    two disagree by more than rounding: on the ``Test 2`` assembly the fine
    mesh gives 2.98 cm^2 and the coarsened grid the solver runs on gives
    7.46 cm^2. The stage used to report the fine-grid figure next to results
    produced on the coarse one, which invited a filter-ratio check against a
    choke the fill never had. ``FillingResult`` now carries both, labelled.

    Returns:
        Choke area in cm^2, or 0.0 when no gating metal is present.
    """
    cells = _min_flow_slice_cells(dense_types)
    if cells <= 0:
        return 0.0
    return float(cells) * (dx_m * 100.0) ** 2


def pour_weight_lb(voxel_mesh, alloy) -> float:
    """
    Total weight of metal to be poured, in pounds.

    Everything the pour has to fill: the casting, the risers and the whole
    gating system. Mold, cores and chills are excluded -- no metal goes into
    them. Measured from the voxel volumes rather than the CAD bounding boxes so
    it matches the geometry the solver actually sees.

    Returns:
        Pounds of metal, or 0.0 when the mesh carries no part types.
    """
    part_types = getattr(voxel_mesh, "voxel_part_types", None)
    volumes = getattr(voxel_mesh, "voxel_volumes", None)
    if part_types is None:
        return 0.0

    metal = is_metal(np.asarray(part_types, dtype=np.int32))
    if not np.any(metal):
        return 0.0

    if volumes is not None and len(volumes) == len(metal):
        volume_in3 = float(np.sum(np.asarray(volumes, dtype=np.float64)[metal]))
    else:
        sizes = np.asarray(voxel_mesh.voxel_sizes, dtype=np.float64)[metal]
        volume_in3 = float(np.sum(sizes ** 3))

    # density_liquid is kg/m^3; 1 in^3 = 1.6387064e-5 m^3, 1 kg = 2.20462262 lb.
    kg = float(alloy.density_liquid) * volume_in3 * 1.6387064e-5
    return kg * 2.20462262


def filter_flow_limit_kg_s(
    filter_zones: Sequence["FilterZoneSpec"],
    alloy,
    pour_mode_code: int = 0,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Mass flow the gating system's filters are rated to pass.

    A ceramic foam filter is a rated restriction, not just a drag term: run
    metal through it faster than its rating and it does not politely throttle,
    it blinds off or breaks up and sends ceramic into the casting. The pour
    rate therefore has a hard ceiling whenever a filter is in the flow path,
    and that ceiling is a property of the filter the user already placed.

    Filters are summed, not minimised: a gating system with several of them
    runs one per ingate, in parallel, so the capacities add. Filters stacked
    in series would share one rating instead -- that is not a layout this
    reads from the geometry, and it is the rarer arrangement by far.

    Args:
        filter_zones: Resolved filter bodies; empty means no restriction.
        alloy: Provides ``name``, to pick the Foseco steel table.
        pour_mode_code: 0 = lip, 1 = bottom. Bottom pouring is rated higher.

    Returns:
        ``(limit_kg_s, note)``. ``limit_kg_s`` is None when no filter is
        present or none can be rated; ``note`` explains the latter case and is
        None otherwise.
    """
    if not filter_zones:
        return None, None

    steel_class = steel_class_for_alloy(getattr(alloy, "name", "") or "")
    if steel_class is None:
        # Refusing to guess, for the same reason flow_rate_kg_s_for_area
        # refuses: substituting a steel figure for a non-ferrous alloy gives a
        # plausible-looking rating that is wrong by tens of percent.
        return None, (
            f"{getattr(alloy, 'name', 'this alloy')} is not covered by the "
            f"Foseco steel flow tables, so the filter's rated flow could not "
            f"be checked -- verify the pour rate against the filter's own "
            f"data sheet."
        )

    pour_mode = POUR_MODE_BOTTOM if int(pour_mode_code) == 1 else POUR_MODE_LIP
    total = 0.0
    for zone in filter_zones:
        area_cm2 = float(zone.front_face_area_m2) * 1.0e4
        if area_cm2 <= 0.0:
            continue
        total += flow_rate_kg_s_for_area(area_cm2, steel_class, pour_mode)
    return (total if total > 0.0 else None), None


def recommended_pour_rate_lb_s(weight_lb: float) -> float:
    """
    Pour rate from the foundry square-root rule.

    ``t = sqrt(W)`` gives the pour time in seconds for a casting of ``W``
    pounds, so the rate is ``W / t = W / sqrt(W) = sqrt(W)``. Pour time in
    seconds and pour rate in lb/s are therefore the same number, which is the
    whole convenience of the rule.

    It replaces a fixed 5 lb/s default that was independent of the casting --
    a 20 lb bracket and a 2,000 lb housing are not poured at the same rate,
    and the mesh already knows which one it is looking at.

    Returns:
        Pounds per second, or 0.0 for a non-positive weight.
    """
    if weight_lb <= 0.0:
        return 0.0
    return float(np.sqrt(weight_lb))


def _min_flow_slice_cells(dense_types: np.ndarray) -> int:
    """
    Cells in the smallest gating cross-section, on the fill grid.

    A cross-sectional *area* in cell units, which is what
    :func:`measure_choke_area_cm2` needs. It is emphatically NOT "cells
    across" -- see :func:`measure_choke_thickness_cells`.
    """
    dense_types = np.asarray(dense_types)
    flow = np.isin(dense_types, (PART_TYPE_SPRUE, PART_TYPE_RUNNER, PART_TYPE_GATE))
    if not np.any(flow):
        return 0

    best = np.inf
    for axis in (0, 1, 2):
        counts = flow.sum(axis=tuple(a for a in (0, 1, 2) if a != axis))
        occupied = counts[counts > 0]
        if occupied.size:
            best = min(best, float(occupied.min()))
    if not np.isfinite(best):
        return 0
    return int(best)


def _local_thickness_cells(region: np.ndarray) -> np.ndarray:
    """
    Channel thickness at every cell of a boolean region, in cells.

    A cell's thickness is the diameter of the **largest inscribed sphere that
    covers it**, not the distance from it to the nearest wall. The difference
    is the whole point. A distance transform reports 1.0 for every surface
    cell of every body, thick or thin, so any statistic taken over those
    numbers -- median, percentile, anything but the maximum -- says 1.0 about
    a solid block. Only the maximum carries thickness information, and the
    maximum is exactly what a fat junction hijacks.

    Computed the standard way (Hildebrand & Ruegsegger): take the distance
    transform ``d``, then for each radius ``r``, every cell within ``r`` of a
    cell whose ``d >= r`` is covered by a sphere of that radius. Sweeping
    ``r`` downward in half-cell steps and keeping the largest that reaches
    each cell gives the map. Rounding the radii down to a half cell is
    deliberate: it can only under-report thickness, and under-reporting errs
    toward warning.

    Thickness is ``2*r - 1``, not ``2*r``, for the reason the scalar measure
    has always used: ``distance_transform_edt`` measures to the nearest
    *background* cell, so a cell in a one-thick section reports 1.0 -- the gap
    to the wall, not the metal. Exact for odd thicknesses, one cell
    conservative for even ones.

    Args:
        region: Boolean mask of the connected material to measure.

    Returns:
        Float array shaped like ``region``: thickness in cells inside the
        region, 0.0 outside it.
    """
    from scipy import ndimage

    region = np.asarray(region, dtype=bool)
    out = np.zeros(region.shape, dtype=np.float64)
    if not np.any(region):
        return out

    # Cropped to the region's own bounding box, with a cell of margin. The
    # sweep below runs one distance transform per radius, so the cost is worth
    # bounding; nothing outside the box can change a result inside it.
    occupied = np.nonzero(region)
    box = tuple(
        slice(int(axis.min()), int(axis.max()) + 1) for axis in occupied
    )
    # Padded with one cell of background. Without it a channel that reaches
    # the edge of the grid has nothing to measure to on that side, and a
    # domain entirely full of metal returns the array diagonal rather than a
    # thickness. The pad is physically right as well as defensive: the lattice
    # ends in obstacle, not in more metal.
    sub = np.pad(region[box], 1, mode="constant", constant_values=False)

    distance = ndimage.distance_transform_edt(sub)
    largest = np.zeros_like(distance)
    radius = np.floor(float(distance.max()) * 2.0) / 2.0
    while radius >= 0.5:
        seeds = distance >= radius
        if np.any(seeds):
            # Distance from every cell to the nearest sphere centre of this
            # radius; a cell within `radius` of one is inside that sphere.
            reach = ndimage.distance_transform_edt(~seeds)
            np.maximum(largest, np.where(reach <= radius, radius, 0.0), out=largest)
        radius -= 0.5

    inner = largest[(slice(1, -1),) * largest.ndim]
    out[box] = np.where(region[box], 2.0 * inner - 1.0, 0.0)
    return out


def measure_choke_thickness_by_part(dense_types: np.ndarray) -> Dict[int, float]:
    """
    Robust channel thickness of each metal-path part, in fill-grid cells.

    This is the number that decides whether metal can get through: halfway
    bounce-back puts the walls at +-0.5 cells, so a channel one cell thick has
    no interior at all and seals shut regardless of how wide it is.

    Cross-sectional *area* cannot answer that question, and the first version
    of this check used area and was wrong for the case that matters most. A
    gate is characteristically wide and thin -- the one on the reference
    assembly is 4.0 x 0.5 in -- so on a coarse grid it presents a perfectly
    healthy-looking 11-cell cross-section while being one cell thick. Area
    passes; the metal does not.

    The version after that measured thickness, but as **one maximum per
    part**, and a maximum is the wrong end of the distribution. Measured on
    the ``Test 2`` assembly at its solved 8.6 mm grid: the gates voxelize to
    11 x 13 x 2 plates -- two cells thick, sealed -- yet the check reported
    3.00 cells and stayed silent, because a single fat cell where each gate
    meets the runner set the maximum for the entire plate. The thin plate is
    the part the metal has to cross; the fillet at its root is not.

    So the statistic is the **median** thickness over the part's own cells,
    taken from :func:`_local_thickness_cells`. Justified against four measured
    grids rather than chosen for elegance:

    ======================================  =====  ======  ======  ====
    grid                                    sprue  runner  filter  gate
    ======================================  =====  ======  ======  ====
    ``Test 2``, 8.6 mm (the failing case)     5      4       7      1
    ``Test 2``, 4.3 mm                       11      7      15      3
    ``Full_Gating_Test``, 0.20 in             5      7       -      5
    ``Filling_Test``, 0.20 in                 7      7       -      3
    ======================================  =====  ======  ======  ====

    The failing gate reads 1 and warns; the same gate on the finer grid reads
    3 and does not; ``Full_Gating_Test`` is silent at 5 and up;
    ``Filling_Test`` reads the 3 its field notes record for that gate.

    The *minimum* would fail two of the four. ``Full_Gating_Test``'s sprue has
    one 1-cell corner on an otherwise 5-cell channel, and ``Test 2`` at 4.3 mm
    has the same in its runner and filter -- three false alarms on grids that
    resolve their gating perfectly well. The 25th percentile agrees with the
    median on all four; it is simply nearer that noise floor for no gain.

    What the median gives up, honestly: a short pinch in an otherwise
    generous channel is voted down by the rest of the part. Persistent
    thinness is what it detects, which is the failure mode that stops a fill;
    a one-cell nick in a runner is not.

    Per **body**, not per part code
    -------------------------------
    Part codes are per *type*, not per instance: every gate in the assembly is
    ``PART_TYPE_GATE``. Pooling one median over all of them lets a healthy body
    outvote a sealed one, which is not a hypothetical -- ``Test 2`` has two GATE
    bodies and two FILTER bodies. Measured on a synthetic pair sharing the code,
    one 6 cells thick and one 2: the pooled median reads **5.0 cells** and the
    stage says nothing, while the 2-cell body is sealed and the metal cannot
    cross it. So each code is split into connected bodies first and the
    **thinnest body** is what gets reported and acted on.

    26-connectivity, so two blocks meeting only at an edge or a corner count as
    one body. That is the conservative direction here: over-splitting would
    invent thin components out of corner contacts and cry wolf, and a corner
    touch in cast metal really is one fused piece. (``build_filter_zone_specs``
    labels with the default 6-connectivity instead, correctly -- there the
    question is which *physical insert* a cell belongs to, and separate
    inserts touching at a corner are still separate inserts.)

    Returns:
        ``{part code: thinnest resolvable body's thickness in cells}`` for every
        metal-path part present. Bodies below :data:`MIN_CHOKE_BODY_CELLS` are
        skipped here -- see that constant -- unless a code has nothing else, in
        which case its thinnest fragment is reported so the code is not silently
        absent. Empty when the grid holds no gating metal.
        :func:`measure_choke_thickness_components` returns every body and
        :func:`measure_choke_body_fragments` the ones skipped.
    """
    by_part: Dict[int, float] = {}
    for code, bodies in _choke_bodies(dense_types).items():
        resolvable = [t for t, cells in bodies if cells >= MIN_CHOKE_BODY_CELLS]
        by_part[code] = min(resolvable) if resolvable else min(
            t for t, _ in bodies
        )
    return by_part


def measure_choke_body_fragments(
    dense_types: np.ndarray,
) -> Dict[int, Tuple[int, ...]]:
    """
    Cell counts of the metal-path bodies too small to judge a grid by.

    Reported rather than dropped: a fragment is a modelling or voxelisation
    observation, not a resolution one, and a reader who sees a part's thickness
    ignore a body it can see in the mesh should be told why. See
    :data:`MIN_CHOKE_BODY_CELLS`.

    Returns:
        ``{part code: (cells, ...)}`` smallest first, only for codes that have
        at least one such body *and* at least one body large enough to measure
        -- a code made entirely of fragments is measured on them instead, so
        nothing is being ignored there.
    """
    fragments: Dict[int, Tuple[int, ...]] = {}
    for code, bodies in _choke_bodies(dense_types).items():
        small = sorted(
            cells for _, cells in bodies if cells < MIN_CHOKE_BODY_CELLS
        )
        if small and len(small) < len(bodies):
            fragments[code] = tuple(small)
    return fragments


def _choke_bodies(
    dense_types: np.ndarray,
) -> Dict[int, Tuple[Tuple[float, int], ...]]:
    """
    ``{part code: ((thickness, cells), ...)}`` per connected body, thinnest
    first.

    The one measurement behind :func:`measure_choke_thickness_by_part`,
    :func:`measure_choke_thickness_components` and
    :func:`measure_choke_body_fragments`, so the three cannot disagree.

    **6-connectivity**, matching ``build_filter_zone_specs``. Two same-code
    bodies touching only at an edge or a corner are separate bodies, and that
    matters: 26-connectivity fuses them, and fusing a sealed 2-cell plate onto a
    healthy 8-cell one reports **7.0 cells and stays silent** where splitting
    them reports **1.0 and hard-warns** -- the same "healthy body outvotes a
    sealed one" failure that separating bodies exists to remove. A corner-only
    contact is one diagonal D3Q19 link, which is not a channel the solver can
    carry flow through, so fusing across it also asserts a path that is not
    there.

    Splitting costs nothing in false alarms, which is worth stating because the
    opposite was assumed: ``_local_thickness_cells`` is computed **once** over
    the whole metal path and each body only *indexes* into that map, so a body's
    pieces cannot read thinner than the body did. Measured -- a 4-cell-thick
    diagonal staircase channel reads 3.0 under both connectivities, and a
    1-cell diagonal string reads 1.0 under both (as 1 body or as 20).
    """
    from scipy import ndimage

    dense_types = np.asarray(dense_types)
    metal_path = np.isin(dense_types, _METAL_PATH_CODES)
    if not np.any(metal_path):
        return {}
    # One map over the whole connected path, so a gate cell touching the
    # runner is interior rather than boundary -- measuring each part in
    # isolation would report every junction as a constriction.
    thickness = _local_thickness_cells(metal_path)

    by_part: Dict[int, Tuple[Tuple[float, int], ...]] = {}
    for code in _METAL_PATH_CODES:
        part = dense_types == code
        if not np.any(part):
            continue
        labels, count = ndimage.label(part, structure=_SIX_CONNECTED)
        bodies = []
        for index in range(1, count + 1):
            body = labels == index
            cells = int(np.count_nonzero(body))
            if not cells:
                continue
            bodies.append(
                (max(0.0, float(np.median(thickness[body]))), cells)
            )
        if bodies:
            by_part[code] = tuple(sorted(bodies))
    return by_part


def measure_choke_thickness_components(
    dense_types: np.ndarray,
) -> Dict[int, Tuple[float, ...]]:
    """
    Thickness of every connected body of each metal-path part, thinnest first.

    The per-body detail behind :func:`measure_choke_thickness_by_part`, which
    reports only the first (thinnest) entry of each tuple. Kept separate so the
    scalar-per-code shape that :class:`FillingResult` and the pipeline payload
    already serialize does not change type -- see that function for why the
    split exists at all.

    Returns:
        ``{part code: (thinnest, ..., thickest)}`` for every metal-path part
        present, each value a median over that body's own cells. Includes the
        bodies below :data:`MIN_CHOKE_BODY_CELLS` that
        :func:`measure_choke_thickness_by_part` skips, so the reported list is
        the whole truth about the grid.
    """
    return {
        code: tuple(t for t, _ in bodies)
        for code, bodies in _choke_bodies(dense_types).items()
    }


@dataclass(frozen=True)
class MetalPathSevering:
    """
    What a coarsening did to the connectivity of the metal path.

    Attributes:
        severed: True when at least one connected run of metal path on the mesh
            is broken, or gone, on the solve grid.
        fine_components: Connected components of the metal-path mask on the
            mesh grid.
        solve_components: The same on the solve grid.
        erased: ``{part code: bodies}`` for fine bodies of that code that kept
            no metal-path cell anywhere -- total erasure. A *second* test, not
            a special case of the connectivity one; see
            :func:`metal_path_severing`.
        split: ``{part code: fine components}`` for fine metal-path components
            that came apart into two or more on the solve grid, attributed to
            the codes that lost cells at the break.
    """

    severed: bool
    fine_components: int
    solve_components: int
    erased: Dict[int, int]
    split: Dict[int, int]

    @property
    def parts(self) -> Tuple[int, ...]:
        """Every part code implicated, in flow order (sprue first)."""
        codes = set(self.erased) | set(self.split)
        return tuple(code for code in _METAL_PATH_CODES if code in codes)

    def describe(self) -> str:
        """One clause naming what happened, for a message."""
        pieces = []
        for code in _METAL_PATH_CODES:
            if code in self.erased:
                count = self.erased[code]
                pieces.append(
                    f"{count} {_METAL_PATH_LABELS.get(code, 'gating')} "
                    f"{'body' if count == 1 else 'bodies'} erased outright"
                )
        broken = tuple(code for code in _METAL_PATH_CODES if code in self.split)
        if broken:
            names = "/".join(
                _METAL_PATH_LABELS.get(code, "gating") for code in broken
            )
            pieces.append(f"the path through the {names} broken in two")
        if not pieces:
            return "the metal path is intact"
        return " and ".join(pieces)


def metal_path_severing(
    fine_types: np.ndarray, solve_types: np.ndarray, factor: int
) -> MetalPathSevering:
    """
    Whether the coarsening cut the metal path, by comparing its connectivity.

    The failure this exists for
    ---------------------------
    :data:`COARSENING_PRIORITY` puts CORE and CHILL *above* GATE and RUNNER --
    deliberately, because a chill that vanishes changes the thermal answer --
    and :func:`coarsen_part_types` gives each coarse block the single
    highest-priority code in it. So a channel one mesh cell thick against a core
    becomes CORE. Every guard downstream is keyed on parts that are *present*:
    :func:`measure_choke_thickness_by_part` skips a code with no cells, so the
    sealed-choke check sees nothing to complain about and the metal path is
    severed in silence. That is the worst shape a defect can have here -- the
    run completes and reports a misrun on a path the solve grid never had.

    Connectivity, not presence
    --------------------------
    The first version of this test asked whether each fine body kept *any*
    metal-path cell, which only ever catches **total** erasure. A body severed
    in its *middle* keeps cells at both ends and was never flagged -- and that
    is the common geometry, not a corner case: a wide thin gate plate with a
    boss or fillet somewhere along it, which
    :func:`measure_choke_thickness_by_part`'s own docstring records Test 2's
    gates having. Measured on exactly that shape: the fine metal path is **1**
    connected component and the solve grid's is **2**, the old test returned
    ``{}``, both parts measured 3.0 cells so nothing was at or below the hard
    minimum, and the only thing the user heard was the generic 3-cell marginal
    note -- the same sentence a perfectly healthy channel gets.

    So the test is the connectivity itself. Label the metal-path mask on both
    grids with the same 6-connectivity (see :data:`_SIX_CONNECTED`), map each
    fine component onto the coarse blocks its cells fall in, and count the
    distinct coarse components it lands on:

    * **>= 2** -- the component came apart. Severing.
    * **1** -- intact, however the part *codes* moved around inside it.

    Total erasure is a **second** test, not a special case
    -----------------------------------------------------
    It is tempting to fold it in -- a component that lands on *zero* coarse
    components has plainly gone -- and that is not sufficient, because a whole
    part can vanish while the surviving path stays connected. A one-cell gate
    stub hanging off a sprue column shares its fine metal-path component with
    the sprue, so the component still lands on one coarse component (the
    sprue's) and the connectivity test says intact -- while all 256 of the
    gate's cells became CORE. The path is not cut, but a part of the casting is
    simply missing from the grid, which is the same misrun-by-coarsening. So the
    per-code check is kept alongside: for each fine body of each metal-path
    code, did its blocks keep any metal-path cell at all.

    That last point is what keeps the false positive away, and it is a real one:
    losing a *code* while keeping the *path* is routine and harmless. A gate
    plate cutting through a sprue column leaves a one-cell sprue stub that
    becomes GATE at coarsening 2, because GATE outranks SPRUE -- the channel is
    still open, still connected, and still measured, since the thickness map is
    built over the whole metal path rather than per part. Keying on the body's
    own code reports the stage's own choke rig as severed.

    Args:
        fine_types: The mesh-resolution dense part-type grid.
        solve_types: The coarsened grid, as :func:`coarsen_part_types` returns.
        factor: The coarsening factor relating the two.

    Returns:
        A :class:`MetalPathSevering`. Never severed at ``factor == 1``, where
        the two grids are the same by construction.
    """
    from scipy import ndimage

    fine_types = np.asarray(fine_types)
    solve_types = np.asarray(solve_types)
    factor = max(1, int(factor))

    fine_mask = np.isin(fine_types, _METAL_PATH_CODES)
    solve_mask = np.isin(solve_types, _METAL_PATH_CODES)
    fine_labels, fine_count = ndimage.label(fine_mask, structure=_SIX_CONNECTED)
    _solve_labels, solve_count = ndimage.label(
        solve_mask, structure=_SIX_CONNECTED
    )
    intact = MetalPathSevering(
        severed=False,
        fine_components=int(fine_count),
        solve_components=int(solve_count),
        erased={},
        split={},
    )
    if factor == 1 or not fine_count:
        return intact

    shape = solve_types.shape

    def blocks_of(mask_labels: np.ndarray, index: int):
        ix, iy, iz = np.nonzero(mask_labels == index)
        if ix.size == 0:
            return None
        return (
            np.minimum(ix // factor, shape[0] - 1),
            np.minimum(iy // factor, shape[1] - 1),
            np.minimum(iz // factor, shape[2] - 1),
            (ix, iy, iz),
        )

    # (a) the connectivity comparison: did a connected run come apart?
    split: Dict[int, int] = {}
    for index in range(1, fine_count + 1):
        mapped = blocks_of(fine_labels, index)
        if mapped is None:
            continue
        cx, cy, cz, (ix, iy, iz) = mapped
        landed = _solve_labels[cx, cy, cz]
        if np.unique(landed[landed > 0]).size < 2:
            continue
        # Attribute the break to the codes that actually lost cells: a fine
        # cell whose block came back non-metal-path is where the cut is.
        lost = landed == 0
        codes = np.unique(
            fine_types[ix[lost], iy[lost], iz[lost]]
            if np.any(lost)
            else fine_types[ix, iy, iz]
        )
        for code in codes:
            if int(code) in _METAL_PATH_CODES:
                split[int(code)] = split.get(int(code), 0) + 1

    # (b) the per-code check: did a whole body of a part vanish, connectivity
    # of the survivors notwithstanding? See the docstring.
    erased: Dict[int, int] = {}
    for code in _METAL_PATH_CODES:
        part = fine_types == code
        if not np.any(part):
            continue
        part_labels, part_count = ndimage.label(part, structure=_SIX_CONNECTED)
        gone = 0
        for index in range(1, part_count + 1):
            mapped = blocks_of(part_labels, index)
            if mapped is None:
                continue
            cx, cy, cz, _ = mapped
            if not np.any(np.isin(solve_types[cx, cy, cz], _METAL_PATH_CODES)):
                gone += 1
        if gone:
            erased[code] = gone

    if not (split or erased):
        return intact
    return MetalPathSevering(
        severed=True,
        fine_components=int(fine_count),
        solve_components=int(solve_count),
        erased=erased,
        split=split,
    )


def narrowest_choke_part_code(by_part: Dict[int, float]) -> Optional[int]:
    """
    The metal-path part code with the smallest robust thickness.

    Ties break in flow order (sprue first), so the message names the part
    furthest upstream -- the one the metal meets first.
    """
    if not by_part:
        return None
    return min(
        (code for code in _METAL_PATH_CODES if code in by_part),
        key=lambda code: by_part[code],
    )


@dataclass(frozen=True)
class CoarseningDecision:
    """
    What fill grid the solve will actually run on, and why.

    Attributes:
        coarsening: The factor chosen. Equal to ``configured`` unless
            auto-resolution refined it.
        configured: The factor ``fill_grid_coarsening`` asked for.
        solve_types: The coarsened part-type grid at ``coarsening``, so the
            caller does not pay for the downsample twice.
        by_part: ``{part code: median thickness in cells}`` measured on
            ``solve_types`` -- the **thinnest body large enough to judge a grid
            by**, see :func:`measure_choke_thickness_by_part`.
        components_by_part: ``{part code: (thinnest, ..., thickest)}``, every
            connected body of each code, fragments included. So
            ``components_by_part[code][0]`` is the thinnest body outright and
            ``by_part[code]`` is the thinnest one that votes; they differ
            exactly when the code has a body below
            :data:`MIN_CHOKE_BODY_CELLS` alongside a larger one.
        fragments_by_part: ``{part code: (cells, ...)}`` for bodies too small
            to judge the grid by -- see :data:`MIN_CHOKE_BODY_CELLS`. Reported,
            not acted on.
        severing: What ``coarsening`` did to the metal path's connectivity --
            see :class:`MetalPathSevering`. ``severing.severed`` on the chosen
            grid means the path is cut there and the caller must refuse rather
            than solve it, which can only happen when auto-resolution was not
            allowed to act: every grid the loop *chooses* has an intact path,
            and at ``coarsening == 1`` the two grids are identical.
        forced_by: Part code that blocked the configured grid and forced the
            refinement, or ``None`` when nothing did.
        forced_by_cause: Why that part blocked it -- ``"thickness"`` for a
            channel at or below :data:`MIN_CHOKE_THICKNESS_CELLS`, ``"severed"``
            for a metal path the downsample cut, ``""`` when nothing blocked.
            Carried because the two need different sentences and because a
            severed part has *no cells*, so its ``forced_by_thickness`` is 0.0
            and reads as a nonsense measurement.
        forced_by_thickness: That part's thickness at ``configured``, or 0.0
            when ``forced_by_cause`` is ``"severed"``.
        forced_by_severing: The severing that blocked the configured grid, when
            ``forced_by_cause`` is ``"severed"``; ``None`` otherwise. Distinct
            from ``severing``, which describes the grid actually chosen.
        cells_configured: Solve-grid cell count at ``configured``.
        cells_chosen: Solve-grid cell count at ``coarsening``.
        resolved: True when every metal-path part clears the hard minimum at
            ``coarsening`` **and** the metal path is connected there. False
            means even coarsening 1 could not fix it and the run proceeds under
            the hard warning -- or, when ``severing.severed``, must not proceed
            at all.
        enabled: Whether auto-resolution was allowed to act.
    """

    coarsening: int
    configured: int
    solve_types: np.ndarray
    by_part: Dict[int, float]
    components_by_part: Dict[int, Tuple[float, ...]]
    fragments_by_part: Dict[int, Tuple[int, ...]]
    severing: MetalPathSevering
    forced_by: Optional[int]
    forced_by_cause: str
    forced_by_thickness: float
    forced_by_severing: Optional[MetalPathSevering]
    cells_configured: int
    cells_chosen: int
    resolved: bool
    enabled: bool

    @property
    def refined(self) -> bool:
        """True when the grid actually changed from what was configured."""
        return self.coarsening != self.configured


def choose_fill_coarsening(
    fine_types: np.ndarray,
    configured: int,
    *,
    enabled: bool = True,
) -> CoarseningDecision:
    """
    Pick the fill-grid coarsening that resolves every gating channel.

    A channel at or below :data:`MIN_CHOKE_THICKNESS_CELLS` has no interior
    once halfway bounce-back has put the walls at +-0.5 cells: it seals, and
    the fill through it is not a result whatever the run reports. Round 1 of
    this work made that visible as a warning. Warning about a grid the stage
    chose itself, and could just as easily choose better, is only half an
    answer -- so the stage now measures the channels at the configured
    coarsening and, if any is sealed, refines a step at a time until they all
    clear or it reaches 1.

    Refining is not free: cost goes as the cube of the resolution in cells
    plus a linear factor in steps (the timestep shrinks with the cell), so
    2 -> 1 is roughly a sixteen-fold increase in work. That is why this only
    triggers on the *hard* threshold, never on the marginal one -- a
    three-cell gate is worth a note, not a sixteen-fold bill.

    Refinement steps down by one rather than jumping to 1: the first factor
    that clears the threshold is the cheapest grid that can carry the fill,
    and on ``Test 2`` that is 1 only because its gate is a two-cell plate at
    the shipped coarsening of 2.

    When it cannot be fixed -- the gate is still sealed at coarsening 1, so
    the *mesh* is too coarse and no downsampling choice can help -- the
    decision comes back at coarsening 1 with ``resolved`` False and the
    caller keeps the hard warning. An honest misrun beats a silent one.

    Args:
        fine_types: The mesh-resolution dense part-type grid.
        configured: ``fill_grid_coarsening``, at least 1.
        enabled: ``choke_auto_resolution``. When False the configured factor
            is returned untouched, measured but not acted on.

    Returns:
        A :class:`CoarseningDecision`, always carrying the ``solve_types``
        grid for the factor it chose.
    """
    configured = max(1, int(configured))

    class _Measured(NamedTuple):
        types: np.ndarray
        by_part: Dict[int, float]
        components: Dict[int, Tuple[float, ...]]
        fragments: Dict[int, Tuple[int, ...]]
        severing: MetalPathSevering

    class _Blocked(NamedTuple):
        code: Optional[int]
        cause: str

    def measure(factor: int) -> _Measured:
        # One _choke_bodies call per factor, not three: the thickness map it
        # builds is the expensive part of this whole function (an EDT sweep per
        # half-cell of inscribed radius over the full grid), so the three public
        # views are derived here rather than each recomputing it.
        types = coarsen_part_types(fine_types, factor)
        bodies = _choke_bodies(types)
        by_part: Dict[int, float] = {}
        fragments: Dict[int, Tuple[int, ...]] = {}
        for code, group in bodies.items():
            resolvable = [
                t for t, cells in group if cells >= MIN_CHOKE_BODY_CELLS
            ]
            by_part[code] = (
                min(resolvable) if resolvable else min(t for t, _ in group)
            )
            small = sorted(
                cells for _, cells in group if cells < MIN_CHOKE_BODY_CELLS
            )
            if small and len(small) < len(group):
                fragments[code] = tuple(small)
        return _Measured(
            types=types,
            by_part=by_part,
            components={
                code: tuple(t for t, _ in group)
                for code, group in bodies.items()
            },
            fragments=fragments,
            severing=metal_path_severing(fine_types, types, factor),
        )

    def blocked(m: _Measured) -> _Blocked:
        """
        The upstream-most metal-path part this grid cannot carry, and why.

        Two ways to fail, and a severed path is checked first because it is the
        more severe *and* the more invisible: a part at or below the hard
        minimum is a channel with no interior but at least has cells to
        measure, whereas a part the coarsening cut may have none, so every
        thickness check silently skips it. The cause travels with the code
        because the two need different sentences -- a severed part's thickness
        is 0.0, which reads as a measurement and is not one.
        """
        if m.severing.severed:
            parts = m.severing.parts
            return _Blocked(parts[0] if parts else None, "severed")
        under = {
            code: value
            for code, value in m.by_part.items()
            if 0.0 < value <= MIN_CHOKE_THICKNESS_CELLS
        }
        code = narrowest_choke_part_code(under)
        return _Blocked(code, "thickness" if code is not None else "")

    def decide(m: _Measured, factor: int, **kw) -> CoarseningDecision:
        return CoarseningDecision(
            coarsening=factor,
            configured=configured,
            solve_types=m.types,
            by_part=m.by_part,
            components_by_part=m.components,
            fragments_by_part=m.fragments,
            severing=m.severing,
            cells_configured=cells_configured,
            cells_chosen=int(np.prod(m.types.shape)),
            enabled=enabled,
            **kw,
        )

    base = measure(configured)
    cells_configured = int(np.prod(base.types.shape))
    first = blocked(base)
    forced_kw = dict(
        forced_by=first.code,
        forced_by_cause=first.cause,
        forced_by_thickness=(
            base.by_part.get(first.code, 0.0)
            if first.cause == "thickness" and first.code is not None
            else 0.0
        ),
        forced_by_severing=base.severing if first.cause == "severed" else None,
    )

    if not enabled or first.code is None or configured == 1:
        return decide(base, configured, resolved=first.code is None, **forced_kw)

    for factor in range(configured - 1, 0, -1):
        m = measure(factor)
        if blocked(m).code is None:
            return decide(m, factor, resolved=True, **forced_kw)
        if factor == 1:
            # Out of room. Hand back the finest grid available and let the
            # caller warn: it is the best this mesh can do, and saying so is
            # more use than silently running the coarse one. The path cannot be
            # severed here -- at factor 1 the two grids are the same -- so the
            # caller warns rather than refusing.
            return decide(m, 1, resolved=False, **forced_kw)

    raise AssertionError("unreachable: the loop above always returns")


def measure_choke_thickness_cells(dense_types: np.ndarray) -> float:
    """
    Thickness of the thinnest gating section, in fill-grid cells.

    The scalar summary of :func:`measure_choke_thickness_by_part` -- the
    smallest of the per-part figures. See that function for what is measured
    and why the median rather than the maximum.

    Returns:
        Thickness in cells at the narrowest section, or 0.0 when there is no
        gating metal.
    """
    by_part = measure_choke_thickness_by_part(dense_types)
    if not by_part:
        return 0.0
    return max(0.0, float(min(by_part.values())))


# ---------------------------------------------------------------------------
# Stage orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FillFrame:
    """
    One instant of the pour, gathered onto voxels.

    Attributes:
        time_s: Physical time since the pour started.
        fill_fraction: Per-voxel metal content, 0 (empty) to 1 (full). The
            partial values are the free surface -- the interface layer is
            exactly where 0 < eps < 1, so a threshold on this field draws the
            advancing front rather than a staircase of whole cells.
        speed_m_s: Per-voxel velocity magnitude, in metres per second.
    """

    time_s: float
    fill_fraction: np.ndarray
    speed_m_s: np.ndarray


@dataclass(frozen=True)
class FillingResult:
    """
    What the filling stage hands back to the pipeline.

    ``fill_time_s`` and ``temperature_c`` are ``(n_voxels,)`` -- the *occupied*
    voxel count, which is what
    ``SolidificationEngine.set_initial_temperature_field`` requires. They are
    not dense-grid shaped.

    Attributes:
        fill_time_s: Arrival time per voxel; ``-1`` where metal never arrived.
        temperature_c: Initial temperature per voxel, degrees Celsius. F2:
            the metal's actual thermal history through the fill -- the metal
            that entered first and travelled furthest starts cooler than
            metal at the gate -- not the two-constant (pour temp / mold temp)
            placeholder F1 used.
        inlet_voxel_count: Voxels the pour location resolved to.
        filled_voxel_count: Metal voxels the fill reached.
        misrun_voxel_count: Metal voxels it did not.
        choke_area_cm2: Narrowest gating cross-section, measured **on the grid
            the solver ran** -- the coarsened one, when
            ``fill_grid_coarsening`` is above 1. That is the choke the fill
            actually had.
        filter_zones: One entry per ceramic filter found.
        steps: Lattice steps taken.
        fill_seconds: Physical time simulated.
        coarsening: Fill-grid coarsening factor used.
        min_fill_temperature_c: Coolest filled metal at handover. Below the
            alloy's liquidus this is incipient cold-shut territory.
        mean_fill_temperature_c: Average filled-metal temperature at handover.
        entrapped_voxel_count: Voxels of trapped gas at the end of the fill
            (F3). Zero when every gas pocket found a vent.
        peak_entrapped_voxel_count: The largest entrapped-gas volume seen at
            any periodic check during the fill, not just at the end -- a
            pocket can be entrapped mid-fill and vent later, so the final
            snapshot alone understates the risk. Always ``>=
            entrapped_voxel_count``.
        cold_shut_voxel_count: Voxels flagged as a cold-shut seam (F3): a
            wet cell where fronts converged from two or more directions and
            the metal had already cooled to within
            ``FillingParams.cold_shut_temp_margin_c`` of the liquidus.
        entrapped_gas: Per-voxel boolean, ``(n_voxels,)``, the final
            entrapped-gas set gathered the same way ``fill_time_s`` is.
        cold_shut: Per-voxel boolean, ``(n_voxels,)``, the cold-shut set.
        frames: Playback snapshots of the pour, oldest first. Empty when the
            run diverged, because a frame taken after the blow-up shows the
            blow-up. See :class:`FillFrame`.
        warnings: Human-readable conditions worth surfacing.
        status: How the run ended. One of ``STATUS_COMPLETE`` (every metal
            cell filled), ``STATUS_TIME_CAPPED`` (still advancing when
            ``max_fill_time_s`` ran out), ``STATUS_STALLED`` (the front stopped
            advancing while metal remained unfilled) or ``STATUS_DIVERGED``
            (the solver lost mass conservation -- the fields are numerically
            meaningless and must not be trusted or handed downstream).

            This distinction is the difference between "the metal could not
            get there", "the run needed longer" and "the answer is garbage".
            All three used to surface identically as a misrun count.
        diverged_at_step: Lattice step at which divergence was detected, or 0.
        choke_thickness_cells: Thickness of the thinnest gating section, in
            fill-grid cells. At or below ``MIN_CHOKE_THICKNESS_CELLS`` the
            channel has no interior and halfway bounce-back seals it shut.
        choke_thickness_by_part: ``{"sprue"|"runner"|"filter"|"gate":
            thickness in cells}`` for every metal-path part present, so a
            reader can see *which* part is under-resolved rather than only
            that something is. The value is the **thinnest body** of that
            part, not a pooled median over all of them -- see
            :func:`measure_choke_thickness_by_part` for the two-gate case that
            made the difference.
        choke_thickness_bodies_by_part: The same keys, each mapped to *every*
            connected body's thickness, thinnest first -- including the
            fragments ``choke_thickness_by_part`` skips. Added alongside rather
            than widening ``choke_thickness_by_part``, whose scalar-per-part
            shape the pipeline payload and its readers already depend on.
        choke_body_fragments_by_part: The same keys, each mapped to the cell
            counts of the bodies below :data:`MIN_CHOKE_BODY_CELLS` that did
            not vote on the grid decision.
        narrowest_choke_part: Name of the thinnest of them, or ``""`` when the
            grid holds no gating metal.
        choke_area_fine_cm2: The same cross-section measured on the *mesh*
            grid, before coarsening. Reported alongside ``choke_area_cm2``
            rather than instead of it, because the two answer different
            questions -- what the geometry has, and what the solver saw. They
            are not close: 2.98 against 7.46 cm^2 on the ``Test 2`` assembly.
        delivered_lb: Metal the inlet actually delivered over the run, in
            pounds -- the integral of what the boundary put into the domain,
            not an estimate from the fill fraction. Reported because the
            difference between this and ``target_delivery_lb_s`` times
            ``fill_seconds`` is the whole story of whether the pour ran at
            rate or starved.
        target_delivery_lb_s: The rate the pour was metered against, after
            the filter rating and the Mach guard had their say. In head mode
            it is the smaller of what the casting wants (the square-root
            rule) and what the mouth can pass (Torricelli through its
            measured area), unless ``pour_rate_lb_s`` sets one explicitly; in
            velocity mode it is the prescribed rate.
        configured_coarsening: ``fill_grid_coarsening`` as configured, which
            differs from ``coarsening`` when auto-resolution refined the grid.
    """

    fill_time_s: np.ndarray
    temperature_c: np.ndarray
    inlet_voxel_count: int
    filled_voxel_count: int
    misrun_voxel_count: int
    choke_area_cm2: float
    filter_zones: List[FilterZoneSpec]
    steps: int
    fill_seconds: float
    coarsening: int
    min_fill_temperature_c: float
    mean_fill_temperature_c: float
    entrapped_voxel_count: int = 0
    peak_entrapped_voxel_count: int = 0
    cold_shut_voxel_count: int = 0
    entrapped_gas: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    cold_shut: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    warnings: Tuple[str, ...] = ()
    status: str = STATUS_COMPLETE
    diverged_at_step: int = 0
    choke_thickness_cells: float = 0.0
    choke_thickness_by_part: Dict[str, float] = field(default_factory=dict)
    choke_thickness_bodies_by_part: Dict[str, List[float]] = field(
        default_factory=dict
    )
    choke_body_fragments_by_part: Dict[str, List[int]] = field(
        default_factory=dict
    )
    narrowest_choke_part: str = ""
    choke_area_fine_cm2: float = 0.0
    delivered_lb: float = 0.0
    target_delivery_lb_s: float = 0.0
    configured_coarsening: int = 0
    frames: Tuple[FillFrame, ...] = ()

    @property
    def filled_fraction(self) -> float:
        total = self.filled_voxel_count + self.misrun_voxel_count
        return self.filled_voxel_count / total if total else 0.0

    @property
    def trustworthy(self) -> bool:
        """
        Whether the fields may be used downstream.

        A diverged run still carries a ``fill_time_s`` and a ``temperature_c``
        of the right shape and dtype -- they are simply meaningless, because
        the blow-up flags cells as filled that the metal never reached. Seeding
        solidification from them produces a plausible-looking casting result
        built on nothing, which is worse than not running the stage at all.
        """
        return self.status != STATUS_DIVERGED


def _obstacle_mask(dense_types: np.ndarray) -> np.ndarray:
    """
    Cells the metal cannot enter.

    Empty cells (outside the meshed geometry) and mold materials are solid --
    **except the filter**, which metal flows *through*. ``is_mold()`` alone
    would make a ceramic filter an impermeable wall and stop the fill dead at
    the gate, which is why the flow-restrictor axis exists separately.
    """
    from core.part_manager import is_flow_restrictor, is_mold

    types = np.asarray(dense_types)
    empty = types == EMPTY_CELL
    safe = np.where(empty, PART_TYPE_CAST, types)
    solid: np.ndarray = empty | (is_mold(safe) & ~is_flow_restrictor(safe))
    return solid


def _metal_mask(dense_types: np.ndarray) -> np.ndarray:
    """Cells that should end up holding metal."""
    from core.part_manager import is_metal

    types = np.asarray(dense_types)
    empty = types == EMPTY_CELL
    safe = np.where(empty, PART_TYPE_CHILL, types)
    metal: np.ndarray = (~empty) & is_metal(safe)
    return metal


def _resolve_inlet_cells(
    pour_location,
    mapping: GridMapping,
    dense_types: np.ndarray,
    coarsening: int,
) -> np.ndarray:
    """
    Turn the user's clicked point into a set of lattice cells.

    Works in lattice index space rather than voxel indices, because the fill
    grid may be coarser than the mesh and voxel indices do not survive
    downsampling.

    Only metal cells are eligible: a click that lands on a chill, a core or
    empty space is a user error, and injecting metal there would produce a
    fill that starts inside the mold.
    """
    from core.pour_location import resolve_inlet_voxels

    voxel_idx = resolve_inlet_voxels(
        pour_location,
        _voxel_centers_from_mapping(mapping),
        mapping.dx_in,
    )
    if voxel_idx.size == 0:
        return np.zeros((0, 3), dtype=np.int64)

    ijk = mapping.voxel_to_grid[voxel_idx].astype(np.int64)
    if coarsening > 1:
        ijk = ijk // coarsening
        ijk = np.unique(ijk, axis=0)

    keep = _metal_mask(dense_types)[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    selected: np.ndarray = ijk[keep]
    return selected


def _expand_inlet_to_mouth(
    inlet: np.ndarray,
    dense_types: np.ndarray,
) -> np.ndarray:
    """
    Widen the resolved inlet from a point to the whole mouth of its feature.

    The pour location's job is to pick *which* feature is poured into. The
    physical inlet is that feature's entire top cross-section, because a
    pouring basin sits on the full mouth of the sprue, not on a spot in the
    middle of it.

    Measured on ``Test_Model/Full_Gating_Test``: :func:`_resolve_inlet_cells`'
    1.5-voxel radius picked 8 cells of the sprue top's cross-section -- 44
    cells when that was first measured, 48 on the current mesh. The 8-cell jet
    atomised on the way down the sprue and the run diverged at 24 % by mass.
    Feeding the same pour through the whole mouth -- changing nothing else --
    cleared the divergence and ran the sprue full.

    The flood is confined to the seeds' own top layer and to the part types the
    seeds are standing on, so a pour cup widens over the cup, a sprue over the
    sprue and a bare casting over its own top face. None of them leak into a
    neighbouring part that happens to share the layer, and none of them leak
    downwards into the feature's bore.

    It widens the layer the click landed in, not "the topmost layer of the
    feature", so a pour location placed on a *side* face resolves to the whole
    horizontal cross-section at that height rather than to a patch of the
    flank. That is the intended reading -- the click chooses the feature and
    the metal enters across it -- but it is why the UI asks for a point on the
    top face of the sprue.

    Args:
        inlet: ``(n, 3)`` seed cells, as :func:`_resolve_inlet_cells` returns.
        dense_types: The part-type grid those cells index into.

    Returns:
        ``(m, 3)`` int64 cells: the seeds plus the flooded mouth, ``m >= n``.
    """
    from scipy import ndimage

    seeds = np.asarray(inlet, dtype=np.int64).reshape(-1, 3)
    if seeds.shape[0] == 0:
        return seeds

    types = np.asarray(dense_types)
    z_top = int(seeds[:, 2].max())
    top = seeds[seeds[:, 2] == z_top]
    layer = types[:, :, z_top]

    # Only the types the seeds actually stand on, and only metal ones: the
    # flood must not walk out of the feature being poured into.
    codes = np.unique(layer[top[:, 0], top[:, 1]])
    member = np.isin(layer, codes) & _metal_mask(layer)
    if not member.any():
        return seeds

    # 8-connectivity in the plane. A sprue mouth is a disc drawn on a lattice;
    # its corner-touching cells are one body, not several.
    labels, _ = ndimage.label(member, structure=np.ones((3, 3), dtype=bool))
    seed_labels = np.unique(labels[top[:, 0], top[:, 1]])
    seed_labels = seed_labels[seed_labels > 0]
    if seed_labels.size == 0:
        return seeds

    mx, my = np.nonzero(np.isin(labels, seed_labels))
    mouth = np.column_stack(
        (mx, my, np.full(mx.shape[0], z_top))
    ).astype(np.int64)
    widened: np.ndarray = np.unique(np.vstack((seeds, mouth)), axis=0)
    return widened


def _voxel_centers_from_mapping(mapping: GridMapping) -> np.ndarray:
    """Reconstruct voxel centres in inches from the mapping."""
    return mapping.origin_in[None, :] + mapping.voxel_to_grid.astype(np.float64) * mapping.dx_in


def _upsample(coarse: np.ndarray, fine_shape: Tuple[int, int, int], factor: int) -> np.ndarray:
    """Nearest-neighbour expansion of a coarse field back to the fine grid."""
    if factor == 1:
        return coarse
    idx = [np.minimum(np.arange(n) // factor, coarse.shape[a] - 1)
           for a, n in enumerate(fine_shape)]
    return coarse[np.ix_(idx[0], idx[1], idx[2])]


def run_filling_stage(
    voxel_mesh,
    alloy,
    analysis_options,
    pour_location,
    *,
    cancel_callback=None,
    progress_message_callback=None,
    detailed_progress_callback=None,
) -> Optional[FillingResult]:
    """
    Simulate the pour and return the state it leaves behind.

    Args:
        voxel_mesh: A generated :class:`~core.voxel_mesh.AdaptiveVoxelMesh`
            with part types assigned.
        alloy: An :class:`~core.materials.AlloyProperties`.
        analysis_options: Carries ``solidification_params.filling``.
        pour_location: A :class:`~core.pour_location.PourLocation`.
        cancel_callback: Raises to abort.
        progress_message_callback: Receives human-readable progress.

    Returns:
        A :class:`FillingResult`, or ``None`` if the stage could not run. The
        reason is emitted through ``progress_message_callback`` rather than
        raised: an unfillable pour point is a modelling problem the user should
        see and fix, not a crash.
    """
    from physics.filling import (
        FLAG_FLUID,
        FLAG_GAS,
        FLAG_INTERFACE,
        FLAG_OBSTACLE,
        REFERENCE_DENSITY,
        advect_temperature,
        apply_wall_heat_loss,
        create_state,
        ensure_interface_skin,
        find_cold_shuts,
        find_entrapped_gas,
        fluid_fraction,
        step,
    )
    from physics.fluid_flow import LATTICE_VELOCITIES, equilibrium, lattice_units
    from core.part_manager import CODE_TO_PART_TYPE, is_gating_metal

    def emit(message: str) -> None:
        if progress_message_callback is not None:
            progress_message_callback(message)

    params = getattr(analysis_options, "solidification_params", None)
    filling = getattr(params, "filling", None)
    if filling is None:
        emit("Filling stage skipped: no filling parameters configured.")
        return None
    thermal_boundary = getattr(params, "thermal_boundary", None)

    # 0 means "use the alloy's own pouring temperature" -- see FillingParams'
    # docstring for pour_temperature_c.
    pour_temp = float(getattr(filling, "pour_temperature_c", 0.0)) or float(alloy.pouring_temp)
    mold_temp = float(getattr(thermal_boundary, "mold_temperature", 21.0))

    centers = np.asarray(voxel_mesh.voxel_centers, dtype=np.float64)
    dx_in = float(np.asarray(voxel_mesh.voxel_sizes).ravel()[0])
    try:
        mapping = build_grid_mapping(centers, dx_in)
    except ValueError as exc:
        emit(f"Filling stage skipped: {exc}")
        return None

    fine_types = dense_part_types(mapping, np.asarray(voxel_mesh.voxel_part_types))

    # Which grid to solve on is decided before anything is built on it, because
    # a sealed gating channel is not an inaccuracy to report afterwards -- it
    # is a fill that cannot happen. See choose_fill_coarsening.
    coarsening_decision = choose_fill_coarsening(
        fine_types,
        max(1, int(getattr(filling, "fill_grid_coarsening", 2))),
        enabled=int(getattr(filling, "choke_auto_resolution", 1)) != 0,
    )
    coarsening = coarsening_decision.coarsening
    solve_types = coarsening_decision.solve_types
    if coarsening_decision.refined:
        forced = coarsening_decision.forced_by
        cost = (
            coarsening_decision.cells_chosen
            / max(1, coarsening_decision.cells_configured)
        )
        label = _METAL_PATH_LABELS.get(forced, "gating") if forced is not None else "gating"
        severing = coarsening_decision.forced_by_severing
        if severing is not None:
            # A severed part may have no cells at all, so there is no thickness
            # to quote -- quoting one anyway printed "measured 0.0 cells
            # across" for a part that was simply gone.
            cause = (
                f"coarsening x{coarsening_decision.configured} cut the metal "
                f"path -- {severing.describe()}, "
                f"{severing.fine_components} connected "
                f"{'run' if severing.fine_components == 1 else 'runs'} of gating "
                f"metal on the mesh against "
                f"{severing.solve_components} on that grid. The downsample "
                f"gives each coarse block its highest-priority code and mold "
                f"outranks the flow path by design, so a channel one mesh cell "
                f"thick against a core is replaced by the core"
            )
        else:
            cause = (
                f"the {label} measured "
                f"{coarsening_decision.forced_by_thickness:.1f} cells across at "
                f"coarsening x{coarsening_decision.configured}, at or below the "
                f"{MIN_CHOKE_THICKNESS_CELLS:.0f} cells a channel needs before "
                f"it has an interior"
            )
        emit(
            f"Filling: refining the fill grid -- {cause}. "
            f"Coarsening x{coarsening_decision.configured} -> "
            f"x{coarsening}, {coarsening_decision.cells_configured:,} -> "
            f"{coarsening_decision.cells_chosen:,} cells "
            f"({cost:.0f}x the cells, and a shorter timestep with them, so "
            f"expect a substantially longer solve)."
            + (
                ""
                if coarsening_decision.resolved
                else " Even at coarsening x1 the channel is still sealed -- "
                "the mesh itself is too coarse. Running anyway, under the "
                "warning below."
            )
        )

    # The metal path is cut on the grid we were about to solve. Refusing rather
    # than solving, because there is nothing to solve: the fill would run with
    # the path severed and report the resulting misrun as a result.
    #
    # Only reachable with auto-resolution off. Every grid choose_fill_coarsening
    # *chooses* has an intact path, and at coarsening 1 the solve grid is the
    # mesh grid, so severing is impossible there -- which is why the remedy
    # below names the toggle and does not offer a second arm for a case that
    # cannot occur. See metal_path_severing.
    if coarsening_decision.severing.severed:
        severing = coarsening_decision.severing
        emit(
            f"Filling stage skipped: coarsening x{coarsening} cuts the metal "
            f"path -- {severing.describe()}. The gating metal is "
            f"{severing.fine_components} connected "
            f"{'run' if severing.fine_components == 1 else 'runs'} on the mesh "
            f"and {severing.solve_components} on the fill grid. The downsample "
            f"gives each coarse block its highest-priority part code, and mold "
            f"outranks the flow path by design (a chill that vanishes changes "
            f"the thermal answer), so a channel one mesh cell thick against a "
            f"core is replaced by the core. Nothing downstream can see it: the "
            f"thickness guards are keyed on parts that are present, so the fill "
            f"would run with the path severed and report a misrun. Set "
            f"choke_auto_resolution to 1 and the stage will refine the grid "
            f"itself, or reduce fill_grid_coarsening."
        )
        return None

    # Seal the lattice: one cell of mold on every face, plus the headspace
    # above it. See MOLD_PAD_CELLS for the wrap this removes and HEADSPACE_CELLS
    # for why the top gets more. Everything below this line works on the padded
    # grid; ``solve_core`` is the view that still speaks the mapping's
    # coordinates, and ``to_voxels`` is the only way a solve field is allowed
    # back onto the voxel list.
    pad_lo = int(MOLD_PAD_CELLS)
    core_shape: Tuple[int, int, int] = (
        int(solve_types.shape[0]),
        int(solve_types.shape[1]),
        int(solve_types.shape[2]),
    )
    core = tuple(slice(pad_lo, pad_lo + n) for n in core_shape)
    solve_types = np.pad(
        solve_types,
        (
            (pad_lo, pad_lo),
            (pad_lo, pad_lo),
            (pad_lo, pad_lo + int(HEADSPACE_CELLS)),
        ),
        mode="constant",
        constant_values=EMPTY_CELL,
    )
    solve_core = solve_types[core]

    def to_voxels(field: np.ndarray) -> np.ndarray:
        """
        One solve field back onto the voxel list, pad cropped off first.

        ``gather_to_voxels`` asserts the dense shape against ``mapping.shape``,
        and ``_upsample`` at coarsening 1 returns its input untouched, so a
        padded field either raises there or -- at coarsening > 1 -- gathers
        every voxel one cell off along all three axes. Cropping here is the one
        place that cannot be forgotten per call site.
        """
        return gather_to_voxels(
            mapping, _upsample(np.asarray(field)[core], mapping.shape, coarsening)
        )

    # Resolved on the unpadded view, because the pour point arrives in voxel
    # coordinates and the mapping only knows those, then shifted onto the
    # padded grid the solver runs on.
    inlet_seeds = _resolve_inlet_cells(pour_location, mapping, solve_core, coarsening)
    if inlet_seeds.shape[0] == 0:
        emit(
            "Filling stage skipped: the pour location does not resolve to any "
            "metal cell. Move it onto the pour cup, sprue or casting."
        )
        return None

    # The click picks the feature; the mouth of that feature is the inlet. This
    # is geometry resolution rather than boundary mechanics, so it applies to
    # both inlet modes. In velocity mode the prescribed mass rate is divided by
    # the now-larger inlet area, which lowers the imposed velocity for the same
    # kg/s -- correct, and exactly what a basin sitting on the whole sprue
    # does. See _expand_inlet_to_mouth for the measurement that made this
    # necessary: 8 cells of a sprue mouth is a jet, not a pour.
    inlet = _expand_inlet_to_mouth(inlet_seeds, solve_core) + pad_lo
    inlet_seed_count = int(inlet_seeds.shape[0])
    inlet_widening = ""
    if inlet.shape[0] > inlet_seed_count:
        mouth_top = inlet[:, 2] == inlet[:, 2].max()
        mouth_names = [
            CODE_TO_PART_TYPE[int(code)].value.replace("_", " ")
            for code in np.unique(
                solve_types[inlet[mouth_top, 0], inlet[mouth_top, 1],
                            inlet[mouth_top, 2]]
            )
            if int(code) in CODE_TO_PART_TYPE
        ]
        inlet_widening = (
            f" ({inlet_seed_count:,} at the pour point, widened to the "
            f"{'/'.join(mouth_names) or 'poured feature'} mouth)"
        )

    dx_m = mapping.dx_m * coarsening
    nu = float(alloy.dynamic_viscosity) / float(alloy.density_liquid)

    # The timestep follows from the fastest velocity that must be
    # representable. Free fall from the top of the assembly, sqrt(2 g H), is a
    # defensible upper bound: nothing in a gravity pour moves faster than metal
    # dropped from the highest point of the rigging. Under-estimating it breaks
    # the Mach guard; over-estimating it only costs steps.
    height_m = float(inches_to_meters(
        (mapping.shape[2] - 1) * mapping.dx_in
        + float(getattr(filling, "initial_head_in", 4.0))
    ))
    u_max_physical = max(0.5, float(np.sqrt(2.0 * 9.81 * max(height_m, 1e-6))))
    units = lattice_units(
        dx_m,
        nu,
        u_max_physical_m_s=u_max_physical,
        u_max_lattice=float(getattr(filling, "lattice_velocity_max", 0.05)),
    )

    obstacle = _obstacle_mask(solve_types)
    metal = _metal_mask(solve_types)
    if not np.any(metal):
        emit("Filling stage skipped: no metal cells in the assembly.")
        return None

    # Atmosphere over whatever reaches the original +z face -- the riser tops,
    # and on some assemblies the sprue mouth. See HEADSPACE_CELLS.
    headspace = np.zeros(solve_types.shape, dtype=bool)
    if HEADSPACE_CELLS:
        from scipy import ndimage

        z_top = core[2].stop - 1
        open_above = metal[:, :, z_top].copy()
        # Keep the headspace out of the inlet's *whole reach*, not just out of
        # the mouth's own columns.
        #
        # ``inlet_halo`` below is an 18-direction dilation of the mouth, and
        # five of those directions carry c_z = +1: they land in the first
        # headspace plane over the columns *diagonally beside* the mouth. If
        # such a column is metal on the original top face its headspace is
        # open, so it survives ``inlet_halo &= ~obstacle`` and
        # ensure_interface_skin promotes it every step -- wet cells above the
        # geometry, outside ``metal`` and cropped out of every result field, so
        # their mass shows up in ``poured`` and nowhere else. Measured on Test
        # 2's grid: 0 such cells for the sprue pour the validation suite runs,
        # but 35 for a riser-top pour and 6 for a cast-top pour, both of which
        # _expand_inlet_to_mouth explicitly supports. Same geometry, different
        # click.
        #
        # An 8-neighbourhood in (x, y) is exactly the halo's footprint on the
        # plane above, so clearing it is sufficient and costs only the ring of
        # columns around the mouth -- which have the mouth's own atmosphere
        # beside them anyway. binary_dilation rather than np.roll: it treats the
        # array edge as False instead of wrapping, which is the whole point of
        # this section of the file.
        mouth = np.zeros(solve_types.shape[:2], dtype=bool)
        on_mouth = inlet[:, 2] == z_top
        mouth[inlet[on_mouth, 0], inlet[on_mouth, 1]] = True
        open_above &= ~ndimage.binary_dilation(
            mouth, structure=np.ones((3, 3), dtype=bool)
        )
        headspace[:, :, core[2].stop : core[2].stop + int(HEADSPACE_CELLS)] = (
            open_above[:, :, None]
        )
        obstacle &= ~headspace

    # The lattice is a torus unless every array face is solid, and both step
    # paths wrap silently rather than raising -- so this is checked here, once,
    # where the grid is built, rather than left to be rediscovered as a phantom
    # pool 12,000 steps into a pour. See MOLD_PAD_CELLS.
    #
    # Reported the way every other unrunnable geometry in this function is
    # reported -- emit and return None -- rather than raised. Nothing wraps
    # maybe_run_filling in a try (core/simulation_pipeline.py:574,
    # core/surrogate/pipeline.py:201), so raising here would tear down a run
    # that has already paid for meshing, with no FillingResult and no message
    # the UI can show. The message is the loud part.
    open_faces = [
        name
        for name, plane in (
            ("x = 0", obstacle[0]),
            ("x = max", obstacle[-1]),
            ("y = 0", obstacle[:, 0]),
            ("y = max", obstacle[:, -1]),
            ("z = 0", obstacle[:, :, 0]),
            ("z = max", obstacle[:, :, -1]),
        )
        if not plane.all()
    ]
    if open_faces:
        emit(
            f"Filling stage skipped: the solve lattice is not closed. "
            f"Non-obstacle cells sit on the {', '.join(open_faces)} "
            f"face{'' if len(open_faces) == 1 else 's'} of the array, and both "
            f"step paths index neighbours periodically, so opposite faces of "
            f"the domain are neighbours of each other -- metal on one face "
            f"would stream and exchange mass straight into whatever sits on "
            f"the other, however far apart they are. The "
            f"{pad_lo}-cell mold pad is what makes that unrepresentable and it "
            f"has been defeated; this is a bug in the stage, not in the model."
        )
        return None

    filter_zones = build_filter_zone_specs(
        solve_types,
        dx_m=dx_m,
        porosity=float(getattr(filling.filter, "porosity", 0.85)),
        pore_diameter_m=float(getattr(filling.filter, "pore_diameter_mm", 2.54)) / 1000.0,
        ergun_prefactor=float(getattr(filling.filter, "ergun_prefactor", 2.2)),
        forchheimer_c_f=float(getattr(filling.filter, "forchheimer_c_f", 0.55)),
        surface_tension_n_m=float(getattr(alloy, "surface_tension", 1.8)),
        contact_angle_deg=float(getattr(filling.filter, "contact_angle_deg", 120.0)),
        window_fraction=float(getattr(filling.filter, "window_fraction", 0.45)),
        density_kg_m3=float(alloy.density_liquid),
    )
    porous = build_porous_zone(
        solve_types,
        filter_zones,
        dx_m=dx_m,
        dt_s=units.dt_s,
        nu_lattice=units.nu_lattice,
        density_kg_m3=float(alloy.density_liquid),
    )

    seed = np.zeros(solve_types.shape, dtype=bool)
    seed[inlet[:, 0], inlet[:, 1], inlet[:, 2]] = True
    state = create_state(
        obstacle,
        dx_m=dx_m,
        dt_s=units.dt_s,
        initial_fluid=seed,
        porous=porous,
        initial_temperature_c=pour_temp,
    )

    # F2 thermal coupling. alpha_lattice is rescaled exactly the way
    # lattice_units() rescales viscosity into nu_lattice, so it drops
    # straight into physics.filling's explicit stencil as a dimensionless
    # number -- see tests/test_filling_thermal.py for the stability margin
    # this buys at realistic settings.
    alpha_si = float(alloy.thermal_conductivity) / (
        float(alloy.density_liquid) * float(alloy.specific_heat)
    )
    alpha_lattice = alpha_si * units.dt_s / (dx_m * dx_m)

    # Gating-system surfaces (pour cup, sprue, runner, gate) and the casting
    # itself sit in different sand contexts and lose heat at different rates
    # -- see ThermalBoundaryParams.htc_gating's docstring for why an
    # unconfigured gating surface would otherwise integrate adiabatically.
    wall_htc_multiplier = float(getattr(filling, "wall_htc_multiplier", 1.0))
    htc_gating = float(getattr(thermal_boundary, "htc_gating", 400.0)) * wall_htc_multiplier
    htc_cast = float(getattr(thermal_boundary, "htc_cast_interface", 500.0)) * wall_htc_multiplier
    htc_dense = np.where(is_gating_metal(solve_types), htc_gating, htc_cast)

    max_time = float(getattr(filling, "max_fill_time_s", 60.0))
    max_steps = int(max_time / units.dt_s)
    emit(
        f"Filling: {int(metal.sum()):,} cells at {dx_m * 1000:.1f} mm "
        f"(coarsening x{coarsening}), up to {max_steps:,} steps"
    )

    warnings: List[str] = []

    # The subgrid constant the collision operator runs at. Out-of-range values
    # are refused rather than clamped: silently pulling 1.2 back to 0.5 would
    # still hand the user a viscosity 40x what they think they asked for, and
    # the whole reason this knob is visible is that a control which quietly
    # does something else is worse than no control. NaN fails the comparison
    # too, which is the intent.
    smagorinsky_cs = float(getattr(filling, "smagorinsky_cs", SMAGORINSKY_CS))
    if not 0.0 <= smagorinsky_cs <= MAX_SMAGORINSKY_CS:
        warnings.append(
            f"smagorinsky_cs of {smagorinsky_cs:.3f} is outside the usable "
            f"range 0-{MAX_SMAGORINSKY_CS:.1f} (literature values are "
            f"0.1-0.2); running at the default {SMAGORINSKY_CS:.2f}."
        )
        smagorinsky_cs = float(SMAGORINSKY_CS)

    # Inlet density from the metallostatic head above the pour point:
    # rho = rho_0 (1 + g h / c_s^2) in lattice units. Imposing a *pressure*
    # rather than a velocity is what lets the sprue choke meter the flow, which
    # is the behaviour Foseco's own gating guidance asks for.
    head_m = float(inches_to_meters(float(getattr(filling, "initial_head_in", 4.0))))
    head_lattice = head_m / dx_m
    inlet_density = REFERENCE_DENSITY * (
        1.0 + units.gravity_lattice * head_lattice / LATTICE_CS2
    )

    # What the inlet imposes -- and what it must not.
    #
    # The original source rebuilt the inlet populations from ``state.u``, the
    # *Guo force-corrected* velocity, which carries half of that step's
    # gravitational impulse by construction (see guo_velocity's docstring).
    # Feeding it back into equilibrium() re-imposed that half-impulse every
    # step with nothing to remove it, so the inlet gained 0.5*g_lattice of
    # velocity per step forever and took the whole solve with it.
    #
    # Its replacement -- a constant *prescribed* velocity, Torricelli's
    # sqrt(2 g h) in head mode -- could not accumulate, and could not work
    # either. A fixed-velocity equilibrium pin cannot feel back-pressure: it
    # overwrites whatever the domain streams back into it. Instrumented on
    # Test_Model/Filling_Test at 0.20 in voxels, coarsening 2, that produced
    # three measurements, and they are the reason this design is gone:
    #
    #   * Delivery collapsed. Net mass added by the inlet fell from 0.110
    #     lattice-mass/step -- the full prescribed rate -- to 0.006 by step
    #     28,000, about 6 %, and then went negative: the pin was deleting
    #     metal that had streamed back into it.
    #   * The column starved. With delivery choked, the whole sprue sat in
    #     suction: mean rho 0.99995 at the sprue bottom against a hydrostatic
    #     requirement of ~1.0028. No head developed, so nothing pushed the
    #     metal onward, and 428 casting cells crept at eps ~ 0.5, about 50x
    #     slower than the geometry allows.
    #   * The inlet seeded a staggered oscillation. Per-cell mass in the top
    #     sprue layers reached +-218 against a physical range of [0, ~1], u_z
    #     alternating sign per z-layer, total wet mass swinging 170 <-> 2,815
    #     between 200-step samples, until the divergence guard fired at step
    #     126,400.
    #
    # So head mode imposes PRESSURE and lets the velocity be whatever the flow
    # field itself produces. The pressure is the metallostatic head of a
    # pouring basin held topped up, which is exactly what ``initial_head_in``
    # has always described. Run freely, gravity accelerates the column and the
    # inlet feeds it; back the downstream up and the measured velocity falls,
    # so delivery falls with it -- what a real basin does, and what a
    # prescribed velocity cannot do. Momentum cannot accumulate, because the
    # velocity is re-measured from the populations every step and the rebuild
    # adds no force term. The measurement is the RAW first moment, deliberately
    # without the Guo half-force correction: that correction is precisely the
    # term that made the original a pump.
    #
    # That is the *momentum* boundary, and it stays. What it could not do is
    # deliver metal, and this is the third measurement, taken on the assemblies
    # named below:
    #
    #   * Net delivery was 0.0174 lattice-mass/step -- about 1.26 lb/s -- on
    #     BOTH Test_Model/Full_Gating_Test (22 lb) and the 459 lb Test 2, at
    #     every head and every grid tried. A number that does not move when the
    #     head, the mouth area and the model all move is not a physical flow
    #     rate; it is a numerical residue. Torricelli through the same mouths
    #     is 25-50x larger.
    #
    # The mechanism, and why the pressure BC alone cannot fix it. Delivery is
    # accounted -- correctly -- as the mass the pin puts back into the inlet
    # cells, which under Koerner's weight-1 exchange is exactly their density
    # change over the step. The inlet layer is the top of the metal: nothing
    # streams in from above it, so whatever leaves through its underside is
    # what the pin must replace, and delivery is therefore identically
    # ``rho * u_z`` through the mouth. That identity is sound. The problem is
    # what sets ``u_z``: the inlet layer's momentum after streaming is
    # dominated by exchange with the closed boundary above it -- its own
    # up-going populations bounce straight back down, and it receives only
    # what the layer below sends up. It never sees a hydrostatic column above
    # it, because there is no column above it, only a boundary condition. So
    # the layer sits at u_z ~ 5e-4 lattice against a Torricelli 0.02, the
    # sprue below it is fed a trickle, the front spreads as interface cells
    # that never reach eps = 1, and Koerner's eps-averaged exchange weight
    # then throttles the trickle further. Starvation is self-reinforcing:
    # low delivery -> mist -> low exchange weight -> low delivery. On a small
    # model 1.26 lb/s still finishes; on Test 2 the mist sat at eps ~ 0.37 for
    # 13 s and died in the negative-population cascade.
    #
    # The fix keeps the pressure boundary and adds an explicit, metered mass
    # source -- which is what a pouring basin *is*. See the delivery block in
    # the step loop: the target rate below is Torricelli through the measured
    # mouth, and the loop places at most that much metal per step into the
    # inlet cells and the few cells of throat directly beneath them, and only
    # into the room they actually have. Mass is added; momentum is not, so it
    # cannot become the pump the original was. Back-pressure throttles it
    # exactly: when the throat is full there is no room, the top-up is zero,
    # and delivery falls back to whatever the column's own dynamics carry.
    #
    # Velocity mode prescribes its constant velocity from the pour rate as
    # before, and is metered against that rate rather than against Torricelli:
    # there the user is asserting a delivery rate.
    inlet_area_m2 = float(inlet.shape[0]) * dx_m * dx_m
    inlet_mode = int(getattr(filling, "inlet_mode_code", 0))
    ix, iy, iz = inlet[:, 0], inlet[:, 1], inlet[:, 2]
    # The flow area is the mouth's horizontal cross-section -- the top layer of
    # the inlet -- not every cell the resolver returned. _expand_inlet_to_mouth
    # floods one layer, so the two agree except where the pour point was
    # clicked on a flank and dragged a few seed cells down with it.
    mouth_cells = int(np.count_nonzero(inlet[:, 2] == int(inlet[:, 2].max())))
    mouth_area_m2 = float(max(mouth_cells, 1)) * dx_m * dx_m
    # Hoisted: head mode needs it for the square-root cross-check, and the
    # velocity branch for its own rate.
    weight_lb = pour_weight_lb(voxel_mesh, alloy)
    recommended_lb_s = recommended_pour_rate_lb_s(weight_lb)
    u_inlet_physical = 0.0
    inlet_mode_label = ""
    if inlet_mode == 1:
        # 0 means "derive", the same convention pour_time_s and
        # pour_temperature_c already use. The square-root rule ties the rate to
        # this casting's own weight; a fixed number cannot, and 5 lb/s is
        # wrong for almost every casting that is not about 25 lb.
        pour_rate_lb_s = float(getattr(filling, "pour_rate_lb_s", 0.0) or 0.0)
        if pour_rate_lb_s <= 0.0:
            pour_rate_lb_s = recommended_lb_s
            rate_source = f"sqrt of {weight_lb:.0f} lb"
        else:
            rate_source = "set"
        if pour_rate_lb_s <= 0.0:
            # No part types, so no weight to take a root of. Fall back to the
            # head rather than dividing by zero and reporting a still pour.
            warnings.append(
                "velocity inlet requested but the pour weight could not be "
                "measured; falling back to the metallostatic head."
            )
            inlet_mode = 0
        else:
            pour_rate_kg_s = pour_rate_lb_s * 0.45359237
            u_inlet_physical = pour_rate_kg_s / max(
                float(alloy.density_liquid) * inlet_area_m2, 1e-9
            )
            inlet_mode_label = (
                f"velocity ({pour_rate_lb_s:.1f} lb/s, {rate_source}; "
                f"{pour_rate_lb_s:.1f} s pour)"
            )
    if inlet_mode != 1:
        inlet_mode_label = f"head ({head_m * 39.3701:.1f} in)"

    # The filter ceiling, applied to BOTH inlet modes. A filter restricts the
    # flow path itself, so it binds whatever is driving the metal -- a deep
    # head does not entitle the pour to exceed the filter's rating, it just
    # means the sprue backs up. In velocity mode it lowers the prescribed
    # velocity. In head mode nothing is prescribed to lower, so it lowers the
    # *cap* the measured velocity is clamped against instead: the restriction
    # is still modelled, it just binds a measurement rather than a setting.
    flow_limit_kg_s, flow_limit_note = filter_flow_limit_kg_s(
        filter_zones, alloy, int(getattr(filling, "pour_mode_code", 0))
    )
    if flow_limit_note:
        warnings.append(flow_limit_note)
    u_filter_max: Optional[float] = None
    if flow_limit_kg_s is not None:
        u_filter_max = flow_limit_kg_s / max(
            float(alloy.density_liquid) * inlet_area_m2, 1e-9
        )
        if inlet_mode == 1 and u_inlet_physical > u_filter_max:
            requested_lb_s = (
                u_inlet_physical * float(alloy.density_liquid) * inlet_area_m2
                / 0.45359237
            )
            warnings.append(
                f"pour rate limited by the filter: {requested_lb_s:.1f} lb/s "
                f"requested, {flow_limit_kg_s / 0.45359237:.1f} lb/s rated "
                f"across {len(filter_zones)} filter(s). Running at the rating. "
                f"Above it a ceramic foam filter blinds off or breaks up "
                f"rather than throttling."
            )
            u_inlet_physical = u_filter_max
            inlet_mode_label += ", filter-limited"

    # The Mach guard the timestep was sized for, lowered by the filter rating
    # where one applies. Nothing the inlet writes may exceed it: a boundary
    # condition that violates the solver's own stability limit is not a
    # boundary condition.
    u_cap_lattice = float(units.u_max_lattice)
    if u_filter_max is not None:
        u_cap_lattice = min(
            u_cap_lattice,
            float(units.to_lattice_velocity(np.asarray(u_filter_max, dtype=np.float64))),
        )
        if inlet_mode != 1 and u_cap_lattice < float(units.u_max_lattice):
            inlet_mode_label += ", filter-limited"

    # ---- The delivery rate the pour is metered against --------------------
    #
    # Physics, not numerical residue. In head mode that is Torricelli's
    # efflux from the configured head through the *measured* mouth, with a
    # discharge coefficient for the vena contracta and entrance loss --
    # Q = Cd A sqrt(2 g h) -- which is the same expression a foundry uses to
    # size a sprue. In velocity mode it is the rate the user asserted, which
    # is already in hand as u_inlet_physical through the same area.
    #
    # Reported against the square-root rule (a W-pound casting pours in
    # sqrt(W) seconds) as a cross-check, because the two are independent
    # statements about the same pour: Torricelli says what the rigging can
    # pass, sqrt(W) says what the casting wants. They disagreeing is
    # information -- an under-sized sprue, or an over-sized one -- so both are
    # logged rather than silently reconciled.
    lb_per_lattice_mass = (
        float(alloy.density_liquid) * (dx_m ** 3)
        / float(REFERENCE_DENSITY) / 0.45359237
    )
    torricelli_m_s = float(np.sqrt(2.0 * 9.81 * max(head_m, 0.0)))
    if inlet_mode == 1:
        target_kg_s = (
            u_inlet_physical * float(alloy.density_liquid) * inlet_area_m2
        )
        target_source = "prescribed"
    else:
        pour_rate_override_lb_s = float(
            getattr(filling, "pour_rate_lb_s", 0.0) or 0.0
        )
        if pour_rate_override_lb_s > 0.0:
            # An explicit rate is an assertion about the pour, and a boundary
            # that quietly ignored it would be the same fault the old
            # self-metering inlet had, pointed the other way.
            target_kg_s = pour_rate_override_lb_s * 0.45359237
            target_source = "set"
        else:
            # Two independent statements about the same pour, and the pour
            # runs at the smaller.
            #
            # Torricelli through the measured mouth, Q = Cd A sqrt(2 g h), is
            # what the *rigging* can pass -- the ceiling, set by the basin and
            # the mouth it sits on. The square-root rule, sqrt(W) lb/s, is what
            # the *casting* wants: a W-pound casting is poured in sqrt(W)
            # seconds. A foundry pours at the second and uses the first to
            # check the sprue is big enough, and metering at the ceiling
            # instead is how a pour is over-driven.
            #
            # Both test rigs have mouths well above their castings' needs,
            # because a sprue mouth is sized generously and the real choke is
            # downstream: 29.6 against 4.4 lb/s on Full_Gating_Test, 44.1
            # against 20.8 on Test 2. Metering Test 2 at its 44.1 lb/s ceiling
            # drove the free-surface reconstruction into persistent negative
            # populations and the precursor guard aborted it at step 9,800.
            mouth_ceiling_lb_s = (
                INLET_DISCHARGE_COEFFICIENT
                * float(alloy.density_liquid)
                * mouth_area_m2
                * torricelli_m_s
                / 0.45359237
            )
            if recommended_lb_s > 0.0 and recommended_lb_s < mouth_ceiling_lb_s:
                target_kg_s = recommended_lb_s * 0.45359237
                target_source = (
                    f"sqrt of {weight_lb:.0f} lb; the {mouth_cells:,}-cell "
                    f"mouth could pass {mouth_ceiling_lb_s:.1f}"
                )
            else:
                target_kg_s = mouth_ceiling_lb_s * 0.45359237
                target_source = (
                    f"Cd {INLET_DISCHARGE_COEFFICIENT:.2f} x "
                    f"{mouth_cells:,}-cell mouth x sqrt(2gh)"
                )
                if recommended_lb_s > 0.0:
                    warnings.append(
                        f"the mouth cannot pass the rate this casting wants: "
                        f"{mouth_ceiling_lb_s:.1f} lb/s through the "
                        f"{mouth_cells:,}-cell mouth at "
                        f"{head_m * 39.3701:.1f} in of head, against "
                        f"{recommended_lb_s:.1f} lb/s from the square-root "
                        f"rule for {weight_lb:.0f} lb. Pouring at what the "
                        f"rigging passes -- the fill will be slower than the "
                        f"casting wants, which is where cold shuts come from. "
                        f"Widen the sprue mouth or raise the head."
                    )

    # The filter rating binds the target the same way it binds the velocity
    # cap: a deep head does not entitle the pour to exceed what the filter can
    # pass.
    if flow_limit_kg_s is not None and target_kg_s > flow_limit_kg_s:
        if inlet_mode != 1:
            # Velocity mode has already warned about this above.
            warnings.append(
                f"delivery limited by the filter: the head would drive "
                f"{target_kg_s / 0.45359237:.1f} lb/s through the mouth, "
                f"{flow_limit_kg_s / 0.45359237:.1f} lb/s is rated across "
                f"{len(filter_zones)} filter(s). Metering at the rating."
            )
            inlet_mode_label += ", filter-limited"
        target_kg_s = float(flow_limit_kg_s)
        target_source += ", filter-rated"

    # The Mach guard binds it last, and honestly. Delivering Q through an area
    # A implies a mean entry velocity Q / (rho A); if that is above the guard
    # the timestep was sized for, this grid cannot carry the pour, and saying
    # so is the only defensible answer. Silently delivering it anyway would
    # reintroduce exactly the atomised jet _expand_inlet_to_mouth was written
    # to remove.
    u_target_lattice = float(
        units.to_lattice_velocity(
            np.asarray(
                target_kg_s / max(float(alloy.density_liquid) * mouth_area_m2, 1e-9),
                dtype=np.float64,
            )
        )
    )
    if u_target_lattice > u_cap_lattice:
        capped_kg_s = target_kg_s * (u_cap_lattice / max(u_target_lattice, 1e-12))
        warnings.append(
            f"pour rate unachievable at this resolution: delivering "
            f"{target_kg_s / 0.45359237:.1f} lb/s through the "
            f"{mouth_cells:,}-cell mouth ({mouth_area_m2 * 1550.0:.2f} in^2) "
            f"implies {u_target_lattice:.4f} lattice velocity against a guard "
            f"of {u_cap_lattice:.4f}. Metering at "
            f"{capped_kg_s / 0.45359237:.1f} lb/s instead -- the fill will run "
            f"slower than the real pour. Reduce fill_grid_coarsening or voxel "
            f"size to widen the mouth in cells, or raise "
            f"lattice_velocity_max."
        )
        target_kg_s = capped_kg_s
        u_target_lattice = u_cap_lattice
        target_source += ", Mach-capped"

    # What the basin can supply through the mouth, whatever the domain pulls:
    # Torricelli with the discharge coefficient, under the Mach guard. Fixed
    # for the run -- the head does not change -- and used as the hard ceiling
    # on the head boundary's imposed entry velocity.
    u_mouth_ceiling_lattice = min(
        u_cap_lattice,
        float(units.to_lattice_velocity(
            np.asarray(
                INLET_DISCHARGE_COEFFICIENT * torricelli_m_s, dtype=np.float64
            )
        )),
    )

    target_lb_s = target_kg_s / 0.45359237
    # Lattice mass the metered source may add per step. One lattice mass unit
    # at REFERENCE_DENSITY is one cell volume of liquid metal, which is what
    # lb_per_lattice_mass inverts -- so this and the poured counter are in the
    # same units by construction and the ledger closes exactly.
    target_mass_per_step = (
        target_lb_s * float(units.dt_s) / max(lb_per_lattice_mass, 1e-30)
    )

    inlet_rho = np.full(inlet.shape[0], inlet_density, dtype=np.float64)
    inlet_u = np.zeros((3, inlet.shape[0]), dtype=np.float64)
    if inlet_mode == 1:
        u_inlet_lattice = float(units.to_lattice_velocity(u_inlet_physical))
        if u_inlet_lattice > units.u_max_lattice:
            warnings.append(
                f"inlet velocity {u_inlet_physical:.2f} m/s exceeds the Mach guard "
                f"the timestep was sized for; clamped to "
                f"{units.u_max_lattice * units.velocity_scale:.2f} m/s. Raise "
                f"lattice_velocity_max or lower the pour rate."
            )
            u_inlet_lattice = float(units.u_max_lattice)
        # Gravity points -Z (physics.filling.step's default), and metal enters
        # falling, so the inlet drives along it.
        inlet_u[2, :] = -u_inlet_lattice
        # Velocity mode's mass pin is unchanged: a whole cell's worth, every
        # step, matching the constant delivery it is asserting.
        inlet_full_mass = np.full(
            inlet.shape[0], float(REFERENCE_DENSITY), dtype=np.float64
        )
        emit(
            f"Filling: inlet {inlet.shape[0]:,} cells{inlet_widening}, "
            f"{inlet_mode_label}, "
            f"{u_inlet_physical:.2f} m/s ({u_inlet_lattice:.4f} lattice); "
            f"metering {target_lb_s:.1f} lb/s ({target_source}), "
            f"sqrt-rule cross-check {recommended_lb_s:.1f} lb/s "
            f"for {weight_lb:.0f} lb"
        )
    else:
        # A full cell holds rho * porosity -- apply_transitions' own definition
        # of full. Pinning REFERENCE_DENSITY instead would read as under-filled
        # the moment the free-surface pass looked at an inlet inside a filter.
        inlet_full_mass = inlet_rho * np.asarray(
            state.porous.porosity[ix, iy, iz], dtype=np.float64
        )
        emit(
            f"Filling: inlet {inlet.shape[0]:,} cells{inlet_widening}, "
            f"{inlet_mode_label} "
            f"pressure boundary, rho {inlet_density:.5f} "
            f"(+{(inlet_density / REFERENCE_DENSITY - 1.0) * 100.0:.3f}% over "
            f"reference); velocity follows the flow, capped at "
            f"{u_cap_lattice * units.velocity_scale:.2f} m/s "
            f"({u_cap_lattice:.4f} lattice); metering {target_lb_s:.1f} lb/s "
            f"({target_source}, {torricelli_m_s:.2f} m/s efflux, "
            f"{u_target_lattice:.4f} lattice), sqrt-rule cross-check "
            f"{recommended_lb_s:.1f} lb/s for {weight_lb:.0f} lb"
        )
        # A mouth larger than the casting needs is the normal case, and not a
        # defect: the sprue tapers and the real choke is downstream. It is
        # reconciled in the target above rather than warned about -- the pour
        # simply runs at the square-root rate. Only the other direction, a
        # mouth that cannot pass what the casting wants, gets a warning.

    # Whether the head can reach the top of the metal at all.
    #
    # Metal does not climb above the free surface of the basin feeding it, so
    # a pour point plus head sitting below the highest metal cell cannot
    # finish the casting however long the solve runs. Without this the run
    # spends its whole budget proving that and then reports a time cap, which
    # reads as "needed longer" when it means "cannot get there from here".
    # Head mode only: velocity mode is a pump, and a pump has no free surface.
    if inlet_mode != 1:
        solve_dx_in = mapping.dx_in * coarsening
        # Padded z index, so the pad offset comes back off before the index
        # becomes a height: index pad_lo is the original grid's z = 0.
        z_centres_in = float(mapping.origin_in[2]) + (
            (np.arange(solve_types.shape[2], dtype=np.float64) - pad_lo)
            * coarsening
            + (coarsening - 1) / 2.0
        ) * mapping.dx_in
        metal_top_z_in = float(
            z_centres_in[np.nonzero(np.any(metal, axis=(0, 1)))[0].max()]
        )
        inlet_z_in = float(z_centres_in[iz].max())
        head_reach_in = inlet_z_in + head_m * 39.3701
        if head_reach_in < metal_top_z_in + solve_dx_in:
            warnings.append(
                f"the metallostatic head cannot reach the top of the metal, so "
                f"the fill physically cannot complete: the pour point sits at "
                f"z = {inlet_z_in:.2f} in and {head_m * 39.3701:.1f} in of head "
                f"reaches {head_reach_in:.2f} in, while metal extends to "
                f"z = {metal_top_z_in:.2f} in. Raise the pour point or "
                f"increase initial_head_in."
            )

    # An under-resolved choke is not a small inaccuracy -- it is the
    # difference between a fill and a sealed channel -- and it is measured
    # here rather than left for the user to infer from a misrun count.
    #
    # Measured on solve_types: the grid the solver runs on, not the mesh it
    # was coarsened from. A gate that is four cells across the mesh is two
    # across a coarsening of 2, and it is the two that seals.
    #
    # Reported per part, and warned about per part. One number for "the gating
    # system" cannot say which body the metal will not cross, and the answer
    # is usually the gate while the sprue and runner are fine.
    # Already measured on exactly this grid by choose_fill_coarsening, which
    # had to measure it to decide what the grid should be.
    choke_by_part = coarsening_decision.by_part
    choke_thickness_cells = (
        min(choke_by_part.values()) if choke_by_part else 0.0
    )
    narrowest_code = narrowest_choke_part_code(choke_by_part)
    if choke_by_part:
        emit(
            "Filling: channel thickness on the "
            f"{dx_m * 1000:.1f} mm fill grid -- "
            + ", ".join(
                f"{_METAL_PATH_LABELS[code]} "
                + (
                    "/".join(
                        f"{value:.1f}"
                        for value in coarsening_decision.components_by_part[code]
                    )
                    if len(coarsening_decision.components_by_part.get(code, ()))
                    > 1
                    else f"{choke_by_part[code]:.1f}"
                )
                + " cells"
                for code in _METAL_PATH_CODES
                if code in choke_by_part
            )
            + (
                f" (narrowest: {_METAL_PATH_LABELS[narrowest_code]})"
                if narrowest_code is not None
                else ""
            )
        )
    choke_components = coarsening_decision.components_by_part
    for code in _METAL_PATH_CODES:
        thickness = choke_by_part.get(code)
        if thickness is None or thickness <= 0.0:
            continue
        # Name the body, not just the part code. A code covers every body of
        # that type -- Test 2 has two gates and two filters -- and the number
        # being warned about is the *thinnest* of them, so saying "the gate"
        # when one gate is fine and the other is sealed sends the reader to
        # the wrong geometry. See measure_choke_thickness_by_part.
        bodies = choke_components.get(code, ())
        label = _METAL_PATH_LABELS[code]
        if len(bodies) > 1:
            label = f"thinnest of the {len(bodies)} {label} bodies"
        inches = thickness * dx_m * 39.3701
        if thickness <= MIN_CHOKE_THICKNESS_CELLS:
            warnings.append(
                f"the {label} is {thickness:.1f} cells thick on a "
                f"{dx_m * 1000:.1f} mm fill grid ({inches:.2f} in) -- at or "
                f"below the {MIN_CHOKE_THICKNESS_CELLS:.0f} cells a channel "
                f"needs before it has any interior. Halfway bounce-back seals "
                f"it: this grid cannot resolve the {label}, and the fill "
                f"through it cannot be trusted whatever this run reports. "
                f"Reduce voxel size or fill_grid_coarsening."
            )
        elif thickness <= MARGINAL_CHOKE_THICKNESS_CELLS:
            warnings.append(
                f"the {label} resolves to {thickness:.1f} cells across on a "
                f"{dx_m * 1000:.1f} mm fill grid ({inches:.2f} in), which is "
                f"the thinnest channel this solver has ever carried a complete "
                f"fill through -- and it carries it at the edge of the velocity "
                f"range the timestep was sized for. Treat the flow through the "
                f"{label} as marginal, not as a result; if it matters to the "
                f"answer, resolve it and rerun."
            )

    # Bodies too small to judge the grid by, said out loud rather than dropped.
    # A fragment always measures 1.0 and can never be resolved, so letting it
    # drive the decision pins the grid at coarsening 1 forever -- see
    # MIN_CHOKE_BODY_CELLS. It is a modelling observation, not a resolution one.
    for code, sizes in coarsening_decision.fragments_by_part.items():
        label = _METAL_PATH_LABELS.get(code, "gating")
        listed = ", ".join(f"{size:,}" for size in sizes)
        warnings.append(
            f"{len(sizes)} disconnected "
            f"{'fragment' if len(sizes) == 1 else 'fragments'} of {label} "
            f"({listed} cells) "
            f"{'was' if len(sizes) == 1 else 'were'} ignored by the choke "
            f"guard: below {MIN_CHOKE_BODY_CELLS} cells a body has no "
            f"cross-section in any orientation, and it would read as sealed at "
            f"every resolution and pin the grid at coarsening 1 for good. The "
            f"{label}'s reported thickness is its main body. Check the part "
            f"assignment if you did not expect a detached "
            f"{'piece' if len(sizes) == 1 else 'pieces'} of {label}."
        )

    # F3: air entrapment. The vent mask is reassembled at every periodic
    # check (see below) because its "open-air gas" component is a live
    # snapshot -- built once here, "static_vent" only covers the parts that do
    # not change as the fill progresses (open riser tops, the inlet).
    static_vent = _static_vent_mask(
        solve_types, inlet, bool(getattr(filling, "vent_risers", 1))
    )
    # Where the cavity is open to air, for the entrapment check.
    #
    # Two parts. The headspace the pad opens above top-face metal, which *is*
    # the atmosphere over the risers and any non-poured feature that reaches
    # the top; and the original grid's ``+z`` plane, which covers the ring of
    # top-face columns the headspace deliberately skips beside the pour mouth.
    # Built on the original shape and embedded at the pad offset, because on the
    # padded grid every array face is obstacle and so never GAS -- measured: 0
    # non-obstacle cells, against 2,247 riser-top vents _static_vent_mask still
    # finds by type.
    #
    # The five other faces are deliberately NOT here; see _open_air_mask for
    # why they used to be and why counting them was inverting the detector's
    # answer on the geometry it exists to flag.
    open_air_mask = np.zeros(solve_types.shape, dtype=bool)
    open_air_mask[core] = _open_air_mask(core_shape)
    open_air_mask |= headspace
    open_air_mask &= ~obstacle
    entrapment_check_interval = max(
        1, int(getattr(filling, "entrapment_check_interval", 50))
    )
    # Peak rather than final: a pocket can be entrapped at one moment and
    # vent later as the geometry around it changes, so checking only at the
    # end understates the risk (design doc section 6.4).
    peak_entrapped_voxels = 0
    vent_warning: Optional[str] = None

    report_every = max(1, max_steps // 20)
    # Captured on an adaptive stride, not on max_steps / FILL_ANIMATION_FRAMES.
    # A run almost never uses its whole budget -- a good pour finishes in a
    # few seconds of a 60 s cap, and a stalled one gives up early -- so a
    # stride sized for the budget yields a handful of frames exactly when the
    # animation matters most. Instead sample finely and halve the stride
    # whenever the buffer doubles, which keeps the frames evenly spaced across
    # however long the run actually turns out to be, for a bounded cost.
    frame_every = max(1, max_steps // (FILL_ANIMATION_FRAMES * 4))
    frames: List[FillFrame] = []

    def capture_frame(step_index: int) -> None:
        """Gather the live free surface and velocity onto voxels."""
        eps = np.clip(fluid_fraction(state), 0.0, 1.0)
        eps[state.flags == FLAG_OBSTACLE] = 0.0
        speed = np.sqrt(np.sum(np.square(state.u.astype(np.float64)), axis=0))
        speed *= units.velocity_scale
        frames.append(
            FillFrame(
                time_s=step_index * units.dt_s,
                fill_fraction=to_voxels(eps).astype(np.float32),
                speed_m_s=to_voxels(speed).astype(np.float32),
            )
        )

    taken = 0
    metal_total = int(metal.sum())
    # Mass the metal region holds when full. Full is rho * porosity -- the same
    # definition apply_transitions converts on -- so inside a filter this sits
    # below the cell count, which is correct: a filter cell full of metal holds
    # only its pore volume.
    metal_full_mass = float(
        np.sum(
            float(REFERENCE_DENSITY)
            * np.asarray(state.porous.porosity, dtype=np.float64)[metal]
        )
    )
    # Net mass the inlet has injected, in lattice mass units, accumulated at
    # each pin: what the pin writes, minus what was there to overwrite. The
    # conversion to kilograms is total_mass_kg's: one lattice mass unit at
    # REFERENCE_DENSITY is one cell volume of liquid metal.
    poured = 0.0
    poured_lb_per_lattice_mass = lb_per_lattice_mass
    # Hoisted out of the loop: the inlet's raw first moment is measured every
    # step from a (19, n_inlet) slice, and n_inlet is a handful of cells.
    lattice_c_t = LATTICE_VELOCITIES.T.astype(np.float64)
    inlet_u_measured = np.zeros((3, inlet.shape[0]), dtype=np.float64)

    # The cells the inlet streams into, so the pin can keep them wet.
    #
    # Pinning FLAG_FLUID by hand puts the inlet outside apply_transitions'
    # bookkeeping, and with it outside the scheme's central invariant: no FLUID
    # cell is ever directly adjacent to a GAS cell. Measured on a 2x2x12 column
    # -- the cell above the inlet drained to gas, streaming zeroed it, and the
    # inlet stopped receiving every population arriving with c_z = -1, one
    # sixth of the weight. The measured z-momentum jumped from -0.009 (falling)
    # to +0.147 (rising) in one step and the whole column reversed at the Mach
    # cap. Rolled with the same periodic wrap the streaming kernel uses, so the
    # halo is exactly the set of cells the inlet can stream into.
    inlet_mask = np.zeros(solve_types.shape, dtype=bool)
    inlet_mask[ix, iy, iz] = True
    inlet_halo = np.zeros(solve_types.shape, dtype=bool)
    for _c in LATTICE_VELOCITIES[1:]:
        inlet_halo |= np.roll(
            inlet_mask, shift=(int(_c[0]), int(_c[1]), int(_c[2])), axis=(0, 1, 2)
        )
    inlet_halo &= ~inlet_mask & ~obstacle
    hx, hy, hz = np.nonzero(inlet_halo)

    # The throat the metered source may pour into, layer by layer.
    #
    # Grown straight down from the mouth through metal only, so it is the bore
    # directly under the inlet and stops at the first floor -- it cannot jump
    # an obstacle into a cavity that happens to sit below. Each layer is kept
    # separately because the source fills them in order, nearest the mouth
    # first: spreading the same mass evenly over three layers is how a mist
    # forms, and a mist is the failure this whole block exists to prevent.
    #
    # Only INTERFACE cells in these layers ever have room. Under weight-1
    # exchange a FLUID cell's mass tracks its density identically, so a
    # draining FLUID cell's "full" falls with it and its deficit is zero by
    # construction. That is the correct behaviour, not a limitation: once the
    # throat is running full the column's own dynamics carry the metal, the
    # top-up finds no room, and delivery falls back to what the inlet pin
    # measures -- which for a coherent full column IS the physical flux
    # ``rho u A``. The source bootstraps and sustains the column; it does not
    # replace it.
    delivery_layers: List[Tuple[Tuple[np.ndarray, ...], np.ndarray]] = []
    _frontier = inlet_mask
    _claimed = inlet_mask.copy()
    for _ in range(INLET_DELIVERY_DEPTH_CELLS):
        _below = np.zeros_like(_frontier)
        _below[:, :, :-1] = _frontier[:, :, 1:]
        _below &= metal & ~obstacle & ~_claimed
        if not _below.any():
            break
        _claimed |= _below
        _idx = np.nonzero(_below)
        delivery_layers.append(
            (_idx, np.asarray(state.porous.porosity[_idx], dtype=np.float64))
        )
        _frontier = _below

    status = STATUS_TIME_CAPPED
    diverged_at_step = 0
    flat_reports = 0
    last_reached = -1
    best_mass_fraction = -1.0
    # Rolling window of "did this check see a negative population?", the
    # instability precursor -- see NEGATIVE_POPULATION_WINDOW.
    negative_history: List[int] = []
    # Consecutive divergence checks the negative-population cascade condition
    # has held -- see NEGATIVE_POPULATION_CASCADE_CHECKS.
    cascade_streak = 0
    # Consecutive divergence checks that found a FLUID cell outside the fill
    # fraction band -- see FLUID_FILL_FRACTION_CHECKS.
    fill_fraction_breaches = 0
    # Delivery over the previous step, which is what the head boundary's
    # rate controller closes the loop on. Starts at zero, i.e. fully starved,
    # so the controller opens the boundary up from rest.
    delivered_previous = 0.0
    # The head boundary's driving pressure difference, low-passed across
    # steps, and the entry velocity it produces. Started at the full head:
    # before any metal is below the mouth the throat is at atmospheric, which
    # is what the filter should already be holding.
    inlet_drive_head = inlet_density - float(REFERENCE_DENSITY)
    u_drive = 0.0
    # Where its back-pressure is read: the first layer of throat below the
    # mouth, or the inlet's own stream halo when the mouth sits on a floor.
    thx, thy, thz = (
        delivery_layers[0][0] if delivery_layers else (hx, hy, hz)
    )
    # Delivery is differenced between reports rather than averaged from the
    # start, so a rate that collapses shows up as a collapse instead of being
    # hidden under the healthy part of the run.
    poured_at_last_report = 0.0
    step_at_last_report = 0
    for taken in range(1, max_steps + 1):
        # Hold the inlet full. What that means differs by mode, and the
        # difference is the whole redesign -- see the block above inlet_area_m2
        # for the three measurements that forced it.
        state.flags[ix, iy, iz] = FLAG_FLUID
        # Restore the interface skin the pin needs and cannot get from
        # apply_transitions -- see the inlet_halo derivation above. Checked on
        # the handful of halo cells first; the full-domain mask is only built
        # on the steps where something has actually gone dry.
        if np.any(state.flags[hx, hy, hz] == FLAG_GAS):
            ensure_interface_skin(state, inlet_halo)
        if inlet_mode == 1:
            # Prescribed velocity: constant, so it cannot accumulate.
            state.f[:, ix, iy, iz] = equilibrium(inlet_rho, inlet_u).astype(np.float32)
        else:
            # Head boundary: the head DIFFERENCE drives the entry velocity.
            #
            # Reading the inlet cells' own raw first moment back into their
            # equilibrium, as the previous boundary did alone, does not
            # measure the flow's velocity, and the instrumented 4x4x40 column
            # that proved it is
            # worth keeping: with the head at 4 in the inlet layer measured
            # u_z = -0.002 against a Torricelli 0.0202, and the cell five below
            # it -0.006 -- an order of magnitude short, for 12,000 steps, while
            # the column ran at 26 % wet and never filled.
            #
            # The reason is the lid. The inlet layer is the top of the metal
            # and the mold sits directly above it, so every population it
            # sends upward bounces straight back down, and the only momentum
            # it receives from below is that cell's *upward* group. Those two
            # very nearly cancel, so the layer's measured momentum is pinned
            # near zero every step no matter what the column beneath it is
            # doing. Delivery is identically ``rho u A`` at that layer, so a
            # velocity held near zero is a delivery held near zero: the flat
            # 0.0174 lattice-mass/step that appeared at every head and every
            # model size. It was never a flow rate; it was the lid.
            #
            # So the velocity is *driven* as well as measured, from the
            # pressure difference the boundary actually has to work against:
            #
            #     u = Cd sqrt(2 c_s^2 (rho_head - rho_throat) / rho_0)
            #
            # -- Bernoulli through an orifice, which reduces to Torricelli's
            # sqrt(2 g h) when the throat is at atmospheric and to zero when
            # the throat has backed up to the head. rho_throat is measured on
            # the wet cells immediately below the mouth, so back-pressure is a
            # measurement and the throttle is closed-loop: fill the column to
            # the head and delivery stops on its own, which is the one thing a
            # constant prescribed velocity could never do. That expression is
            # the *ceiling* -- see the rate controller below, which trims the
            # velocity under it to hit the target mass rate.
            #
            # It cannot accumulate either. u is a function of a measured
            # density difference and a constant, bounded by the Mach guard;
            # nothing about last step's velocity enters it, so the Guo
            # half-force that made the original inlet a pump has no path back
            # in. The same 4x4x40 column under this boundary delivers 97 % of
            # target, fills all 640 cells by step 4,000, and then throttles
            # itself to 0.0 % and holds there with no negative populations.
            throat_wet = state.flags[thx, thy, thz] >= FLAG_INTERFACE
            rho_throat = (
                float(
                    np.asarray(
                        state.rho[thx, thy, thz], dtype=np.float64
                    )[throat_wet].mean()
                )
                if throat_wet.any()
                else float(REFERENCE_DENSITY)
            )
            # Low-passed BEFORE the square root, and that order is the whole
            # difference between a boundary that shuts off and one that does
            # not. ``max(0, .)`` rectifies: filter the velocity and the
            # driving difference's symmetric population noise passes through
            # the clamp as a train of one-sided spikes whose average is
            # positive, so the boundary keeps pushing on a column that has
            # already backed up. Measured on Full_Gating_Test with the filter
            # on the wrong side: max|u| held at 0.040 for 231,000 steps and
            # the fill oscillated between 1,050 and 1,290 FLUID cells forever
            # instead of closing, stalling at 79 %. Filtering the difference
            # first, the same boundary reads a clean zero once the throat is
            # up to head and delivery settles inside +-0.1 % of nothing.
            inlet_drive_head += INLET_VELOCITY_BETA * (
                (inlet_density - rho_throat) - inlet_drive_head
            )
            # What the head can do is the CEILING on the entry velocity, not
            # the entry velocity itself.
            #
            # Imposing the Bernoulli velocity directly pours at whatever the
            # mouth happens to pass, which on both test rigs is far more than
            # the casting wants: Full_Gating_Test delivered 4.8 lb/s whatever
            # target it was given, and Test 2 was driven hard enough that the
            # free-surface reconstruction went into persistent negatives and
            # the precursor guard aborted it at step 9,600 of 873,186. It also
            # made the reported "x of y target" a fiction, because y was not
            # controlling anything.
            u_ceiling = min(
                u_cap_lattice,
                INLET_DISCHARGE_COEFFICIENT * float(np.sqrt(
                    2.0 * LATTICE_CS2
                    * max(0.0, inlet_drive_head)
                    / float(REFERENCE_DENSITY)
                )),
            )
            # So the velocity is trimmed to hit the target *mass* rate, and
            # the mass rate is what is measured: the pin's own refill, which
            # for this boundary is identically the flux through the mouth. An
            # integrator rather than a formula because the map from imposed
            # velocity to delivered mass is not the textbook rho u A -- the
            # lid above the mouth reflects part of it, measured at about 40 %
            # on a 4x4x40 column -- and that factor is geometry, not a
            # constant worth hard-coding.
            #
            # Clipping to [0, u_ceiling] is the anti-windup and the physics at
            # once: the basin cannot suck, and it cannot push harder than its
            # own head. When the column backs up the ceiling falls to zero and
            # takes the integrator with it, however far behind target the
            # delivery is -- which is the throttle, and the reason this cannot
            # force metal into a closed mold.
            delivery_error = (
                target_mass_per_step - delivered_previous
            ) / max(target_mass_per_step, 1e-30)
            u_drive = float(np.clip(
                u_drive + INLET_RATE_GAIN * delivery_error * u_target_lattice,
                0.0,
                u_ceiling,
            ))
            # The driven velocity is a FLOOR on the old measured boundary, not
            # a replacement for it, and that is deliberate.
            #
            # The measurement below is the previous boundary in full: the raw
            # first moment of the inlet cells' own populations, low-passed,
            # deliberately without the Guo half-force correction -- that
            # correction is the term that made the very first inlet a momentum
            # pump, and it does not come back in by this door either. What the
            # measurement is bad at is *starting* a pour, because the lid
            # above the mouth holds it near zero; what it is good at is the
            # endgame, where it tracks the small real motions that push the
            # last few cells over the transition threshold.
            #
            # Taking the stronger of the two keeps both. Measured on the
            # walled tower fixture with the driven velocity replacing the
            # measurement outright: delivery ran at 105-133 % of target
            # through the fill -- the starvation fixed -- and then shut off
            # hard at hydrostatic equilibrium with 96 % of the mass in and 12
            # of 80 cells still flickering across the threshold, so the run
            # stalled at 85 % instead of completing. As a floor it completes
            # in 984 steps against the 1,148 that fixture has always taken:
            # the pour is driven while there is head to drive it and hands
            # back to the measurement once there is not.
            #
            # There is still one ceiling the measurement does not get to
            # cross, and it is not the target: it is Torricelli through the
            # mouth. A basin holding this head over this mouth cannot supply
            # metal faster than sqrt(2 g h), whatever the column below is
            # willing to pull, and letting the measurement impose more is
            # inventing a basin pressure that no basin has.
            #
            # It binds only where the domain pulls harder than the basin can
            # pour, which is a short free-falling column rather than a real
            # rigging: on the 2x2x20 tower fixture, whose 4-cell mouth could
            # pass 2.9 lb/s against a 0.57 lb/s square-root target, it takes
            # the peak delivery from 650 % of target to 533 % -- the mouth
            # ceiling, as it should. On Full_Gating_Test and on Test 2 it
            # never binds at all: both ledgers are unchanged to the byte with
            # it in place, because the rate controller is already holding
            # delivery at 88 % and 99 % of target respectively.
            #
            # A constant, deliberately, and not the back-pressure ceiling
            # above: damming the endgame with a ceiling that falls to zero is
            # what left 12 of 80 cells flickering on that same tower.
            f_slice = state.f[:, ix, iy, iz].astype(np.float64)
            rho_now = f_slice.sum(axis=0)
            u_raw = lattice_c_t @ f_slice / np.clip(rho_now, 1e-9, None)
            inlet_u_measured *= 1.0 - INLET_VELOCITY_BETA
            inlet_u_measured += INLET_VELOCITY_BETA * u_raw
            u_used = inlet_u_measured.copy()
            # Gravity points -Z and metal enters falling, so "stronger" means
            # more negative. Only the driven component is imposed; the lateral
            # components stay as measured, so the boundary cannot invent a
            # sideways jet the flow did not have.
            np.minimum(u_used[2], -u_drive, out=u_used[2])
            np.maximum(u_used[2], -u_mouth_ceiling_lattice, out=u_used[2])
            # Clamped by norm rather than per component, so clamping cannot
            # rotate the flow direction.
            speed = np.sqrt(np.sum(u_used * u_used, axis=0))
            u_used *= np.minimum(1.0, u_cap_lattice / np.maximum(speed, 1e-12))
            state.f[:, ix, iy, iz] = equilibrium(inlet_rho, u_used).astype(np.float32)
        # ---- Metered delivery ---------------------------------------------
        #
        # Two parts, in this order, and the order is the throttle.
        #
        # First the pin: the inlet cells go back to full, and what that costs
        # is the mass that drained out of them last step -- for a coherent
        # column, exactly the flux the flow field carried away. Measured before
        # the write, so it is what the pin replaced rather than what it wrote.
        # On a backed-up fill this goes to zero on its own.
        #
        # Then the top-up, and only if the pin came in under the target: the
        # basin still has metal to give, so it is offered to the throat below
        # the mouth, nearest layer first, and only into the room each cell
        # actually has. When the throat is running full there is no room and
        # the top-up is zero -- delivery is then whatever the column carries,
        # which is the honest answer and the reason this cannot force mass
        # into a closed system.
        #
        # Mass only. No population, no momentum: the pressure boundary above
        # is the entire momentum story, and adding one here is how the
        # original inlet became a pump.
        #
        # Every term added to ``poured`` is a term written into ``state.mass``,
        # so the ledger closes to float64 round-off by construction. See
        # tests/test_filling_stage.py's delivery-ledger test.
        inlet_refill = float(np.sum(inlet_full_mass)) - float(
            np.sum(state.mass[ix, iy, iz])
        )
        state.mass[ix, iy, iz] = inlet_full_mass
        delivered = inlet_refill
        budget = target_mass_per_step - delivered
        if budget > 0.0:
            for _idx, _porosity in delivery_layers:
                wet_here = state.flags[_idx] == FLAG_INTERFACE
                if not wet_here.any():
                    continue
                room = np.where(
                    wet_here,
                    np.asarray(state.rho[_idx], dtype=np.float64) * _porosity
                    - np.asarray(state.mass[_idx], dtype=np.float64),
                    0.0,
                )
                np.maximum(room, 0.0, out=room)
                want = float(room.sum())
                if want <= 0.0:
                    continue
                if want > budget:
                    room *= budget / want
                    want = budget
                state.mass[_idx] += room
                delivered += want
                budget -= want
                if budget <= 0.0:
                    break
        poured += delivered
        # What the rate controller reads next step. The pin's refill alone,
        # not the topped-up total: a controller fed its own top-up would see
        # the target met, back the velocity off to zero, and deliver the whole
        # pour by placing mass rather than by flowing it.
        delivered_previous = inlet_refill
        # Metal entering at the inlet is at the pour temperature -- pinned
        # every step, same as mass and momentum above, so the source stays a
        # true Dirichlet boundary rather than drifting as wall loss and
        # diffusion nibble at it.
        state.temperature[ix, iy, iz] = pour_temp

        step(state, units, track_free_surface=True, smagorinsky_cs=smagorinsky_cs)
        advect_temperature(state, alpha_lattice)
        apply_wall_heat_loss(
            state,
            htc_w_m2k=htc_dense,
            mold_temperature_c=mold_temp,
            density_kg_m3=float(alloy.density_liquid),
            specific_heat_j_kgk=float(alloy.specific_heat),
        )

        if detailed_progress_callback is not None and (
            taken % PROGRESS_UPDATE_INTERVAL == 0
        ):
            # Against the step budget, not the fill percentage. The bar has to
            # move monotonically, and filled-cell counts do not: metal sloshes
            # back and forth across the same cells while a front is stalled.
            # A run that finishes early simply finishes early.
            detailed_progress_callback(taken, max_steps, "Filling")

        if taken % frame_every == 0:
            capture_frame(taken)
            if len(frames) >= 2 * FILL_ANIMATION_FRAMES:
                # Drop every other frame and sample half as often from here.
                # The survivors stay evenly spaced, so playback never shows a
                # dense early sequence followed by a sparse late one.
                del frames[1::2]
                frame_every *= 2

        if taken % DIVERGENCE_CHECK_INTERVAL == 0:
            # Mass conservation is the honest health check. The free surface
            # can only hold as much mass as it has wet cells, so a total above
            # that ceiling is not drift -- it is the collision operator having
            # left the range where it means anything. Checking flags alone
            # cannot see this: a blow-up promotes every cell to FLUID, which
            # reads as a perfect 100% fill.
            wet = (state.flags == FLAG_FLUID) | (state.flags == FLAG_INTERFACE)
            wet_cells = int(np.count_nonzero(wet))
            wet_mass = float(np.sum(state.mass[wet])) if wet_cells else 0.0
            ceiling = DIVERGENCE_MASS_FACTOR * max(wet_cells, 1)
            if (not np.isfinite(wet_mass)) or wet_mass > ceiling or wet_mass < 0.0:
                status = STATUS_DIVERGED
                diverged_at_step = taken
                emit(
                    f"Filling ABORTED: solver diverged at step {taken:,} "
                    f"({taken * units.dt_s:.2f} s). Wet mass {wet_mass:.4g} "
                    f"against a physical ceiling of {ceiling:.4g} over "
                    f"{wet_cells:,} wet cells. Results discarded."
                )
                break

            # ---- Instability precursors -------------------------------
            #
            # The mass ceiling above only fires once the bookkeeping has
            # already broken, which on the measured Test 2 divergence was the
            # same 200-step interval in which the field went non-finite. The
            # collapse was not sudden: negative populations appeared 52,000
            # steps earlier and were present in 59 % of samples from then on,
            # in bursts, while every other observable still looked healthy.
            #
            # Negative populations are the precursor, and prevalence is the
            # statistic -- not presence, and not magnitude. An isolated
            # negative is routine and self-heals (see free_surface_kernel);
            # a *persistent* one is the reconstruction failing repeatedly at
            # the same place, and it always ends the same way. Deliberately
            # not built on max|u| or on the density spread: on the healthy
            # Full_Gating_Test run those reach 0.457 against a 0.05 guard and
            # 0.347 respectively, so any threshold that catches the sick run
            # kills the healthy one. See NEGATIVE_POPULATION_WINDOW.
            negative_cells = (
                int(np.count_nonzero(np.any(state.f < 0.0, axis=0) & wet))
                if wet_cells
                else 0
            )
            negative_history.append(1 if negative_cells else 0)
            if len(negative_history) > NEGATIVE_POPULATION_WINDOW:
                del negative_history[0]
            prevalence = (
                sum(negative_history) / float(NEGATIVE_POPULATION_WINDOW)
                if len(negative_history) >= NEGATIVE_POPULATION_WINDOW
                else 0.0
            )
            cascading_now = (
                negative_cells >= NEGATIVE_POPULATION_CASCADE_MIN_CELLS
                and negative_cells
                >= NEGATIVE_POPULATION_CASCADE_FRACTION * max(wet_cells, 1)
            )
            # Presence over two consecutive checks, not one sample -- see
            # NEGATIVE_POPULATION_CASCADE_CHECKS. On a small model the 8-cell
            # floor and the 2 % fraction meet at ~400 wet cells, and the first
            # check lands at step 200 while the splat rim is still transient.
            cascade_streak = cascade_streak + 1 if cascading_now else 0
            cascading = cascade_streak >= NEGATIVE_POPULATION_CASCADE_CHECKS
            if cascading or prevalence >= NEGATIVE_POPULATION_PREVALENCE:
                status = STATUS_DIVERGED
                diverged_at_step = taken
                if cascading:
                    precursor = (
                        f"{negative_cells:,} of {wet_cells:,} wet cells "
                        f"({negative_cells / max(wet_cells, 1):.1%}) are "
                        f"holding a negative population, at "
                        f"{NEGATIVE_POPULATION_CASCADE_CHECKS} consecutive "
                        f"checks "
                        f"({NEGATIVE_POPULATION_CASCADE_CHECKS * DIVERGENCE_CHECK_INTERVAL} "
                        f"steps) -- a reconstruction cascade in progress, not a "
                        f"transient"
                    )
                else:
                    precursor = (
                        f"negative populations have been present at "
                        f"{prevalence:.0%} of the last "
                        f"{NEGATIVE_POPULATION_WINDOW} checks "
                        f"({NEGATIVE_POPULATION_WINDOW * DIVERGENCE_CHECK_INTERVAL:,} "
                        f"steps), {negative_cells:,} cells at this one. The "
                        f"free-surface reconstruction is failing persistently "
                        f"and this run will go non-finite"
                    )
                emit(
                    f"Filling ABORTED: solver diverging at step {taken:,} "
                    f"({taken * units.dt_s:.2f} s). {precursor}. Stopped "
                    f"before the field lost finiteness, so the state below is "
                    f"still readable -- but it is not a fill. Results "
                    f"discarded."
                )
                break

            # A FLUID cell holding an impossible amount of metal.
            #
            # The second precursor, and the one that names a mechanism rather
            # than its wreckage: see FLUID_FILL_FRACTION_MIN for what eps means
            # for a FLUID cell, why a breach is invisible to every flag scan,
            # and why two consecutive checks are required. Same 200-step cadence
            # as everything else in this block -- three whole-grid vectorised
            # passes, against a step that already does dozens.
            fluid_cells = state.flags == FLAG_FLUID
            eps_fluid = fluid_fraction(state)
            out_of_band = fluid_cells & (
                (eps_fluid < FLUID_FILL_FRACTION_MIN)
                | (eps_fluid > FLUID_FILL_FRACTION_MAX)
            )
            offenders = int(np.count_nonzero(out_of_band))
            fill_fraction_breaches = fill_fraction_breaches + 1 if offenders else 0
            if fill_fraction_breaches >= FLUID_FILL_FRACTION_CHECKS:
                worst = float(eps_fluid[out_of_band].max())
                lowest = float(eps_fluid[out_of_band].min())
                where_z = np.nonzero(np.any(out_of_band, axis=(0, 1)))[0]
                parts = "/".join(
                    sorted(
                        CODE_TO_PART_TYPE[int(code)].value.replace("_", " ")
                        for code in np.unique(solve_types[out_of_band])
                        if int(code) in CODE_TO_PART_TYPE
                    )
                ) or "the mold pad"
                status = STATUS_DIVERGED
                diverged_at_step = taken
                emit(
                    f"Filling ABORTED: solver diverging at step {taken:,} "
                    f"({taken * units.dt_s:.2f} s). {offenders:,} FLUID "
                    f"cell{'' if offenders == 1 else 's'} "
                    f"hold{'s' if offenders == 1 else ''} a fill fraction "
                    f"outside "
                    f"[{FLUID_FILL_FRACTION_MIN:.1f}, "
                    f"{FLUID_FILL_FRACTION_MAX:.1f}] -- as low as "
                    f"{lowest:.3f} and as high as {worst:.3f} -- in the "
                    f"{parts}, at solve z "
                    f"{int(where_z.min()) - pad_lo}-"
                    f"{int(where_z.max()) - pad_lo}, and did so at "
                    f"{FLUID_FILL_FRACTION_CHECKS} consecutive checks "
                    f"({FLUID_FILL_FRACTION_CHECKS * DIVERGENCE_CHECK_INTERVAL:,} "
                    f"steps), so this is not a transient. A full cell holds "
                    f"exactly its pore volume, so the mass bookkeeping has "
                    f"come apart rather than the metal having backed up: "
                    f"nothing in the solver can take mass back out of a FLUID "
                    f"cell, and because the exchange conserves mass exactly "
                    f"the surplus was stolen from real cells. Results "
                    f"discarded."
                )
                break

        if taken % entrapment_check_interval == 0:
            # Per output frame, not per step (design doc 6.4): a
            # connected-component labelling over the whole coarse grid every
            # single lattice step would be the single most expensive thing
            # in this loop.
            vent_mask = static_vent | (open_air_mask & (state.flags == FLAG_GAS))
            if vent_warning is None:
                vent_warning = _entrapment_warning_if_no_vent(vent_mask)
            entrapped_now = find_entrapped_gas(state.flags, vent_mask)
            entrapped_now_count = int(np.count_nonzero(to_voxels(entrapped_now)))
            peak_entrapped_voxels = max(peak_entrapped_voxels, entrapped_now_count)

        if taken % report_every == 0:
            if cancel_callback is not None:
                cancel_callback()
            reached = int(np.count_nonzero((state.flags == FLAG_FLUID) & metal))
            # Two measures of the same fill, because neither alone is honest.
            # The cell count is what "filled" means, but it moves in steps of a
            # whole cell: an interface cell fills continuously from eps 0 to
            # eps 1 and only flips to FLUID at the top of that band. Mass moves
            # the whole time. Reporting both makes a slow fill legible as slow
            # rather than as stopped.
            mass_fraction = float(
                np.clip(float(np.sum(state.mass[metal])), 0.0, metal_full_mass)
            ) / max(metal_full_mass, 1e-12)
            # Delivery, said out loud, every report, forever. The whole defect
            # this instrumentation exists for was a pour running at 2-5% of
            # its physical rate for tens of thousands of steps while every
            # other line in the log looked reasonable. A rate next to its
            # target makes starvation a one-line read.
            interval_steps = max(1, taken - step_at_last_report)
            delivery_lb_s = (
                (poured - poured_at_last_report) * poured_lb_per_lattice_mass
                / (interval_steps * units.dt_s)
            )
            poured_at_last_report = poured
            step_at_last_report = taken
            emit(
                f"Filling: {reached:,}/{metal_total:,} cells "
                f"({reached / max(1, metal_total):.0%}), "
                f"{mass_fraction:.0%} by mass, "
                f"{poured * poured_lb_per_lattice_mass:.1f} lb poured, "
                f"delivering {delivery_lb_s:.2f} of "
                f"{target_lb_s:.2f} lb/s target "
                f"({delivery_lb_s / max(target_lb_s, 1e-9):.0%}), at "
                f"{taken * units.dt_s:.2f} s"
            )
            if reached >= metal_total:
                status = STATUS_COMPLETE
                break
            # A front that has stopped advancing will not restart: the metal
            # either cannot get through the section ahead or has frozen the
            # channel behind it. Running out the clock on that produces the
            # same answer far more slowly, and reports it as a time cap --
            # which reads as "needed longer" when it means "cannot get there".
            #
            # Measured against the high-water mark, not the previous report.
            # A stalled front does not sit still: metal sloshes back and forth
            # across the same cells, so the count ticks up and down by a few
            # percent forever. Comparing consecutive reports resets on every
            # one of those wobbles and never fires.
            #
            # BOTH measures must be flat. The cell count alone declared a stall
            # on a front that was genuinely advancing: measured on
            # Test_Model/Filling_Test, 428 casting cells sat at eps ~ 0.5,
            # taking on mass every report while flipping no flags. Requiring
            # the mass high-water mark to be flat as well is the difference
            # between "cannot get there" and "is getting there slowly".
            advanced = False
            if reached > last_reached:
                last_reached = reached
                advanced = True
            if mass_fraction > best_mass_fraction + STALL_MASS_FRACTION_EPSILON:
                best_mass_fraction = mass_fraction
                advanced = True
            if advanced:
                flat_reports = 0
            else:
                flat_reports += 1
            if flat_reports >= STALL_REPORTS and taken >= STALL_MIN_STEPS:
                status = STATUS_STALLED
                emit(
                    f"Filling stopped: the front has not advanced for "
                    f"{flat_reports} consecutive reports, holding at "
                    f"{reached:,}/{metal_total:,} cells "
                    f"({reached / max(1, metal_total):.0%}, "
                    f"{mass_fraction:.0%} by mass)."
                )
                break

    if status == STATUS_DIVERGED:
        # Every frame after the blow-up shows the blow-up, and the frames
        # before it end mid-pour with no indication that what follows is
        # rubbish. Playing that back would be the most convincing possible
        # presentation of a failed solve, so there is nothing to play.
        frames.clear()
    else:
        # The loop exits on a report boundary, which is up to report_every
        # steps after the state the run actually ended in. Without this the
        # last frame is not the final state -- most visibly on a complete
        # fill, where playback would stop just short of full.
        capture_frame(taken)

    # Final entrapment snapshot -- evaluated unconditionally rather than
    # relying on the last step happening to land on a check interval, so the
    # reported "final" set always reflects the state actually handed off.
    final_vent_mask = static_vent | (open_air_mask & (state.flags == FLAG_GAS))
    if vent_warning is None:
        vent_warning = _entrapment_warning_if_no_vent(final_vent_mask)
    entrapped_coarse = find_entrapped_gas(state.flags, final_vent_mask)

    entrapped_voxels = to_voxels(entrapped_coarse).astype(bool)
    entrapped_count = int(np.count_nonzero(entrapped_voxels))
    # Peak, not just final: a pocket can be entrapped mid-fill and vent later
    # as the geometry around it changes, so the final count alone
    # understates the risk (design doc section 6.4). Tracked in voxel units
    # throughout, at every periodic check during the loop above, so it is
    # never below the final count.
    peak_entrapped_count = max(entrapped_count, peak_entrapped_voxels)

    # F3: cold shuts. Evaluated once on the completed fill/thermal fields --
    # unlike entrapment there is no "peak" concept here: a seam does not heal
    # once the fronts have met and cooled, so the end-of-fill field is the
    # answer.
    liquidus_c = float(getattr(alloy, "liquidus_temp", -np.inf))
    cold_shut_coarse = find_cold_shuts(
        state.fill_time_s,
        state.temperature,
        state.flags,
        liquidus_c=liquidus_c,
        arrival_gap_s=float(getattr(filling, "cold_shut_arrival_gap_s", 0.5)),
        temperature_margin_c=float(getattr(filling, "cold_shut_temp_margin_c", 25.0)),
    )
    cold_shut_voxels = to_voxels(cold_shut_coarse).astype(bool)
    cold_shut_count = int(np.count_nonzero(cold_shut_voxels))

    filled_coarse = state.flags == FLAG_FLUID
    fill_time_coarse = np.asarray(state.fill_time_s, dtype=np.float64)

    fill_time_voxels = to_voxels(fill_time_coarse)
    filled_voxels = to_voxels(filled_coarse).astype(bool)

    voxel_metal = to_voxels(metal).astype(bool)
    filled_count = int(np.count_nonzero(filled_voxels & voxel_metal))
    misrun_count = int(np.count_nonzero(voxel_metal & ~filled_voxels))

    # Mold, core and chill voxels can never fill, so they must not be able to
    # read as misruns. Every consumer of this field -- the desktop renderer,
    # the slider label, the web payload -- defines "misrun" as a negative
    # arrival time, and gather_to_voxels hands back the -1 sentinel for all of
    # them alike. Left as -1 they show up in the Fill Time view as a solid grey
    # shell of "metal that never arrived": on Test_Model/Filling_Test that is
    # 1,485 chill voxels reported as misruns against a true misrun count of 0.
    #
    # NaN is the sentinel for "this voxel is not metal, the question does not
    # apply". It is not a second flavour of -1: NaN is false against `< 0`,
    # `>= 0` and `<= t` alike, so it drops out of the misrun mask, the arrived
    # mask and the time range without any consumer special-casing it, whereas
    # any negative sentinel has to be excluded by hand everywhere.
    fill_time_voxels = np.where(voxel_metal, fill_time_voxels, np.nan)

    # F2: gather the temperature the fill actually left behind, rather than
    # the two-constant (pour temp / mold temp) placeholder F1 used. Metal
    # that entered first and travelled furthest has been advecting and
    # losing heat to the mold the whole run, so it comes back cooler than
    # metal still at the gate.
    temperature_fill = to_voxels(
        np.asarray(state.temperature, dtype=np.float64)
    )

    # A cell can hold an arrival time and no temperature. fill_time_s is written
    # once, when metal first reaches a cell, and is never cleared; the
    # temperature field is NaN-reset whenever a cell drains back to gas. So any
    # cell that filled and later drained -- routine wherever the front sloshes
    # or a channel empties behind it -- carries a valid fill time alongside a
    # NaN temperature.
    #
    # set_initial_temperature_field rejects non-finite values outright, so
    # trusting fill_time alone here crashes the whole run at handover. It never
    # showed up on a synthetic column, where nothing drains, only on real
    # gating geometry. Require BOTH signals before treating a voxel as carrying
    # metal temperature.
    filled_mask = filled_voxels & voxel_metal & np.isfinite(temperature_fill)

    # Every voxel must carry a finite temperature -- the seam validates that
    # -- so unfilled metal cannot simply be left blank. Seeding it at mold
    # temperature says "no metal arrived here", which is what a misrun is. The
    # alternative, seeding it at pouring temperature, would hide the defect
    # behind a plausible-looking result.
    temperature = np.full(mapping.n_voxels, mold_temp, dtype=np.float64)
    temperature[filled_mask] = temperature_fill[filled_mask]

    if not np.all(np.isfinite(temperature)):
        raise AssertionError(
            "filling produced a non-finite handover temperature; "
            "set_initial_temperature_field would reject it"
        )

    if np.any(filled_mask):
        min_fill_temperature_c = float(temperature[filled_mask].min())
        mean_fill_temperature_c = float(temperature[filled_mask].mean())
    else:
        min_fill_temperature_c = mold_temp
        mean_fill_temperature_c = mold_temp

    if status == STATUS_DIVERGED:
        # Said first and said plainly. Everything below this describes fields
        # the solver produced after it stopped conserving mass, so a misrun
        # count or a cold-shut count from here is an artefact of the blow-up,
        # not a casting defect.
        warnings.append(
            f"SOLVER DIVERGED at step {diverged_at_step:,} -- this fill result is "
            f"not physical and must not be used. Any misrun, entrapment or "
            f"cold-shut figure below is an artefact of the divergence."
        )
    elif status == STATUS_STALLED:
        warnings.append(
            f"the fill stalled at {filled_count / max(1, filled_count + misrun_count):.0%} "
            f"-- the front stopped advancing with metal still unfilled. This is "
            f"a genuine misrun or an under-resolved section, not a time limit."
        )
    elif status == STATUS_TIME_CAPPED:
        warnings.append(
            f"the fill was still advancing when max_fill_time_s "
            f"({max_time:.0f} s) ran out; it did not finish. Raise the cap "
            f"before reading the misrun count as a defect."
        )

    if misrun_count:
        warnings.append(
            f"{misrun_count:,} metal voxels never filled (misrun): "
            f"{misrun_count / max(1, filled_count + misrun_count):.1%} of the metal"
        )
    liquidus_temp = float(getattr(alloy, "liquidus_temp", -np.inf))
    if np.any(filled_mask) and min_fill_temperature_c < liquidus_temp:
        warnings.append(
            f"filled metal cooled below liquidus ({liquidus_temp:.0f} C) before "
            f"handover: minimum fill temperature {min_fill_temperature_c:.0f} C -- "
            f"incipient cold-shut risk"
        )
    solver_mode = str(getattr(analysis_options, "solver_mode", "") or "")
    if solver_mode and solver_mode != "thermal_coupled":
        warnings.append(
            f"solver_mode is {solver_mode!r}; the seeded temperature field is "
            f"only consumed by 'thermal_coupled', so the fill's thermal result "
            f"will be ignored by the solidification solve."
        )
    # F3: air entrapment and cold shuts.
    if vent_warning is not None:
        warnings.append(vent_warning)
    if entrapped_count:
        warnings.append(
            f"{entrapped_count:,} voxels of trapped air detected (no vent path "
            f"reached, peak {peak_entrapped_count:,}): risk of gas porosity"
        )
    if cold_shut_count:
        warnings.append(
            f"{cold_shut_count:,} voxels show a cold-shut signature: fill "
            f"fronts met after the metal had cooled to within "
            f"{float(getattr(filling, 'cold_shut_temp_margin_c', 25.0)):.0f} C "
            f"of the liquidus ({liquidus_c:.0f} C)"
        )
    for message in warnings:
        emit(f"Filling warning: {message}")

    return FillingResult(
        fill_time_s=fill_time_voxels.astype(np.float32),
        temperature_c=temperature,
        inlet_voxel_count=int(inlet.shape[0]),
        filled_voxel_count=filled_count,
        misrun_voxel_count=misrun_count,
        choke_area_cm2=measure_choke_area_cm2(solve_core, dx_m),
        filter_zones=filter_zones,
        steps=taken,
        fill_seconds=taken * units.dt_s,
        coarsening=coarsening,
        min_fill_temperature_c=min_fill_temperature_c,
        mean_fill_temperature_c=mean_fill_temperature_c,
        entrapped_voxel_count=entrapped_count,
        peak_entrapped_voxel_count=peak_entrapped_count,
        cold_shut_voxel_count=cold_shut_count,
        entrapped_gas=entrapped_voxels,
        cold_shut=cold_shut_voxels,
        warnings=tuple(warnings),
        status=status,
        diverged_at_step=diverged_at_step,
        choke_thickness_cells=choke_thickness_cells,
        choke_thickness_by_part={
            _METAL_PATH_LABELS[code]: value
            for code, value in choke_by_part.items()
        },
        choke_thickness_bodies_by_part={
            _METAL_PATH_LABELS[code]: list(values)
            for code, values in choke_components.items()
        },
        choke_body_fragments_by_part={
            _METAL_PATH_LABELS[code]: list(sizes)
            for code, sizes in coarsening_decision.fragments_by_part.items()
        },
        narrowest_choke_part=(
            _METAL_PATH_LABELS[narrowest_code]
            if narrowest_code is not None
            else ""
        ),
        choke_area_fine_cm2=measure_choke_area_cm2(fine_types, mapping.dx_m),
        delivered_lb=poured * poured_lb_per_lattice_mass,
        target_delivery_lb_s=target_lb_s,
        configured_coarsening=coarsening_decision.configured,
        frames=tuple(frames),
    )


def maybe_run_filling(
    voxel_mesh,
    alloy,
    analysis_options,
    inputs,
    *,
    cancel_callback=None,
    progress_message_callback=None,
    detailed_progress_callback=None,
) -> Optional[FillingResult]:
    """
    Run the filling stage if, and only if, it is both enabled and possible.

    The single gate both pipelines call. ``core/surrogate/pipeline.py``
    re-implements ``run_pipeline``'s sequence rather than calling it, so a
    stage wired into only one of them silently no-ops in the other -- which is
    why this is a shared helper and not inline code.

    Requirement: filling runs only when the user has picked a pour location.
    A missing location with the toggle on is reported rather than ignored,
    because silently skipping the stage looks identical to running it and
    finding nothing wrong.
    """

    def emit(message: str) -> None:
        if progress_message_callback is not None:
            progress_message_callback(message)

    if not getattr(analysis_options, "filling", False):
        return None

    raw = getattr(inputs, "pour_location", None)
    if not raw:
        emit(
            "Filling is enabled but no pour location has been selected; "
            "skipping the fill. Click a point on the assembly to set one."
        )
        return None

    from core.pour_location import PourLocation

    try:
        location = raw if isinstance(raw, PourLocation) else PourLocation.from_dict(raw)
    except (ValueError, KeyError, TypeError) as exc:
        emit(f"Filling skipped: pour location is invalid ({exc}).")
        return None

    return run_filling_stage(
        voxel_mesh,
        alloy,
        analysis_options,
        location,
        cancel_callback=cancel_callback,
        detailed_progress_callback=detailed_progress_callback,
        progress_message_callback=progress_message_callback,
    )
