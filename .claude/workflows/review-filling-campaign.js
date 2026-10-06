export const meta = {
  name: 'review-filling-campaign',
  description: 'Adversarial multi-lens review of the committed filling-campaign diffs (rounds 1-2)',
  phases: [
    { title: 'Review', detail: 'seven lenses over 460ca0f..57b07d7' },
    { title: 'Verify', detail: 'three refuters per finding, majority kills' },
  ],
}

const REPO = '/Users/brendonjarett/Desktop/Active Codebase/Python_Cast_Sim'

const PREAMBLE = `You are reviewing committed changes in the git repo at "${REPO}" (branch refactor/pouring-seams).
Scope: the diff range 460ca0f..57b07d7 — three commits: 759aab4 (midpoint kd-tree patched into trimesh.proximity), c5f8659 (per-part median local-thickness choke guard), 57b07d7 (driven head-mode inlet + integral rate controller, negative-population precursor abort, choke-driven auto-coarsening with config plumbing through core/solidification_params.py, ui/main_window.py, web_frontend).
Context: this is a free-surface LBM mold-filling solver (Koerner eps/mass, TRT near tau=0.5, Guo forcing, populations stored float32 computed float64). The Numba kernel path (physics/fluid_flow.py free_surface_kernel) and NumPy fallback (physics/filling.py) must behave identically. The changes were validated end-to-end on two models, so do not report "this was never run" — hunt for latent defects the validation runs would NOT exercise.
HARD CONSTRAINTS: READ-ONLY — never edit any file. Never run the test suite, never run simulations, never import the repo's core/ or physics/ modules (importing initializes Metal/Taichi/Numba and this machine is running a long validation job). git log/show/diff and reading files are your tools; reason from source. A separate uncommitted work-in-progress may exist in the working tree — review the COMMITTED state (git show), not the working tree.
Report ONLY defects that would change behavior: wrong results, crashes, unit errors, dead config paths, invariant violations, regressions on inputs unlike the two validated models. No style nits, no docstring complaints, no "consider adding". For each finding give file, line (in the committed version), a one-sentence claim, and a CONCRETE failure scenario (inputs/state -> wrong outcome). Rank most-severe first. At most 6. If you find nothing that clears this bar, return an empty list — an empty list is a good answer.`

const LENSES = [
  { key: 'inlet-physics', prompt: `${PREAMBLE}
Your lens: PHYSICS AND UNITS of the new inlet drive in core/filling_stage.py (commit 57b07d7). Check every conversion chain end to end: head inches -> meters -> lattice density; Bernoulli u = Cd*sqrt(2*cs2*max(0,drho)/rho0) dimensional sanity in lattice units; target rate min(sqrt(W) lb/s, Cd*A*sqrt(2gh)) — is A the mouth area in the right units, is W the metal mass in lb, is the lb/s -> lattice-mass-per-step conversion using the right dt and dx^3 and metal density; the pour_rate_lb_s override path; the sqrt(W) foundry-rule figure; the delivered-vs-target progress-line math. Also: does the integral controller's gain/anti-windup misbehave when target_mass_per_step is tiny (tiny models) or huge; is REFERENCE_DENSITY vs inlet_density (1+g*h/cs2) used consistently in the drive.` },
  { key: 'inlet-edges', prompt: `${PREAMBLE}
Your lens: EDGE CASES AND DEGENERATE INPUTS in the round-2 inlet code path (core/filling_stage.py, commit 57b07d7). Hunt: mouth of zero or one cell; no wet throat cells ever (throat_wet.any() false -> REFERENCE_DENSITY fallback — does the drive then read zero head and never pour?); delivery_layers empty; a pour point placed so _expand_inlet_to_mouth floods into a different part or finds no mouth; inlet cells that are themselves POROUS or adjacent to obstacle; velocity mode (inlet_mode_code=1) — is it untouched, and does any new code run for it that should not; head of zero; the top-up writing mass into INTERFACE cells whose rho is near zero (room = rho*porosity - mass can be negative or huge); float32 casts of driven equilibria; what happens on the very first step when delivered_previous is uninitialized; restart/resume of the stage if supported.` },
  { key: 'choke-math', prompt: `${PREAMBLE}
Your lens: THE CHOKE MEASUREMENT (commit c5f8659, core/filling_stage.py: _local_thickness_cells, measure_choke_thickness_by_part, narrowest_choke_part_code, measure_choke_area_cm2 call-site changes). Check the Hildebrand-Ruegsegger sweep for correctness: the half-cell radius stepping, the 2r-1 convention, the pad/crop bookkeeping (box slices + 1-cell pad + inner slice — off-by-one on any axis?), behavior on a part that touches the domain edge, a part of exactly 1 cell, two disconnected bodies sharing one part code (two gates DO share a code in the real assembly — does the median over both bodies mask one thin gate welded wide?), non-cubic anisotropy assumptions, and the cost of the sweep on a large fine grid (the radius loop runs one EDT per half-cell of max radius — on a 1.7M-cell grid at coarsening 1 with a fat sprue, how many EDTs, and is that seconds or minutes at setup?). Also the warning wiring: thresholds <=2 hard vs <=3 marginal — boundary values exactly 2.0 and 3.0, and parts absent from the grid.` },
  { key: 'guard-and-status', prompt: `${PREAMBLE}
Your lens: THE PRECURSOR ABORT AND STATUS REPORTING (commit 57b07d7). Check the negative-population prevalence bookkeeping: the 40-check window state (initialization, wraparound), the >=50% and >=2%-of-wet-cells (min 8) triggers at boundary values, interaction with the pre-existing wet-mass-ceiling divergence guard (can both fire? which message wins?), whether the abort path sets STATUS_DIVERGED/diverged_at_step consistently with the old path so downstream consumers (core/simulation_pipeline.py filling summary, solidification handover fallback, web report) behave identically, and whether counting negative populations every 200 steps scans the full f array (memory/time cost on 1.7M cells x 19 populations) or a maintained counter. Also: does the guard read the float32 stored populations or a float64 copy, and can the count be triggered by benign -0.0 or denormals.` },
  { key: 'autocoarsen-config', prompt: `${PREAMBLE}
Your lens: AUTO-COARSENING AND CONFIG PLUMBING (commit 57b07d7: choose_fill_coarsening/CoarseningDecision in core/filling_stage.py; choke_auto_resolution through core/solidification_params.py, ui/main_window.py FIELD_DEFS, web_frontend/src/lib/defaults.ts and SolidificationDrawer.tsx). Check: does every consumer that previously used the configured coarsening now use the DECIDED one (dx_m, step budget, units/dt derivation, frame downsampling, mass ceiling, inlet mouth expansion — any spot still reading params.fill_grid_coarsening after the decision is made?); is the decision made before or after filter zones / vent masks / inlet seeds are built on a grid; config round-trip (yaml -> params -> to_dict -> yaml) for the new key including old config files without it; the Qt field and web drawer defaults agreeing with the backend default (ON); int vs bool coercion of the config value; and whether choose_fill_coarsening's re-coarsening from fine_types agrees with how the stage builds solve_types elsewhere (same coarsen_part_types call and priority rules).` },
  { key: 'result-consumers', prompt: `${PREAMBLE}
Your lens: DOWNSTREAM CONSUMERS OF FillingResult AND THE LOGS (commits c5f8659 + 57b07d7). New fields: choke_thickness_by_part, narrowest_choke_part, choke_area_fine_cm2, delivered_lb, target_delivery_lb_s, configured_coarsening. Find every consumer of FillingResult (core/simulation_pipeline.py, core/solidification.py, web/ report builders, ui/, serialization to disk if any) and check: constructors/serializers that enumerate fields explicitly and now miss or crash on new ones; dict/JSON dumps hitting non-serializable numpy types from the new fields; the _filling_summary keys the web frontend reads (does the TS side render or choke on the new keys); old saved results loaded by new code and vice versa; and the reworded progress/setup log lines — anything parsing them (tests, web log tailers) that matched the old wording.` },
  { key: 'kdtree-parity', prompt: `${PREAMBLE}
Your lens: THE KD-TREE PATCH AND CROSS-CUTTING PARITY RISK (commit 759aab4 + interactions). The patch swaps a balanced_tree=False cKDTree subclass into trimesh.proximity's namespace at core/voxel_mesh.py import. Check: does anything else in the process rely on trimesh.proximity.cKDTree being scipy's class (isinstance checks, pickling of trees, trimesh internals that construct with keyword balanced_tree explicitly); multiprocessing spawn workers — is core.voxel_mesh imported in every worker entry path that uses proximity, or is there a path (web worker, CLI tool in tools/, generate_samples.py) that reaches trimesh proximity without the patch; and test pollution (the patch is process-global — can a test assert on scipy default behavior and now fail order-dependently). Separately, scan the round-2 stage-loop changes for anything that silently diverges the Numba vs NumPy step paths' INPUTS (the inlet writes state.f/state.mass/state.flags between steps — are those writes identical regardless of which step path runs, including dtypes).` },
]

const FINDINGS = {
  type: 'object',
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          file: { type: 'string' },
          line: { type: 'integer' },
          claim: { type: 'string' },
          scenario: { type: 'string' },
          severity: { type: 'string', enum: ['critical', 'major', 'minor'] },
        },
        required: ['file', 'claim', 'scenario', 'severity'],
      },
    },
  },
  required: ['findings'],
}

const VERDICT = {
  type: 'object',
  properties: {
    refuted: { type: 'boolean' },
    reason: { type: 'string' },
  },
  required: ['refuted', 'reason'],
}

const refuterPrompt = (f, angle) => `You are an adversarial verifier for a code-review finding in the git repo at "${REPO}" (review the COMMITTED state of range 460ca0f..57b07d7 via git show — the working tree may hold unrelated work in progress). READ-ONLY: no edits, no test runs, no simulations, no importing core/ or physics/ modules (a long job is running on this machine).
FINDING: ${f.claim}
LOCATION: ${f.file}${f.line ? ':' + f.line : ''}
CLAIMED FAILURE SCENARIO: ${f.scenario}
SEVERITY CLAIMED: ${f.severity}
Your angle: ${angle}. Read the actual code and try to REFUTE this finding — show the guard the reviewer missed, the invariant that makes the scenario unreachable, the consumer that actually handles it, or the reason the "wrong" value is correct. If after reading the code the scenario genuinely holds, say refuted=false and state the exact line(s) that make it real. Default to refuted=true if you cannot ground the failure in specific code you read.`

const ANGLES = [
  'trace the exact code path of the claimed scenario line by line',
  'hunt for guards, clamps, invariants, or callers that make the scenario unreachable',
  'check whether the claimed consequence is actually wrong behavior or is intended/benign',
]

const results = await pipeline(
  LENSES,
  (lens) => agent(lens.prompt, { label: `review:${lens.key}`, phase: 'Review', schema: FINDINGS }),
  (rev, lens) => {
    if (!rev || !rev.findings || !rev.findings.length) return []
    return parallel(rev.findings.slice(0, 6).map((f) => () =>
      parallel(ANGLES.map((angle, i) => () =>
        agent(refuterPrompt(f, angle), { label: `verify:${lens.key}:${(f.file || '').split('/').pop()}#${f.line || '?'}v${i}`, phase: 'Verify', schema: VERDICT })
      )).then((votes) => {
        const valid = votes.filter(Boolean)
        const upheld = valid.filter((v) => !v.refuted).length
        return {
          lens: lens.key,
          ...f,
          upheld_votes: upheld,
          total_votes: valid.length,
          survives: upheld >= 2,
          refutations: valid.filter((v) => v.refuted).map((v) => v.reason),
          confirmations: valid.filter((v) => !v.refuted).map((v) => v.reason),
        }
      })
    ))
  }
)

const all = results.filter(Boolean).flat().filter(Boolean)
const confirmed = all.filter((f) => f.survives)
const killed = all.filter((f) => !f.survives)
log(`${all.length} raw findings; ${confirmed.length} survived adversarial verification, ${killed.length} refuted`)
return {
  confirmed,
  refuted_count: killed.length,
  refuted_summaries: killed.map((f) => `${f.lens}: ${f.claim}`),
}