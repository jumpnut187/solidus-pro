import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Rotate3d } from 'lucide-react';

import { CanvasStage } from './components/CanvasStage';
import { IconRail } from './components/IconRail';
import { ResultsPanel } from './components/ResultsPanel';
import { TopBar } from './components/TopBar';
import { TransportBar } from './components/TransportBar';
import { AnalysisPanel } from './components/panels/AnalysisPanel';
import { MaterialPanel } from './components/panels/MaterialPanel';
import { MeshPanel, type MeshStatus } from './components/panels/MeshPanel';
import { ModelPanel } from './components/panels/ModelPanel';
import { OrientPanel } from './components/panels/OrientPanel';
import { PartsPanel } from './components/panels/PartsPanel';
import { PerformancePanel } from './components/panels/PerformancePanel';
import { ProjectsPanel } from './components/panels/ProjectsPanel';
import { SolidificationPanel } from './components/panels/SolidificationPanel';
import { ConfirmDialog, PromptDialog, type ConfirmRequest, type PromptRequest } from './components/modals/ConfirmDialog';
import { AboutModal, HardwareInfoModal, HelpGuideModal, ShortcutsModal } from './components/modals/InfoModals';
import { AnimationModal } from './components/modals/AnimationModal';
import { ReportModal } from './components/modals/ReportModal';
import { LogDrawer } from './components/LogDrawer';
import { useJobLog } from './hooks/useJobLog';
import { recordAnimation } from './lib/animationExport';
import { StatisticsModal } from './components/modals/StatisticsModal';
import { OpenProjectModal } from './components/modals/OpenProjectModal';
import { SolidificationModal } from './components/modals/SolidificationModal';
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts';
import { useModeling } from './hooks/useModeling';
import { useOrient } from './hooks/useOrient';
import { useSimulationJob } from './hooks/useSimulationJob';
import { useToasts } from './hooks/useToasts';
import {
  api,
  parsePartingLine,
  saveFile,
  type AlloyGuideRow,
  type HardwareInfo,
  type MaterialInfo,
  type PartingLine,
  type ProjectPart,
  type ProjectRecord,
  type ProjectSettings,
  type ProjectSummary,
  type ReportFormat,
  type ReportSectionToggles
} from './lib/api';
import { arrangeGrid, arrangeLine, DUPLICATE_OFFSET, type Axis } from './lib/arrange';
import {
  analysisDefaults,
  buildSolidificationDefaults,
  DEFAULT_PROJECT_NAME,
  defaultSettings,
  isPlaceholderProjectName,
  loadStoredSettings,
  meshDefaults,
  performanceDefaults,
  projectNameFromFile,
  saveStoredSettings
} from './lib/defaults';
import { formatCount } from './lib/format';
import { inferPartType } from './lib/partTypes';
import {
  describePartChanges,
  simulatedPartsFromSummary,
  simulatedPartsKey,
  type SimulatedPart
} from './lib/resultsFreshness';
import { parseCutFeatures, parseReferencePlanes, type CutFeature, type ReferencePlane } from './lib/modeling/types';
import {
  loadShellPrefs,
  loadStoredProjectId,
  saveShellPrefs,
  saveStoredProjectId,
  type PanelId,
  type ShellPrefs
} from './lib/uiPrefs';
import {
  isSweepMode,
  isTimeMode,
  type GridMeta,
  type MeasureResult,
  type ProbeReading,
  type ResultViewState,
  type ViewportHandle
} from './lib/viewState';

const CAD_PATTERN = /\.(stl|step|stp)$/i;
/** The canvas status line while an Orient click is armed. */
const ORIENT_PICK_STATUS = {
  faceDown: 'Face down: click a face of the part (Esc to cancel)',
  faceUp: 'Face up: click a face of the part (Esc to cancel)',
  pickHeight: 'Pick height: click a point for the parting line (Esc to cancel)'
} as const;
/** Playback tick. 90 ms matches the desktop timer; speed scales the step. */
const PLAYBACK_INTERVAL_MS = 90;
const PLAYBACK_STEP_PERCENT = 1;
/** How long to wait after the last edit before persisting settings. */
const AUTOSAVE_DEBOUNCE_MS = 700;

type ModalId =
  | 'openProject'
  | 'solidification'
  | 'statistics'
  | 'report'
  | 'animation'
  | 'help'
  | 'hardware'
  | 'about'
  | 'shortcuts'
  | null;

export function App() {
  const [settings, setSettings] = useState<ProjectSettings>(loadStoredSettings);
  const [shell, setShell] = useState<ShellPrefs>(loadShellPrefs);
  // Seeded from storage, not from an effect: the effect that persists this id
  // runs on mount too, and starting at null made it erase the very id the
  // restore was about to read.
  const [projectId, setProjectId] = useState<string | null>(loadStoredProjectId);
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [projectsLoading, setProjectsLoading] = useState(false);
  const [parts, setParts] = useState<ProjectPart[]>([]);
  const [referencePlanes, setReferencePlanes] = useState<ReferencePlane[]>([]);
  const [cutFeatures, setCutFeatures] = useState<CutFeature[]>([]);
  const [selectedPartId, setSelectedPartId] = useState<string | null>(null);
  /** The project's flat parting line (`ui_state.parting_line`), or null when none is set. */
  const [partingLine, setPartingLine] = useState<PartingLine | null>(null);
  /** The project whose "orient this casting" prompt was dismissed this session. */
  const [orientPromptDismissed, setOrientPromptDismissed] = useState<string | null>(null);
  const [materials, setMaterials] = useState<MaterialInfo[]>([]);
  const [hardware, setHardware] = useState<HardwareInfo | null>(null);
  const [alloyGuide, setAlloyGuide] = useState<AlloyGuideRow[]>([]);
  /** Set when the API cannot be reached at all, so panels can say why. */
  const [offline, setOffline] = useState(false);
  const [referenceAttempt, setReferenceAttempt] = useState(0);

  const [resultsAvailable, setResultsAvailable] = useState(false);
  const [resultsVersion, setResultsVersion] = useState(0);
  /** The parts the loaded run simulated; null when unknown (an older run) or no run. */
  const [simulatedParts, setSimulatedParts] = useState<SimulatedPart[] | null>(null);
  const [gridMeta, setGridMeta] = useState<GridMeta | null>(null);
  const [viewportStatus, setViewportStatus] = useState<string | null>(null);
  const [probe, setProbe] = useState<ProbeReading | null>(null);
  const [measurement, setMeasurement] = useState<MeasureResult | null>(null);
  const [defectVolume, setDefectVolume] = useState<number | null>(null);
  /**
   * Whether the "stored at reduced resolution" warning has been dismissed for
   * the run currently loaded. Keyed by resultsVersion rather than a boolean, so
   * dismissing it once does not silence it for every later run: a new run bumps
   * the version and the banner is due again.
   */
  const [downsampledDismissedFor, setDownsampledDismissedFor] = useState<number | null>(null);

  const [modal, setModal] = useState<ModalId>(null);
  const [confirmRequest, setConfirmRequest] = useState<ConfirmRequest | null>(null);
  const [promptRequest, setPromptRequest] = useState<PromptRequest | null>(null);

  const [uploading, setUploading] = useState(false);
  const [loadingProject, setLoadingProject] = useState(false);
  const [reportFormats, setReportFormats] = useState<ReportFormat[]>([]);
  const [reportBusy, setReportBusy] = useState(false);
  const [reportError, setReportError] = useState<string | null>(null);
  const [animationBusy, setAnimationBusy] = useState(false);
  const [animationProgress, setAnimationProgress] = useState(0);
  const [animationError, setAnimationError] = useState<string | null>(null);
  const [meshEstimate, setMeshEstimate] = useState<MeshStatus>(null);
  const [mlStatus, setMlStatus] = useState<MeshStatus>(null);
  const [loop, setLoop] = useState(true);

  const viewportRef = useRef<ViewportHandle | null>(null);
  const animationAbort = useRef<AbortController | null>(null);
  const importInputRef = useRef<HTMLInputElement | null>(null);
  const addPartsRef = useRef<() => void>(() => undefined);
  const skipAutosave = useRef(true);

  const toasts = useToasts();
  const { notify, succeed, fail, warn } = toasts;

  const view = shell.view;
  const patchShell = useCallback((patch: Partial<ShellPrefs>) => {
    setShell((current) => ({ ...current, ...patch }));
  }, []);
  const patchView = useCallback((patch: Partial<ResultViewState>) => {
    setShell((current) => ({ ...current, view: { ...current.view, ...patch } }));
  }, []);

  // --- Reference data ------------------------------------------------------
  // The alloy list is the canary: if it cannot be fetched the API is not
  // reachable, and every panel should say so rather than sitting on a
  // "Loading..." that will never resolve.
  useEffect(() => {
    let cancelled = false;
    api
      .materials()
      .then((payload) => {
        if (cancelled) return;
        setMaterials(payload);
        setOffline(false);
      })
      .catch(() => {
        if (!cancelled) setOffline(true);
      });
    api.hardware().then(setHardware).catch(() => setHardware(null));
    api
      .helpAlloys()
      .then((payload) => setAlloyGuide(payload.alloys))
      .catch(() => setAlloyGuide([]));
    api
      .reportFormats()
      .then((payload) => setReportFormats(payload.formats))
      .catch(() => setReportFormats([]));
    return () => {
      cancelled = true;
    };
  }, [referenceAttempt]);

  const refreshProjects = useCallback(async () => {
    setProjectsLoading(true);
    try {
      setProjects(await api.listProjects());
      setOffline(false);
    } catch (error) {
      // A failed list is the same signal as a failed alloy fetch; report it
      // through the banner rather than a toast the user has to dismiss.
      setOffline(true);
      void error;
    } finally {
      setProjectsLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshProjects();
  }, [refreshProjects]);

  // --- Persistence ---------------------------------------------------------
  useEffect(() => saveStoredSettings(settings), [settings]);
  useEffect(() => saveShellPrefs(shell), [shell]);
  useEffect(() => saveStoredProjectId(projectId), [projectId]);

  const projectIdRef = useRef(projectId);
  projectIdRef.current = projectId;

  const job = useSimulationJob({
    onCompleted: (finishedId) => {
      setResultsAvailable(true);
      setResultsVersion((version) => version + 1);
      // The run's record of what it simulated is in the project row; the parts
      // on screen may already have changed while it ran.
      setSimulatedParts(null);
      api
        .getProject(finishedId)
        .then(({ project }) => {
          if (projectIdRef.current === finishedId) setSimulatedParts(simulatedPartsFromSummary(project.summary));
        })
        .catch(() => undefined);
      patchView({ timePercent: 0, playing: false });
      succeed('Simulation complete.');
      // The project row is now `completed`; refresh so the list and the Open
      // dialog show it as having results.
      void refreshProjects();
    },
    onError: (message) => fail(message, 'Simulation failed')
  });

  // Mold filling can only run against an inlet point, so if the pour location
  // is cleared the analysis toggle has to come off with it -- the panel only
  // disables the checkbox, it does not clear a value that was already set.
  useEffect(() => {
    if (!settings.pour_location && settings.analysis_options.filling) {
      setSettings((current) => ({
        ...current,
        analysis_options: { ...current.analysis_options, filling: false }
      }));
    }
  }, [settings.pour_location, settings.analysis_options.filling]);

  // Debounced autosave. Panels apply live, so the project row has to keep up
  // without the user pressing anything -- but not on every keystroke, and not
  // immediately after a project is loaded (that would just echo it back).
  useEffect(() => {
    if (!projectId) return;
    if (skipAutosave.current) {
      skipAutosave.current = false;
      return;
    }
    const timer = window.setTimeout(() => {
      api.updateProject(projectId, settings).catch((error: unknown) => fail(error, 'Unable to save project'));
    }, AUTOSAVE_DEBOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [projectId, settings, fail]);

  // --- Playback ------------------------------------------------------------
  useEffect(() => {
    if (!view.playing || !(isTimeMode(view.mode) || isSweepMode(view.mode)) || !resultsAvailable) return;
    const timer = window.setInterval(() => {
      setShell((current) => {
        const next = current.view.timePercent + PLAYBACK_STEP_PERCENT * current.view.speed;
        if (next >= 100) {
          return {
            ...current,
            view: { ...current.view, timePercent: loop ? 0 : 100, playing: loop }
          };
        }
        return { ...current, view: { ...current.view, timePercent: next } };
      });
    }, PLAYBACK_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [view.playing, view.mode, view.speed, resultsAvailable, loop]);

  // --- Project lifecycle ---------------------------------------------------
  const applyProjectRecord = useCallback((project: ProjectRecord) => {
    // Merge over the defaults so a project saved before a setting existed still
    // opens with a complete, valid settings object.
    setSettings({
      name: project.name,
      alloy: project.alloy || 'WCB',
      voxel_size: project.voxel_size ?? meshDefaults.voxelSize,
      refinement_level: project.refinement_level ?? meshDefaults.refinement,
      memory_mode: project.memory_mode || 'auto',
      analysis_options: { ...analysisDefaults, ...(project.analysis_options ?? {}) },
      performance_options: { ...performanceDefaults, ...(project.performance_options ?? {}) },
      solidification_params: {
        ...buildSolidificationDefaults(),
        ...(project.solidification_params ?? {})
      },
      pour_location: project.pour_location ?? null
    });
  }, []);

  const openProject = useCallback(
    async (id: string, options?: { silent?: boolean }) => {
      try {
        setLoadingProject(true);
        const { project, parts: storedParts } = await api.getProject(id);
        skipAutosave.current = true;
        applyProjectRecord(project);
        setProjectId(id);
        setPartingLine(parsePartingLine(project.ui_state?.parting_line));
        setParts([]);
        setReferencePlanes(parseReferencePlanes(project.ui_state?.reference_planes));
        setCutFeatures(parseCutFeatures(project.ui_state?.cut_features));
        setSelectedPartId(storedParts[0]?.id ?? null);
        setGridMeta(null);
        setProbe(null);
        job.reset(project.status === 'completed' ? 'Results ready' : 'Ready');
        setResultsAvailable(project.status === 'completed');
        setSimulatedParts(project.status === 'completed' ? simulatedPartsFromSummary(project.summary) : null);
        setResultsVersion((version) => version + 1);
        if (!options?.silent) succeed(`Opened "${project.name}".`);

        // Geometry is not in the project row, so fetch each part's triangles in
        // parallel and append each part only once its mesh is in hand: the
        // renderer builds a part's geometry the first time it sees the part and
        // never revisits it, so a part added before its mesh arrives would stay
        // invisible for the rest of the session. Appending in original order
        // keeps the table stable however the requests interleave.
        const order = new Map(storedParts.map((part, index) => [part.id, index]));
        await Promise.all(
          storedParts.map(async (part) => {
            const meshData = await api.partMesh(id, part.id).catch(() => undefined);
            // A part whose file is gone is still listed; it just cannot render.
            const loaded = meshData ? { ...part, meshData } : part;
            setParts((current) =>
              [...current, loaded].sort((a, b) => (order.get(a.id) ?? 0) - (order.get(b.id) ?? 0))
            );
          })
        );
        return true;
      } catch (error) {
        if (options?.silent) {
          // A remembered project that no longer exists must not wedge startup:
          // fall back to an empty session instead of showing a dead id.
          setProjectId(null);
          saveStoredProjectId(null);
        } else {
          fail(error, 'Unable to open the project');
        }
        return false;
      } finally {
        setLoadingProject(false);
      }
    },
    [applyProjectRecord, fail, job, succeed]
  );

  // Restore the last project on load. This is the fix for the old client's
  // worst behaviour: the project id lived only in React state, so a refresh
  // silently discarded the parts, the settings and the results.
  const restoredRef = useRef(false);
  useEffect(() => {
    if (restoredRef.current) return;
    restoredRef.current = true;
    const stored = loadStoredProjectId();
    if (stored) void openProject(stored, { silent: true });
  }, [openProject]);

  /**
   * The project row for the current session, created on first need.
   *
   * `nextSettings` is for a caller that has just changed the settings in the
   * same event: the state update has not landed yet, so without it the row
   * would be written from the previous render's settings.
   */
  const ensureProject = useCallback(async (nextSettings?: ProjectSettings): Promise<string> => {
    const payload = nextSettings ?? settings;
    if (projectId) {
      await api.updateProject(projectId, payload);
      return projectId;
    }
    const created = await api.createProject(payload);
    skipAutosave.current = true;
    setProjectId(created.id);
    void refreshProjects();
    return created.id;
  }, [projectId, settings, refreshProjects]);

  const saveProject = useCallback(async () => {
    try {
      const id = await ensureProject();
      // Settings already persist on a debounce as they change, so this is a
      // confirmation, not a rescue. Saying so stops people wondering what they
      // would have lost by not pressing it.
      succeed('Project saved. Settings also save automatically as you change them.');
      void refreshProjects();
      return id;
    } catch (error) {
      fail(error, 'Unable to save the project');
      return null;
    }
  }, [ensureProject, fail, refreshProjects, succeed]);

  const modeling = useModeling({
    projectId,
    parts,
    setParts,
    setSelectedPartId,
    ensureProject: () => ensureProject(),
    viewportRef,
    referencePlanes,
    setReferencePlanes,
    cutFeatures,
    setCutFeatures,
    notify: { succeed, fail, warn }
  });
  const modelingActive = modeling.active;
  // A modeling session lives in the Model panel; starting one from anywhere
  // (the tree, a keyboard shortcut) brings the panel up with it.
  useEffect(() => {
    if (modelingActive) setShell((current) => (current.activePanel === 'model' ? current : { ...current, activePanel: 'model' }));
  }, [modelingActive]);
  const orientNotify = useMemo(() => ({ succeed, warn, fail }), [succeed, warn, fail]);
  const orient = useOrient({
    projectId,
    parts,
    setParts,
    selectedPartId,
    setSelectedPartId,
    partingLine,
    setPartingLine,
    active: shell.activePanel === 'orient',
    notify: orientNotify
  });
  const cancelModeling = modeling.cancel;
  // A sketch belongs to the project it was started in.
  useEffect(() => cancelModeling(), [projectId, cancelModeling]);

  const newProject = useCallback(() => {
    cancelModeling();
    setConfirmRequest({
      title: 'New project',
      message: 'Start a new project? The current one stays saved on the server and can be reopened.',
      confirmLabel: 'New project',
      onConfirm: () => {
        skipAutosave.current = true;
        setSettings(defaultSettings());
        setProjectId(null);
        setParts([]);
        setPartingLine(null);
        setReferencePlanes([]);
        setCutFeatures([]);
        setSelectedPartId(null);
        setResultsAvailable(false);
        setSimulatedParts(null);
        setGridMeta(null);
        setProbe(null);
        job.reset();
        patchShell({ activePanel: 'parts' });
        notify('Started a new project.');
      }
    });
  }, [cancelModeling, job, notify, patchShell]);

  const saveProjectAs = useCallback(() => {
    setPromptRequest({
      title: 'Save As',
      label: 'New project name',
      initialValue: `${settings.name || DEFAULT_PROJECT_NAME} (copy)`,
      confirmLabel: 'Save a copy',
      hint: 'Copies the parts and every setting. Results stay with the original.',
      onSubmit: (name) => {
        void (async () => {
          try {
            const sourceId = await ensureProject();
            const created = await api.duplicateProject(sourceId, name);
            await openProject(created.id, { silent: true });
            setSettings((current) => ({ ...current, name }));
            succeed(`Now working on the copy "${name}". Results stay with the original.`);
            void refreshProjects();
          } catch (error) {
            fail(error, 'Unable to save a copy');
          }
        })();
      }
    });
  }, [ensureProject, fail, openProject, refreshProjects, settings.name, succeed]);

  // --- Parts ---------------------------------------------------------------
  const addFiles = useCallback(
    async (files: File[]) => {
      const accepted = files.filter((file) => CAD_PATTERN.test(file.name));
      if (!accepted.length) {
        if (files.length) warn('Only STL, STEP and STP files can be added.');
        return;
      }
      setUploading(true);
      notify(`Uploading ${accepted.length} file${accepted.length === 1 ? '' : 's'}...`);
      try {
        // Every project used to be "Untitled Casting", which made the project
        // list a column of identical rows. The first part added to a project
        // that still has the placeholder name names it: the casting's own file
        // is the name a foundry already uses for the job, and taking it here
        // costs no extra dialog. A cast body is preferred over a riser or sprue
        // dropped alongside it. A name the user has typed is never replaced,
        // and neither is a project that already has parts.
        let named: ProjectSettings | undefined;
        if (parts.length === 0 && isPlaceholderProjectName(settings.name)) {
          const source = accepted.find((file) => inferPartType(file.name) === 'cast') ?? accepted[0];
          const fromFile = projectNameFromFile(source.name);
          if (fromFile) {
            named = { ...settings, name: fromFile };
            setSettings(named);
          }
        }
        const target = await ensureProject(named);
        const uploaded = await api.uploadFiles(accepted, settings.voxel_size);
        const added: ProjectPart[] = [];
        for (const file of uploaded.files) {
          // Prefer the server's clean display name; fall back to the stored
          // filename minus its extension only if an older server omits it.
          const name = file.display_name || file.filename.replace(/\.[^.]+$/, '');
          const response = await api.addPart(target, {
            name,
            part_type: file.part_type ?? inferPartType(name),
            filepath: file.filepath,
            position_x: 0,
            position_y: 0,
            position_z: 0,
            rotation_x: 0,
            rotation_y: 0,
            rotation_z: 0,
            scale: 1,
            visible: true,
            is_open_top: false,
            meshData: file.mesh_data
          });
          added.push({ ...response.part, meshData: file.mesh_data });
        }
        setParts((current) => [...current, ...added]);
        setSelectedPartId(added.at(-1)?.id ?? null);
        // A casting on its own is a fresh CAD import: CAD orientation is rarely
        // the pouring orientation, so the orient tool comes up with it.
        if (parts.length === 0 && added.length === 1 && added[0].part_type === 'cast') {
          patchShell({ activePanel: 'orient' });
        }
        succeed(`Added ${added.length} part${added.length === 1 ? '' : 's'}.`);
        void refreshProjects();
      } catch (error) {
        fail(error, 'Upload failed');
      } finally {
        setUploading(false);
      }
    },
    [ensureProject, fail, notify, parts.length, patchShell, refreshProjects, settings, succeed, warn]
  );

  const updatePart = useCallback(
    (partId: string, patch: Partial<ProjectPart>) => {
      let next: ProjectPart | null = null;
      setParts((current) =>
        current.map((part) => {
          if (part.id !== partId) return part;
          const merged = { ...part, ...patch };
          // Open-top only means anything on a riser; changing the type away
          // from riser has to clear the flag, not leave it set and invisible.
          if (patch.part_type && patch.part_type !== 'riser') merged.is_open_top = false;
          next = merged;
          return merged;
        })
      );
      if (!projectId || !next) return;
      api.updatePart(projectId, next).catch((error: unknown) => fail(error, 'Unable to update the part'));
    },
    [projectId, fail]
  );

  const rebuildDependents = modeling.rebuildDependents;
  const removePart = useCallback(
    (partId: string) => {
      // Before the part leaves state: the rebuild reads which bodies used it.
      rebuildDependents(partId);
      setParts((current) => current.filter((part) => part.id !== partId));
      setSelectedPartId((current) => (current === partId ? null : current));
      if (!projectId) return;
      api.deletePart(projectId, partId).catch((error: unknown) => fail(error, 'Unable to remove the part'));
    },
    [projectId, fail, rebuildDependents]
  );

  const duplicatePart = useCallback(
    (partId: string) => {
      const part = parts.find((item) => item.id === partId);
      if (!part || !projectId) return;
      void (async () => {
        try {
          const response = await api.addPart(projectId, {
            name: `${part.name} (copy)`,
            part_type: part.part_type,
            filepath: part.filepath,
            position_x: part.position_x + DUPLICATE_OFFSET[0],
            position_y: part.position_y + DUPLICATE_OFFSET[1],
            position_z: part.position_z + DUPLICATE_OFFSET[2],
            rotation_x: part.rotation_x,
            rotation_y: part.rotation_y,
            rotation_z: part.rotation_z,
            scale: part.scale,
            visible: part.visible,
            is_open_top: part.is_open_top,
            meshData: part.meshData
          });
          setParts((current) => [...current, { ...response.part, meshData: part.meshData }]);
          setSelectedPartId(response.part.id);
        } catch (error) {
          fail(error, 'Unable to duplicate the part');
        }
      })();
    },
    [parts, projectId, fail]
  );

  const clearAllParts = useCallback(() => {
    if (!parts.length) return;
    setConfirmRequest({
      title: 'Clear all parts',
      message: `Remove all ${parts.length} parts from this project? The uploaded files are deleted from the server.`,
      confirmLabel: 'Remove all',
      destructive: true,
      onConfirm: () => {
        const ids = parts.map((part) => part.id);
        setParts([]);
        setSelectedPartId(null);
        if (!projectId) return;
        void Promise.all(ids.map((id) => api.deletePart(projectId, id).catch(() => undefined))).then(() =>
          notify('All parts removed.')
        );
      }
    });
  }, [parts, projectId, notify]);

  const applyPositions = useCallback(
    (patches: { id: string; position_x: number; position_y: number; position_z: number }[]) => {
      if (!patches.length) {
        warn('Parts need loaded geometry before they can be arranged.');
        return;
      }
      const byId = new Map(patches.map((patch) => [patch.id, patch]));
      setParts((current) =>
        current.map((part) => {
          const patch = byId.get(part.id);
          return patch ? { ...part, ...patch } : part;
        })
      );
      if (!projectId) return;
      // Persist from the merged values rather than the stale `parts` closure.
      setParts((current) => {
        for (const part of current) {
          if (byId.has(part.id)) {
            api.updatePart(projectId, part).catch(() => undefined);
          }
        }
        return current;
      });
    },
    [projectId, warn]
  );

  const arrangeGridAction = useCallback(() => {
    setPromptRequest({
      title: 'Arrange in a grid',
      label: 'Space between parts (inches)',
      initialValue: '1.0',
      onSubmit: (value) => {
        const spacing = Number(value);
        if (!Number.isFinite(spacing) || spacing < 0) {
          warn('Enter a spacing of zero or more inches.');
          return;
        }
        applyPositions(arrangeGrid(parts, spacing));
      }
    });
  }, [applyPositions, parts, warn]);

  const arrangeLineAction = useCallback(() => {
    setPromptRequest({
      title: 'Arrange in a line',
      label: 'Axis (X, Y or Z)',
      initialValue: 'X',
      hint: 'Parts are spaced one inch apart along this axis and aligned to the origin on the others.',
      onSubmit: (value) => {
        const axis = value.trim().toLowerCase();
        if (axis !== 'x' && axis !== 'y' && axis !== 'z') {
          warn('Enter X, Y or Z.');
          return;
        }
        applyPositions(arrangeLine(parts, axis as Axis, 1));
      }
    });
  }, [applyPositions, parts, warn]);

  // --- Mesh ----------------------------------------------------------------
  const estimateMesh = useCallback(async () => {
    try {
      const target = await ensureProject();
      const preview = await api.meshPreview(target, settings.voxel_size);
      const detail = `${formatCount(preview.estimated_voxels)} voxels, ${preview.estimated_memory_mb.toFixed(0)} MB`;
      setMeshEstimate({
        text: preview.can_proceed
          ? detail
          : `${detail}. ${preview.message} Safe voxel size: ${preview.recommended_voxel_size.toFixed(3)} in.`,
        tone: preview.can_proceed ? 'success' : 'warning'
      });
    } catch (error) {
      fail(error, 'Mesh estimate failed');
    }
  }, [ensureProject, settings.voxel_size, fail]);

  const applyMlMesh = useCallback(async () => {
    setMlStatus({ text: 'Requesting a recommendation...', tone: 'info' });
    try {
      const target = await ensureProject();
      const recommendation = await api.mlMesh(target);
      setSettings((current) => ({
        ...current,
        voxel_size: recommendation.base_voxel_size,
        refinement_level: recommendation.refinement_level
      }));
      const source = recommendation.used_model ? 'ML model' : 'heuristic';
      setMlStatus({
        text:
          `${source}: ${recommendation.base_voxel_size.toFixed(3)} in at refinement ${recommendation.refinement_level} ` +
          `(confidence ${recommendation.confidence.toFixed(2)}). ${recommendation.reason}` +
          (recommendation.warning ? ` Warning: ${recommendation.warning}` : ''),
        tone: recommendation.warning ? 'warning' : 'success'
      });
    } catch (error) {
      setMlStatus({ text: error instanceof Error ? error.message : 'ML mesh failed', tone: 'warning' });
    }
  }, [ensureProject]);

  // --- Run -----------------------------------------------------------------
  const runSimulation = useCallback(async () => {
    if (!parts.length) {
      warn('Add at least one CAD part before running.');
      return;
    }
    if (job.isBusy) return;
    try {
      const target = await ensureProject();
      setResultsAvailable(false);
      setSimulatedParts(null);
      setGridMeta(null);
      patchView({ playing: false, timePercent: 0 });
      await job.start(target);
    } catch (error) {
      fail(error, 'Unable to start the simulation');
    }
  }, [parts.length, job, ensureProject, patchView, fail, warn]);

  const stopSimulation = useCallback(() => {
    if (!projectId || !job.isBusy) return;
    void job.cancel(projectId);
  }, [projectId, job]);

  // --- Screenshot ----------------------------------------------------------
  const takeScreenshot = useCallback(async () => {
    const handle = viewportRef.current;
    if (!handle) return;
    try {
      const blob = await handle.screenshot();
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
      link.download = `${(settings.name ?? 'solidus').replace(/[^\w-]+/g, '_')}_${stamp}.png`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      // Revoke on the next tick; revoking synchronously can beat the download.
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
      succeed('Screenshot saved.');
    } catch (error) {
      fail(error, 'Screenshot failed');
    }
  }, [settings.name, succeed, fail]);

  const jobLog = useJobLog(projectId, shell.logOpen, job.isBusy);

  const clearMeasurements = useCallback(() => {
    viewportRef.current?.clearMeasurements();
    setMeasurement(null);
  }, []);

  /**
   * Ask the viewport for the volume above the current threshold.
   *
   * Deliberately on demand rather than recomputed on every threshold tick: the
   * handle walks the whole dense grid, which is not something to do sixty
   * times a second while a slider is being dragged.
   */
  const recomputeDefectVolume = useCallback(() => {
    const value = viewportRef.current?.defectVolumeIn3() ?? null;
    setDefectVolume(value);
    if (value === null) warn('Defect volume is only defined for threshold modes.');
  }, [warn]);

  /**
   * What changed since the loaded run, or null while the results still match
   * the parts. Not judged while a project is loading: its parts arrive one at
   * a time and would look like deletions until the last one lands.
   */
  const staleResults = useMemo(() => {
    if (!resultsAvailable || !simulatedParts || loadingProject || job.isBusy) return null;
    if (simulatedPartsKey(simulatedParts) === simulatedPartsKey(parts)) return null;
    return describePartChanges(simulatedParts, parts);
  }, [resultsAvailable, simulatedParts, loadingProject, job.isBusy, parts]);

  // A stale volume is worse than none: it is a number for a threshold the user
  // has already moved past.
  useEffect(() => setDefectVolume(null), [view.threshold, view.mode, view.castOnly, resultsVersion]);

  // --- Exports -------------------------------------------------------------
  const generateReport = useCallback(
    async (format: string, sections: ReportSectionToggles) => {
      if (!projectId) return;
      setReportBusy(true);
      setReportError(null);
      try {
        let screenshot: string | undefined;
        if (sections.screenshot && viewportRef.current) {
          const blob = await viewportRef.current.screenshot({ width: 1600, height: 1000 });
          screenshot = await new Promise<string>((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(String(reader.result));
            reader.onerror = () => reject(new Error('Could not read the screenshot'));
            reader.readAsDataURL(blob);
          });
        }
        saveFile(await api.generateReport(projectId, { format, sections, screenshot }));
        setModal(null);
        succeed('Report downloaded.');
      } catch (error) {
        setReportError(error instanceof Error ? error.message : 'Report generation failed');
      } finally {
        setReportBusy(false);
      }
    },
    [projectId, succeed]
  );

  const recordAnimationExport = useCallback(
    async ({ fps, frames }: { fps: number; frames: number }) => {
      const canvas = document.querySelector<HTMLCanvasElement>('canvas.viewport-canvas');
      if (!canvas) {
        setAnimationError('The viewport is not ready.');
        return;
      }
      const controller = new AbortController();
      animationAbort.current = controller;
      setAnimationBusy(true);
      setAnimationError(null);
      setAnimationProgress(0);
      // Playback must not fight the recorder for the time slider.
      patchView({ playing: false });
      try {
        const { blob, extension } = await recordAnimation(canvas, {
          fps,
          frames,
          signal: controller.signal,
          onSeek: (timePercent) => patchView({ timePercent }),
          onProgress: setAnimationProgress
        });
        const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
        const base = (settings.name ?? 'solidus').replace(/[^\w-]+/g, '_');
        saveFile({ blob, filename: `${base}_${view.mode}_${stamp}.${extension}` });
        setModal(null);
        succeed('Animation downloaded.');
      } catch (error) {
        setAnimationError(error instanceof Error ? error.message : 'Recording failed');
      } finally {
        animationAbort.current = null;
        setAnimationBusy(false);
      }
    },
    [patchView, settings.name, succeed, view.mode]
  );

  const exportProjectFile = useCallback(async () => {
    if (!projectId) {
      warn('Save the project before exporting it.');
      return;
    }
    try {
      saveFile(await api.exportProjectFile(projectId));
      succeed('Project file downloaded.');
    } catch (error) {
      fail(error, 'Export failed');
    }
  }, [projectId, succeed, fail, warn]);

  const importProjectFiles = useCallback(
    async (files: File[]) => {
      const projectFile = files.find((file) => file.name.endsWith('.json'));
      if (!projectFile) {
        warn('Select a .solidus-project.json file, together with its CAD files.');
        return;
      }
      const cadFiles = files.filter((file) => file !== projectFile && CAD_PATTERN.test(file.name));
      try {
        const response = await api.importProjectFile(projectFile, cadFiles);
        await openProject(response.id, { silent: true });
        void refreshProjects();
        if (response.missing_parts.length) {
          warn(
            `Imported, but ${response.missing_parts.length} part file was not supplied: ` +
              `${response.missing_parts.join(', ')}. Add the CAD files and re-import to restore them.`
          );
        } else {
          succeed(`Imported "${response.project.name}".`);
        }
      } catch (error) {
        fail(error, 'Import failed');
      }
    },
    [openProject, refreshProjects, succeed, warn, fail]
  );

  // --- Keyboard ------------------------------------------------------------
  const closeTopLayer = useCallback(() => {
    if (promptRequest) return setPromptRequest(null);
    if (confirmRequest) return setConfirmRequest(null);
    if (modal) return setModal(null);
    if (orient.pick) return orient.cancelPick();
    // In a sketch Escape belongs to the sketcher (it backs out of the shape
    // being drawn, then the tool); only the plane step cancels outright.
    if (modeling.session) {
      if (modeling.session.stage === 'plane') modeling.cancel();
      return;
    }
    // The Measure tool clears its selections on Escape itself; the plane
    // manager cancels.
    if (modeling.planeSession) return modeling.cancel();
    if (modeling.measurePicks) return;
    if (shell.tool !== 'select') return patchShell({ tool: 'select' });
    if (shell.activePanel) return patchShell({ activePanel: null });
  }, [promptRequest, confirmRequest, modal, orient, modeling, shell.tool, shell.activePanel, patchShell]);

  useKeyboardShortcuts({
    run: () => void runSimulation(),
    stop: stopSimulation,
    addPart: () => {
      patchShell({ activePanel: 'parts' });
      // The panel mounts on the next paint; open its picker after that.
      window.setTimeout(() => addPartsRef.current(), 0);
    },
    screenshot: () => void takeScreenshot(),
    saveProject: () => void saveProject(),
    saveProjectAs,
    fitCamera: () => viewportRef.current?.setCamera('fit'),
    helpGuide: () => setModal('help'),
    shortcuts: () => setModal('shortcuts'),
    escape: closeTopLayer,
    toggleOrient: () =>
      setShell((current) => ({ ...current, activePanel: current.activePanel === 'orient' ? null : 'orient' })),
    togglePlayback: () => {
      if ((isTimeMode(view.mode) || isSweepMode(view.mode)) && resultsAvailable) {
        patchView({ playing: !view.playing });
      }
    }
  });

  // --- Project list actions ------------------------------------------------
  const renameProject = useCallback(
    (project: ProjectSummary) => {
      setPromptRequest({
        title: 'Rename project',
        label: 'Project name',
        initialValue: project.name,
        confirmLabel: 'Rename',
        onSubmit: (name) => {
          void api
            .renameProject(project.id, name)
            .then(() => {
              if (project.id === projectId) setSettings((current) => ({ ...current, name }));
              void refreshProjects();
              succeed(`Renamed to "${name}".`);
            })
            .catch((error: unknown) => fail(error, 'Unable to rename the project'));
        }
      });
    },
    [projectId, refreshProjects, succeed, fail]
  );

  const duplicateProject = useCallback(
    (project: ProjectSummary) => {
      void api
        .duplicateProject(project.id)
        .then((created) => {
          void refreshProjects();
          succeed(`Duplicated as "${created.project.name}".`);
        })
        .catch((error: unknown) => fail(error, 'Unable to duplicate the project'));
    },
    [refreshProjects, succeed, fail]
  );

  const deleteProject = useCallback(
    (project: ProjectSummary) => {
      setConfirmRequest({
        title: 'Delete project',
        message: `Delete "${project.name}"? Its parts, uploaded files and results are removed permanently.`,
        confirmLabel: 'Delete',
        destructive: true,
        onConfirm: () => {
          void api
            .deleteProject(project.id)
            .then(() => {
              if (project.id === projectId) {
                setProjectId(null);
                setParts([]);
                setPartingLine(null);
                setSelectedPartId(null);
                setResultsAvailable(false);
                setGridMeta(null);
              }
              void refreshProjects();
              notify(`Deleted "${project.name}".`);
            })
            .catch((error: unknown) => fail(error, 'Unable to delete the project'));
        }
      });
    },
    [projectId, refreshProjects, notify, fail]
  );

  // --- Derived -------------------------------------------------------------
  // A project that is one casting and nothing else, with no parting line yet,
  // is a fresh import that has not been oriented: offer the tool.
  const showOrientPrompt =
    parts.length === 1 &&
    parts[0].part_type === 'cast' &&
    partingLine === null &&
    shell.activePanel !== 'orient' &&
    orientPromptDismissed !== projectId &&
    !modelingActive &&
    !loadingProject;
  const hasPourLocation = Boolean(settings.pour_location);
  const closePanel = useCallback(() => patchShell({ activePanel: null }), [patchShell]);
  const setPanel = useCallback((panel: PanelId | null) => patchShell({ activePanel: panel }), [patchShell]);

  const panel = useMemo(() => {
    switch (shell.activePanel) {
      case 'parts':
        return (
          <PartsPanel
            parts={parts}
            selectedPartId={selectedPartId}
            onSelectPart={setSelectedPartId}
            onAddFiles={(files) => void addFiles(files)}
            onUpdatePart={updatePart}
            onRemovePart={removePart}
            onDuplicatePart={duplicatePart}
            onClearAll={clearAllParts}
            onArrangeGrid={arrangeGridAction}
            onArrangeLine={arrangeLineAction}
            onClose={closePanel}
            pourLocation={settings.pour_location ?? null}
            tool={shell.tool}
            onSetTool={(tool) => patchShell({ tool })}
            onClearPourLocation={() => {
              setSettings((current) => ({ ...current, pour_location: null }));
              patchShell({ tool: 'select' });
            }}
            uploading={uploading}
          />
        );
      case 'orient':
        return <OrientPanel orient={orient} onClose={closePanel} />;
      case 'model':
        return (
          <ModelPanel
            modeling={modeling}
            parts={parts}
            selectedPartId={selectedPartId}
            onSelectPart={setSelectedPartId}
            onRemovePart={removePart}
            onClose={closePanel}
          />
        );
      case 'material':
        return (
          <MaterialPanel
            alloy={settings.alloy}
            materials={materials}
            offline={offline}
            onSelectAlloy={(alloy) => setSettings((current) => ({ ...current, alloy }))}
            onClose={closePanel}
          />
        );
      case 'mesh':
        return (
          <MeshPanel
            settings={settings}
            onPatch={(patch) => setSettings((current) => ({ ...current, ...patch }))}
            onEstimate={() => void estimateMesh()}
            onApplyMlMesh={() => void applyMlMesh()}
            estimate={meshEstimate}
            mlStatus={mlStatus}
            busy={job.isBusy}
            hasParts={parts.length > 0}
            onReset={() =>
              setSettings((current) => ({
                ...current,
                voxel_size: meshDefaults.voxelSize,
                refinement_level: meshDefaults.refinement,
                memory_mode: 'auto'
              }))
            }
            onClose={closePanel}
          />
        );
      case 'analysis':
        return (
          <AnalysisPanel
            options={settings.analysis_options}
            alloy={settings.alloy}
            hasPourLocation={hasPourLocation}
            onToggle={(key, value) =>
              setSettings((current) => ({
                ...current,
                analysis_options: { ...current.analysis_options, [key]: value }
              }))
            }
            onNumber={(key, value) =>
              setSettings((current) => ({
                ...current,
                analysis_options: { ...current.analysis_options, [key]: value }
              }))
            }
            onReset={() => setSettings((current) => ({ ...current, analysis_options: { ...analysisDefaults } }))}
            onClose={closePanel}
          />
        );
      case 'performance':
        return (
          <PerformancePanel
            options={settings.performance_options}
            hardware={hardware}
            onSet={(key, value) =>
              setSettings((current) => ({
                ...current,
                performance_options: { ...current.performance_options, [key]: value }
              }))
            }
            onReset={() =>
              setSettings((current) => ({ ...current, performance_options: { ...performanceDefaults } }))
            }
            onClose={closePanel}
          />
        );
      case 'solidification':
        return (
          <SolidificationPanel
            params={settings.solidification_params}
            onOpenModal={() => setModal('solidification')}
            onReset={() =>
              setSettings((current) => ({ ...current, solidification_params: buildSolidificationDefaults() }))
            }
            onClose={closePanel}
          />
        );
      case 'projects':
        return (
          <ProjectsPanel
            projects={projects}
            currentProjectId={projectId}
            loading={projectsLoading}
            offline={offline}
            onOpen={(id) => void openProject(id)}
            onRefresh={() => void refreshProjects()}
            onNew={newProject}
            onBrowse={() => setModal('openProject')}
            onClose={closePanel}
          />
        );
      default:
        return null;
    }
  }, [
    shell.activePanel,
    shell.tool,
    modeling,
    orient,
    parts,
    selectedPartId,
    addFiles,
    updatePart,
    removePart,
    duplicatePart,
    clearAllParts,
    arrangeGridAction,
    arrangeLineAction,
    closePanel,
    settings,
    patchShell,
    uploading,
    materials,
    estimateMesh,
    applyMlMesh,
    meshEstimate,
    mlStatus,
    job.isBusy,
    hasPourLocation,
    hardware,
    projects,
    projectId,
    projectsLoading,
    openProject,
    refreshProjects,
    newProject,
    offline
  ]);

  return (
    <div className="app">
      <TopBar
        projectName={settings.name ?? ''}
        onProjectNameChange={(name) => setSettings((current) => ({ ...current, name }))}
        onNewProject={newProject}
        onOpenProject={() => {
          void refreshProjects();
          setModal('openProject');
        }}
        onSaveProject={() => void saveProject()}
        onSaveProjectAs={saveProjectAs}
        onRun={() => void runSimulation()}
        onStop={stopSimulation}
        canRun={parts.length > 0}
        isBusy={job.isBusy}
        job={staleResults !== null ? { ...job.job, message: 'Results out of date' } : job.job}
        onScreenshot={() => void takeScreenshot()}
        onExport={(kind) => {
          if (kind === 'screenshot') void takeScreenshot();
          else if (kind === 'animation') setModal('animation');
          else if (kind === 'report') setModal('report');
          else if (kind === 'projectFile') void exportProjectFile();
          else if (kind === 'importProjectFile') importInputRef.current?.click();
        }}
        onHelp={(topic) => setModal(topic === 'guide' ? 'help' : topic)}
        exportsAvailable={{
          animation: resultsAvailable,
          report: Boolean(projectId),
          projectFile: Boolean(projectId)
        }}
      />

      {offline ? (
        <div className="connection-banner" role="alert">
          <span>
            Cannot reach the simulation server. Panels will stay empty and runs cannot start until it
            responds.
          </span>
          <button
            type="button"
            className="btn btn-sm"
            onClick={() => {
              setReferenceAttempt((attempt) => attempt + 1);
              void refreshProjects();
            }}
          >
            Retry
          </button>
        </div>
      ) : null}

      <div className="workspace">
        <IconRail active={shell.activePanel} onSelect={setPanel} partCount={parts.length} />
        {panel}
        <CanvasStage
          viewportRef={viewportRef}
          projectId={projectId}
          parts={parts}
          selectedPartId={selectedPartId}
          onSelectPart={setSelectedPartId}
          resultsAvailable={resultsAvailable}
          resultsVersion={resultsVersion}
          // Modeling is done on the bodies themselves: the result field would
          // only hide the faces being sketched on.
          view={modelingActive ? { ...view, showResults: false } : view}
          onViewChange={patchView}
          modeling={modeling}
          ortho={shell.ortho}
          tool={shell.tool}
          probe={probe}
          onProbe={setProbe}
          onMeasure={(result) => {
            setMeasurement(result);
            notify(`Distance: ${result.distanceIn.toFixed(3)} in`);
          }}
          pourLocation={settings.pour_location ?? null}
          onPourPicked={(pick) => {
            setSettings((current) => ({
              ...current,
              pour_location: {
                schema_version: 1,
                point_in: pick.point,
                normal: pick.normal,
                part_id: pick.partId
              }
            }));
            patchShell({ tool: 'select' });
            succeed('Pour location set.');
          }}
          gridMeta={gridMeta}
          onGridMeta={setGridMeta}
          status={orient.pick ? ORIENT_PICK_STATUS[orient.pick] : viewportStatus}
          onStatus={setViewportStatus}
          onError={(message) => fail(message)}
          onAddFiles={(files) => void addFiles(files)}
          busy={loadingProject || uploading}
          hasParts={parts.length > 0}
          onOrthoChange={(next) => patchShell({ ortho: next })}
          onToolChange={(next) => patchShell({ tool: next })}
          onCamera={(preset) => viewportRef.current?.setCamera(preset)}
          onClearMeasurements={clearMeasurements}
          measurement={measurement}
          toasts={toasts.toasts}
          onDismissToast={toasts.dismiss}
          showDownsampledWarning={
            resultsAvailable && Boolean(gridMeta?.downsampled) && downsampledDismissedFor !== resultsVersion
          }
          onDismissDownsampledWarning={() => setDownsampledDismissedFor(resultsVersion)}
          staleResults={staleResults}
          orient={orient.viewState}
          onOrientPick={orient.handlePick}
          prompt={
            showOrientPrompt ? (
              <div className="canvas-banner orient-prompt" role="status">
                <Rotate3d size={14} aria-hidden="true" className="canvas-banner-icon" />
                <span className="canvas-banner-text">
                  <b>Orient this casting for pouring.</b> CAD orientation is rarely how it is cast. Turn it, set its
                  height and place the parting line before adding risers.
                </span>
                <button type="button" className="btn btn-sm" onClick={() => patchShell({ activePanel: 'orient' })}>
                  Orient &amp; Parting Line
                </button>
                <button
                  type="button"
                  className="btn btn-ghost btn-sm"
                  aria-label="Dismiss orient prompt"
                  onClick={() => setOrientPromptDismissed(projectId)}
                >
                  Not now
                </button>
              </div>
            ) : null
          }
          onRunAgain={() => void runSimulation()}
          running={job.isBusy}
          onRequestAddParts={() => {
            patchShell({ activePanel: 'parts' });
            window.setTimeout(() => addPartsRef.current(), 0);
          }}
          registerAddParts={(open) => {
            addPartsRef.current = open;
          }}
          log={
            <LogDrawer
              open={shell.logOpen}
              onToggle={() => patchShell({ logOpen: !shell.logOpen })}
              lines={jobLog.lines}
              available={jobLog.available}
              dropped={jobLog.dropped}
              isRunning={job.isBusy}
            />
          }
        />
        <ResultsPanel
          collapsed={shell.resultsCollapsed}
          onToggleCollapsed={() => patchShell({ resultsCollapsed: !shell.resultsCollapsed })}
          sectionsCollapsed={shell.sectionsCollapsed}
          onSectionToggle={(id, next) =>
            patchShell({ sectionsCollapsed: { ...shell.sectionsCollapsed, [id]: next } })
          }
          view={view}
          onViewChange={patchView}
          meta={gridMeta}
          resultsAvailable={resultsAvailable}
          measurement={measurement}
          onClearMeasurements={clearMeasurements}
          defectVolumeIn3={defectVolume}
          onRecomputeDefectVolume={recomputeDefectVolume}
          onOpenStatistics={() => setModal('statistics')}
        />
      </div>

      <TransportBar
        view={view}
        onViewChange={patchView}
        meta={gridMeta}
        resultsAvailable={resultsAvailable}
        loop={loop}
        onLoopChange={setLoop}
      />

      {modal === 'openProject' ? (
        <OpenProjectModal
          projects={projects}
          currentProjectId={projectId}
          loading={projectsLoading}
          onOpen={(id) => {
            setModal(null);
            void openProject(id);
          }}
          onRename={renameProject}
          onDuplicate={duplicateProject}
          onDelete={deleteProject}
          onRefresh={() => void refreshProjects()}
          onClose={() => setModal(null)}
        />
      ) : null}

      {modal === 'solidification' ? (
        <SolidificationModal
          params={settings.solidification_params}
          onChange={(params) => setSettings((current) => ({ ...current, solidification_params: params }))}
          onClose={() => setModal(null)}
        />
      ) : null}

      {modal === 'statistics' ? <StatisticsModal meta={gridMeta} onClose={() => setModal(null)} /> : null}
      {modal === 'report' ? (
        <ReportModal
          formats={reportFormats}
          busy={reportBusy}
          error={reportError}
          canCaptureView={parts.length > 0}
          onGenerate={(format, sections) => void generateReport(format, sections)}
          onClose={() => setModal(null)}
        />
      ) : null}
      {modal === 'animation' ? (
        <AnimationModal
          busy={animationBusy}
          progress={animationProgress}
          error={animationError}
          // A sweep has no stored frames to match, so the recorder is left to
          // its own default rather than being told a count that does not exist.
          frameCount={
            isSweepMode(view.mode)
              ? 0
              : (gridMeta?.frames.fraction_liquid?.count ?? gridMeta?.frames.fill?.count ?? 0)
          }
          onRecord={(options) => void recordAnimationExport(options)}
          onCancel={() => {
            animationAbort.current?.abort();
            if (!animationBusy) setModal(null);
          }}
          onClose={() => setModal(null)}
        />
      ) : null}
      {modal === 'help' ? <HelpGuideModal alloys={alloyGuide} onClose={() => setModal(null)} /> : null}
      {modal === 'hardware' ? <HardwareInfoModal hardware={hardware} onClose={() => setModal(null)} /> : null}
      {modal === 'about' ? <AboutModal onClose={() => setModal(null)} /> : null}
      {modal === 'shortcuts' ? <ShortcutsModal onClose={() => setModal(null)} /> : null}

      {/* Project-file import: a project JSON plus the CAD files it references. */}
      <input
        ref={importInputRef}
        type="file"
        accept=".json,.stl,.step,.stp"
        multiple
        className="visually-hidden"
        onChange={(event) => {
          void importProjectFiles(Array.from(event.target.files ?? []));
          event.target.value = '';
        }}
      />

      {confirmRequest ? (
        <ConfirmDialog request={confirmRequest} onClose={() => setConfirmRequest(null)} />
      ) : null}
      {promptRequest ? <PromptDialog request={promptRequest} onClose={() => setPromptRequest(null)} /> : null}

    </div>
  );
}
