import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { CanvasStage } from './components/CanvasStage';
import { IconRail } from './components/IconRail';
import { ResultsPanel } from './components/ResultsPanel';
import { Toasts } from './components/CanvasOverlays';
import { TopBar } from './components/TopBar';
import { TransportBar } from './components/TransportBar';
import { AnalysisPanel } from './components/panels/AnalysisPanel';
import { MaterialPanel } from './components/panels/MaterialPanel';
import { MeshPanel, type MeshStatus } from './components/panels/MeshPanel';
import { PartsPanel } from './components/panels/PartsPanel';
import { PerformancePanel } from './components/panels/PerformancePanel';
import { ProjectsPanel } from './components/panels/ProjectsPanel';
import { SolidificationPanel } from './components/panels/SolidificationPanel';
import { ConfirmDialog, PromptDialog, type ConfirmRequest, type PromptRequest } from './components/modals/ConfirmDialog';
import { AboutModal, HardwareInfoModal, HelpGuideModal } from './components/modals/InfoModals';
import { OpenProjectModal } from './components/modals/OpenProjectModal';
import { SolidificationModal } from './components/modals/SolidificationModal';
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts';
import { useSimulationJob } from './hooks/useSimulationJob';
import { useToasts } from './hooks/useToasts';
import {
  api,
  type AlloyGuideRow,
  type HardwareInfo,
  type MaterialInfo,
  type ProjectPart,
  type ProjectRecord,
  type ProjectSettings,
  type ProjectSummary
} from './lib/api';
import { arrangeGrid, arrangeLine, DUPLICATE_OFFSET, type Axis } from './lib/arrange';
import { analysisDefaults, buildSolidificationDefaults, defaultSettings, loadStoredSettings, meshDefaults, performanceDefaults, saveStoredSettings } from './lib/defaults';
import { formatCount } from './lib/format';
import { inferPartType } from './lib/partTypes';
import {
  loadShellPrefs,
  loadStoredProjectId,
  saveShellPrefs,
  saveStoredProjectId,
  type PanelId,
  type ShellPrefs
} from './lib/uiPrefs';
import {
  isTimeMode,
  type CameraPreset,
  type GridMeta,
  type ProbeReading,
  type ResultViewState,
  type ViewportHandle,
  type ViewportTool
} from './lib/viewState';

const CAD_PATTERN = /\.(stl|step|stp)$/i;
/** Playback tick. 90 ms matches the desktop timer; speed scales the step. */
const PLAYBACK_INTERVAL_MS = 90;
const PLAYBACK_STEP_PERCENT = 1;
/** How long to wait after the last edit before persisting settings. */
const AUTOSAVE_DEBOUNCE_MS = 700;

type ModalId = 'openProject' | 'solidification' | 'help' | 'hardware' | 'about' | null;

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
  const [selectedPartId, setSelectedPartId] = useState<string | null>(null);
  const [materials, setMaterials] = useState<MaterialInfo[]>([]);
  const [hardware, setHardware] = useState<HardwareInfo | null>(null);
  const [alloyGuide, setAlloyGuide] = useState<AlloyGuideRow[]>([]);

  const [resultsAvailable, setResultsAvailable] = useState(false);
  const [resultsVersion, setResultsVersion] = useState(0);
  const [gridMeta, setGridMeta] = useState<GridMeta | null>(null);
  const [viewportStatus, setViewportStatus] = useState<string | null>(null);
  const [probe, setProbe] = useState<ProbeReading | null>(null);

  const [modal, setModal] = useState<ModalId>(null);
  const [confirmRequest, setConfirmRequest] = useState<ConfirmRequest | null>(null);
  const [promptRequest, setPromptRequest] = useState<PromptRequest | null>(null);

  const [uploading, setUploading] = useState(false);
  const [loadingProject, setLoadingProject] = useState(false);
  const [meshEstimate, setMeshEstimate] = useState<MeshStatus>(null);
  const [mlStatus, setMlStatus] = useState<MeshStatus>(null);
  const [loop, setLoop] = useState(true);

  const viewportRef = useRef<ViewportHandle | null>(null);
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
  useEffect(() => {
    api.materials().then(setMaterials).catch((error: unknown) => fail(error, 'Unable to load alloys'));
    api.hardware().then(setHardware).catch(() => setHardware(null));
    api
      .helpAlloys()
      .then((payload) => setAlloyGuide(payload.alloys))
      .catch(() => setAlloyGuide([]));
  }, [fail]);

  const refreshProjects = useCallback(async () => {
    setProjectsLoading(true);
    try {
      setProjects(await api.listProjects());
    } catch (error) {
      fail(error, 'Unable to list projects');
    } finally {
      setProjectsLoading(false);
    }
  }, [fail]);

  useEffect(() => {
    void refreshProjects();
  }, [refreshProjects]);

  // --- Persistence ---------------------------------------------------------
  useEffect(() => saveStoredSettings(settings), [settings]);
  useEffect(() => saveShellPrefs(shell), [shell]);
  useEffect(() => saveStoredProjectId(projectId), [projectId]);

  const job = useSimulationJob({
    onCompleted: () => {
      setResultsAvailable(true);
      setResultsVersion((version) => version + 1);
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
    if (!view.playing || !isTimeMode(view.mode) || !resultsAvailable) return;
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
        setParts([]);
        setSelectedPartId(storedParts[0]?.id ?? null);
        setGridMeta(null);
        setProbe(null);
        job.reset(project.status === 'completed' ? 'Results ready' : 'Ready');
        setResultsAvailable(project.status === 'completed');
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

  const ensureProject = useCallback(async (): Promise<string> => {
    if (projectId) {
      await api.updateProject(projectId, settings);
      return projectId;
    }
    const created = await api.createProject(settings);
    skipAutosave.current = true;
    setProjectId(created.id);
    void refreshProjects();
    return created.id;
  }, [projectId, settings, refreshProjects]);

  const saveProject = useCallback(async () => {
    try {
      const id = await ensureProject();
      succeed('Project saved.');
      void refreshProjects();
      return id;
    } catch (error) {
      fail(error, 'Unable to save the project');
      return null;
    }
  }, [ensureProject, fail, refreshProjects, succeed]);

  const newProject = useCallback(() => {
    setConfirmRequest({
      title: 'New project',
      message: 'Start a new project? The current one stays saved on the server and can be reopened.',
      confirmLabel: 'New project',
      onConfirm: () => {
        skipAutosave.current = true;
        setSettings(defaultSettings());
        setProjectId(null);
        setParts([]);
        setSelectedPartId(null);
        setResultsAvailable(false);
        setGridMeta(null);
        setProbe(null);
        job.reset();
        patchShell({ activePanel: 'parts' });
        notify('Started a new project.');
      }
    });
  }, [job, notify, patchShell]);

  const saveProjectAs = useCallback(() => {
    setPromptRequest({
      title: 'Save As',
      label: 'New project name',
      initialValue: `${settings.name ?? 'Untitled Casting'} (copy)`,
      confirmLabel: 'Save a copy',
      hint: 'Copies the parts and every setting. Results stay with the original.',
      onSubmit: (name) => {
        void (async () => {
          try {
            const sourceId = await ensureProject();
            const created = await api.duplicateProject(sourceId, name);
            await openProject(created.id, { silent: true });
            setSettings((current) => ({ ...current, name }));
            succeed(`Saved as "${name}".`);
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
        const target = await ensureProject();
        const uploaded = await api.uploadFiles(accepted);
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
        succeed(`Added ${added.length} part${added.length === 1 ? '' : 's'}.`);
        void refreshProjects();
      } catch (error) {
        fail(error, 'Upload failed');
      } finally {
        setUploading(false);
      }
    },
    [ensureProject, fail, notify, refreshProjects, succeed, warn]
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

  const removePart = useCallback(
    (partId: string) => {
      setParts((current) => current.filter((part) => part.id !== partId));
      setSelectedPartId((current) => (current === partId ? null : current));
      if (!projectId) return;
      api.deletePart(projectId, partId).catch((error: unknown) => fail(error, 'Unable to remove the part'));
    },
    [projectId, fail]
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

  // --- Keyboard ------------------------------------------------------------
  const closeTopLayer = useCallback(() => {
    if (promptRequest) return setPromptRequest(null);
    if (confirmRequest) return setConfirmRequest(null);
    if (modal) return setModal(null);
    if (shell.tool !== 'select') return patchShell({ tool: 'select' });
    if (shell.activePanel) return patchShell({ activePanel: null });
  }, [promptRequest, confirmRequest, modal, shell.tool, shell.activePanel, patchShell]);

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
    fitCamera: () => viewportRef.current?.setCamera('fit'),
    helpGuide: () => setModal('help'),
    escape: closeTopLayer,
    togglePlayback: () => {
      if (isTimeMode(view.mode) && resultsAvailable) patchView({ playing: !view.playing });
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
      case 'material':
        return (
          <MaterialPanel
            alloy={settings.alloy}
            materials={materials}
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
    newProject
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
        job={job.job}
        onScreenshot={() => void takeScreenshot()}
        onExport={(kind) => {
          if (kind === 'screenshot') void takeScreenshot();
          else notify('This export arrives in the next milestone.');
        }}
        onHelp={(topic) => setModal(topic === 'guide' ? 'help' : topic === 'hardware' ? 'hardware' : 'about')}
        exportsAvailable={{ animation: false, report: false, projectFile: false }}
      />

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
          view={view}
          onViewChange={patchView}
          ortho={shell.ortho}
          tool={shell.tool}
          probe={probe}
          onProbe={setProbe}
          onMeasure={(result) => notify(`Distance: ${result.distanceIn.toFixed(3)} in`)}
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
          status={viewportStatus}
          onStatus={setViewportStatus}
          onError={(message) => fail(message)}
          onAddFiles={(files) => void addFiles(files)}
          busy={loadingProject || uploading}
          onRequestAddParts={() => {
            patchShell({ activePanel: 'parts' });
            window.setTimeout(() => addPartsRef.current(), 0);
          }}
          registerAddParts={(open) => {
            addPartsRef.current = open;
          }}
        />
        <ResultsPanel
          collapsed={shell.resultsCollapsed}
          onToggleCollapsed={() => patchShell({ resultsCollapsed: !shell.resultsCollapsed })}
          view={view}
          onViewChange={patchView}
          meta={gridMeta}
          resultsAvailable={resultsAvailable}
          ortho={shell.ortho}
          onOrthoChange={(ortho) => patchShell({ ortho })}
          tool={shell.tool}
          onToolChange={(tool: ViewportTool) => patchShell({ tool })}
          onCamera={(preset: CameraPreset) => viewportRef.current?.setCamera(preset)}
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

      {modal === 'help' ? <HelpGuideModal alloys={alloyGuide} onClose={() => setModal(null)} /> : null}
      {modal === 'hardware' ? <HardwareInfoModal hardware={hardware} onClose={() => setModal(null)} /> : null}
      {modal === 'about' ? <AboutModal onClose={() => setModal(null)} /> : null}

      {confirmRequest ? (
        <ConfirmDialog request={confirmRequest} onClose={() => setConfirmRequest(null)} />
      ) : null}
      {promptRequest ? <PromptDialog request={promptRequest} onClose={() => setPromptRequest(null)} /> : null}

      <Toasts toasts={toasts.toasts} onDismiss={toasts.dismiss} />
    </div>
  );
}
