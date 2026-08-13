import { useEffect, useMemo, useState } from 'react';

import { BabylonViewport } from './components/BabylonViewport';
import { ResultLegend } from './components/ResultLegend';
import { Sidebar, type DrawerId } from './components/Sidebar';
import { PartsDrawer } from './components/drawers/PartsDrawer';
import { MaterialDrawer } from './components/drawers/MaterialDrawer';
import { MeshDrawer } from './components/drawers/MeshDrawer';
import { AnalysisDrawer } from './components/drawers/AnalysisDrawer';
import { PerformanceDrawer } from './components/drawers/PerformanceDrawer';
import { SolidificationDrawer } from './components/drawers/SolidificationDrawer';
import { api, type MaterialInfo, type ProjectPart, type ResultFieldsPayload } from './lib/api';
import { loadStoredSettings, saveStoredSettings } from './lib/defaults';
import { loadUiPrefs, saveUiPrefs, type UiPrefs } from './lib/uiPrefs';

const MAX_RESULT_POINTS = 120000;

export function App() {
  const [settings, setSettings] = useState(loadStoredSettings);
  const [ui, setUi] = useState<UiPrefs>(loadUiPrefs);
  const [projectId, setProjectId] = useState<string | null>(null);
  const [materials, setMaterials] = useState<MaterialInfo[]>([]);
  const [parts, setParts] = useState<ProjectPart[]>([]);
  const [selectedPartId, setSelectedPartId] = useState<string | null>(null);
  const [status, setStatus] = useState('Ready');
  const [progress, setProgress] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [isPlayingTimeline, setIsPlayingTimeline] = useState(false);
  const [results, setResults] = useState<Record<string, unknown> | null>(null);
  const [resultFields, setResultFields] = useState<ResultFieldsPayload | null>(null);
  const [meshEstimate, setMeshEstimate] = useState('No mesh estimate');
  const [mlStatus, setMlStatus] = useState('No ML recommendation yet');
  const [cameraCommand, setCameraCommand] = useState<string | null>(null);
  const [activeDrawer, setActiveDrawer] = useState<DrawerId | null>(null);

  const material = materials.find((item) => item.name === settings.alloy);
  const resultsReady = Boolean(results && resultFields);
  const statusWarning = /axis mismatch|⚠/i.test(status);
  const activeResultCount = useMemo(
    () => countActiveResults(resultFields, ui.resultMode, ui.threshold, ui.timePercent),
    [resultFields, ui.resultMode, ui.threshold, ui.timePercent]
  );

  function patchUi(patch: Partial<UiPrefs>) {
    setUi((prev) => ({ ...prev, ...patch }));
  }

  useEffect(() => {
    saveStoredSettings(settings);
  }, [settings]);

  useEffect(() => {
    saveUiPrefs(ui);
  }, [ui]);

  useEffect(() => {
    api.materials().then(setMaterials).catch((error) => setStatus(error.message));
  }, []);

  useEffect(() => {
    if (!isPlayingTimeline || ui.resultMode !== 'fraction_liquid' || !resultsReady) return;
    const interval = window.setInterval(() => {
      setUi((prev) => {
        const next = prev.timePercent + prev.flSpeed;
        return { ...prev, timePercent: next >= 100 ? 0 : Math.min(100, next) };
      });
    }, 90);
    return () => window.clearInterval(interval);
  }, [isPlayingTimeline, ui.resultMode, resultsReady]);

  useEffect(() => {
    if (ui.resultMode !== 'fraction_liquid') {
      setIsPlayingTimeline(false);
    }
  }, [ui.resultMode]);

  async function ensureProject(): Promise<string> {
    if (projectId) {
      await api.updateProject(projectId, settings);
      return projectId;
    }
    const created = await api.createProject(settings);
    setProjectId(created.id);
    return created.id;
  }

  async function handleFiles(files: FileList | File[]) {
    const fileArray = Array.from(files).filter((file) => /\.(stl|step|stp)$/i.test(file.name));
    if (!fileArray.length) return;
    setStatus(`Uploading ${fileArray.length} CAD file(s)...`);
    const nextProjectId = await ensureProject();
    const uploaded = await api.uploadFiles(fileArray);
    const added: ProjectPart[] = [];
    for (const file of uploaded.files) {
      const response = await api.addPart(nextProjectId, {
        name: file.filename.replace(/\.(stl|step|stp)$/i, ''),
        part_type: file.part_type,
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
    setSelectedPartId(added.at(-1)?.id ?? selectedPartId);
    setStatus(`Loaded ${added.length} part(s)`);
  }

  async function updateSelectedPart(patch: Partial<ProjectPart>) {
    if (!projectId) return;
    const targetId = patch.id ?? selectedPartId;
    if (!targetId) return;
    const current = parts.find((p) => p.id === targetId);
    if (!current) return;
    const next = { ...current, ...patch };
    if (patch.part_type && patch.part_type !== 'riser' && next.is_open_top) {
      next.is_open_top = false;
    }
    setParts((prev) => prev.map((part) => (part.id === next.id ? next : part)));
    if (patch.id && selectedPartId !== patch.id) setSelectedPartId(patch.id);
    try {
      await api.updatePart(projectId, next);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Unable to update part');
    }
  }

  async function removePart(partId: string) {
    if (!projectId) return;
    setParts((current) => current.filter((p) => p.id !== partId));
    if (selectedPartId === partId) setSelectedPartId(null);
    try {
      await api.deletePart(projectId, partId);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Unable to remove part');
    }
  }

  async function clearAllParts() {
    if (!projectId) return;
    const ids = parts.map((p) => p.id);
    setParts([]);
    setSelectedPartId(null);
    for (const id of ids) {
      try {
        await api.deletePart(projectId, id);
      } catch {
        // ignore individual failures
      }
    }
  }

  async function duplicatePart(partId: string) {
    if (!projectId) return;
    const part = parts.find((p) => p.id === partId);
    if (!part) return;
    try {
      const response = await api.addPart(projectId, {
        name: `${part.name} (copy)`,
        part_type: part.part_type,
        filepath: part.filepath,
        position_x: part.position_x,
        position_y: part.position_y,
        position_z: part.position_z,
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
      setStatus(error instanceof Error ? error.message : 'Unable to duplicate part');
    }
  }

  async function updateMeshPreview() {
    const nextProjectId = await ensureProject();
    try {
      const preview = await api.meshPreview(nextProjectId, settings.voxel_size);
      let estimate = `${preview.estimated_voxels.toLocaleString()} voxels, ${preview.estimated_memory_mb.toFixed(0)} MB`;
      if (!preview.can_proceed && preview.recommended_voxel_size) {
        estimate += ` | Safe size: ${preview.recommended_voxel_size.toFixed(3)}"`;
      }
      setMeshEstimate(estimate);
      if (!preview.can_proceed) setStatus(preview.message);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Mesh preview failed');
    }
  }

  async function applyMlMesh() {
    const nextProjectId = await ensureProject();
    setMlStatus('Requesting ML mesh recommendation...');
    try {
      const rec = await api.mlMesh(nextProjectId);
      setSettings((current) => ({
        ...current,
        voxel_size: rec.base_voxel_size,
        refinement_level: rec.refinement_level
      }));
      const source = rec.used_model ? 'ML model' : 'heuristic';
      setMlStatus(
        `${source}: ${rec.base_voxel_size.toFixed(3)}" @ refinement ${rec.refinement_level} ` +
          `(confidence ${rec.confidence.toFixed(2)}) — ${rec.reason}` +
          (rec.warning ? ` — Warning: ${rec.warning}` : '')
      );
      setStatus('Applied ML mesh recommendation.');
    } catch (error) {
      setMlStatus(error instanceof Error ? error.message : 'ML mesh failed');
    }
  }

  async function runSimulation() {
    if (!parts.length) {
      setStatus('Upload at least one CAD part first.');
      return;
    }
    const nextProjectId = await ensureProject();
    setResults(null);
    setResultFields(null);
    setIsRunning(true);
    setProgress(0);
    setStatus('Submitting simulation...');
    try {
      await api.runProject(nextProjectId);
      pollStatus(nextProjectId);
    } catch (error) {
      setIsRunning(false);
      setStatus(error instanceof Error ? error.message : 'Simulation failed to start');
    }
  }

  async function pollStatus(id: string) {
    try {
      const job = await api.status(id);
      setProgress(job.progress ?? 0);
      setStatus(job.message || job.status);
      if (job.status === 'completed') {
        const payload = await api.results(id);
        setResults(payload.data);
        setStatus('Loading result cache...');
        const fields = await api.resultFields(id, { maxPoints: MAX_RESULT_POINTS });
        setResultFields(fields);
        patchUi({ resultMode: 'fraction_liquid', timePercent: 0 });
        setIsPlayingTimeline(false);
        setProgress(100);
        setIsRunning(false);
        setStatus('Simulation complete');
      } else if (job.status === 'error' || job.status === 'cancelled') {
        setIsRunning(false);
      } else {
        window.setTimeout(() => pollStatus(id), job.status === 'queued' ? 1500 : 650);
      }
    } catch (error) {
      setIsRunning(false);
      setStatus(error instanceof Error ? error.message : 'Status polling failed');
    }
  }

  async function cancelSimulation() {
    if (!projectId) return;
    await api.cancelProject(projectId);
    setIsRunning(false);
    setStatus('Simulation cancelled');
  }

  function setAnalysis(key: string, value: boolean | number) {
    setSettings((prev) => {
      const current = prev.analysis_options[key as keyof typeof prev.analysis_options];
      return {
        ...prev,
        analysis_options: {
          ...prev.analysis_options,
          [key]: typeof current === 'number' ? Number(value) : Boolean(value),
        },
      };
    });
  }

  return (
    <div className="app-shell">
      <Sidebar
        parts={parts}
        settings={settings}
        resultsReady={resultsReady}
        resultFields={resultFields}
        results={results}
        ui={ui}
        patchUi={patchUi}
        isRunning={isRunning}
        progress={progress}
        status={status}
        statusWarning={statusWarning}
        activeResultCount={activeResultCount}
        isPlayingTimeline={isPlayingTimeline}
        setIsPlayingTimeline={setIsPlayingTimeline}
        onRun={() => void runSimulation()}
        onStop={() => void cancelSimulation()}
        onOpenDrawer={setActiveDrawer}
        setCameraCommand={setCameraCommand}
      />
      <main className="stage">
        <BabylonViewport
          parts={parts}
          selectedPartId={selectedPartId}
          resultFields={resultFields}
          resultMode={ui.resultMode}
          threshold={ui.threshold}
          timePercent={ui.timePercent}
          modelOpacity={ui.modelOpacity / 100}
          showModel={ui.showModel}
          showResults={ui.showResults}
          cameraCommand={cameraCommand}
          onCameraHandled={() => setCameraCommand(null)}
          ortho={ui.ortho}
          section={ui.section}
          onPartSelected={(id) => setSelectedPartId(id)}
        />
        <ResultLegend
          resultMode={ui.resultMode}
          resultFields={resultFields}
          visible={resultsReady && ui.showResults}
        />
      </main>

      {activeDrawer === 'parts' && (
        <PartsDrawer
          parts={parts}
          selectedPartId={selectedPartId}
          onSelectPart={(id) => setSelectedPartId(id)}
          onUpdatePart={(patch) => void updateSelectedPart(patch)}
          onAddParts={(files) => void handleFiles(files)}
          onRemovePart={(id) => void removePart(id)}
          onClearAll={() => void clearAllParts()}
          onDuplicatePart={(id) => void duplicatePart(id)}
          onClose={() => setActiveDrawer(null)}
        />
      )}
      {activeDrawer === 'material' && (
        <MaterialDrawer
          settings={settings}
          setSettings={setSettings}
          materials={materials}
          material={material}
          onClose={() => setActiveDrawer(null)}
        />
      )}
      {activeDrawer === 'mesh' && (
        <MeshDrawer
          settings={settings}
          setSettings={setSettings}
          projectId={projectId}
          meshEstimate={meshEstimate}
          mlStatus={mlStatus}
          updateMeshPreview={updateMeshPreview}
          applyMlMesh={applyMlMesh}
          onClose={() => setActiveDrawer(null)}
        />
      )}
      {activeDrawer === 'analysis' && (
        <AnalysisDrawer
          settings={settings}
          setAnalysis={setAnalysis}
          onClose={() => setActiveDrawer(null)}
        />
      )}
      {activeDrawer === 'performance' && (
        <PerformanceDrawer
          settings={settings}
          setSettings={setSettings}
          onClose={() => setActiveDrawer(null)}
        />
      )}
      {activeDrawer === 'solidification' && (
        <SolidificationDrawer
          settings={settings}
          setSettings={setSettings}
          onClose={() => setActiveDrawer(null)}
        />
      )}
    </div>
  );
}

function countActiveResults(
  resultFields: ResultFieldsPayload | null,
  resultMode: string,
  threshold: number,
  timePercent: number
): number | null {
  if (!resultFields || resultFields.empty) return null;
  if (resultMode === 'fraction_liquid') {
    const values = fractionLiquidValues(resultFields, timePercent);
    if (!values.length) return null;
    return values.reduce((count, value) => count + (value >= 0.01 ? 1 : 0), 0);
  }
  const field = resultFields.fields[resultMode];
  if (!field) return null;
  const cutoff = thresholdForResultMode(resultFields, resultMode, threshold);
  return field.values.reduce((count, value) => count + (Number.isFinite(value) && value >= cutoff ? 1 : 0), 0);
}

function thresholdForResultMode(resultFields: ResultFieldsPayload, resultMode: string, threshold: number): number {
  if (resultMode === 'niyama_raw') return 0.01;
  const fraction = Math.min(1, Math.max(0, threshold / 100));
  if (resultMode === 'feed_modulus') {
    const field = resultFields.fields.feed_modulus;
    if (!field) return 0;
    return field.min + ((field.max - field.min) * fraction);
  }
  return fraction;
}

function fractionLiquidValues(resultFields: ResultFieldsPayload, timePercent: number): number[] {
  const frames = resultFields.fraction_liquid?.frames ?? [];
  const framePercents = resultFields.fraction_liquid?.frame_percents ?? [];
  if (frames.length) return interpolateFrames(frames, framePercents, timePercent);
  const progress = timePercent / 100;
  const frontWidth = resultFields.solidification.front_width || 0.075;
  return resultFields.solidification.normalized_times.map((time) => Math.max(0, Math.min(1, (time - progress) / frontWidth)));
}

function interpolateFrames(frames: number[][], framePercents: number[], timePercent: number): number[] {
  if (!frames.length) return [];
  if (frames.length === 1 || !framePercents.length) return frames[0];
  const target = Math.min(100, Math.max(0, timePercent));
  if (target <= framePercents[0]) return frames[0];
  const lastIndex = frames.length - 1;
  if (target >= framePercents[lastIndex]) return frames[lastIndex];
  let after = 1;
  while (after < framePercents.length && framePercents[after] < target) after += 1;
  const before = Math.max(0, after - 1);
  const beforePercent = framePercents[before] ?? 0;
  const afterPercent = framePercents[after] ?? 100;
  const alpha = (target - beforePercent) / Math.max(afterPercent - beforePercent, 1e-6);
  const beforeFrame = frames[before] ?? [];
  const afterFrame = frames[after] ?? beforeFrame;
  return beforeFrame.map((value, index) => value * (1 - alpha) + (afterFrame[index] ?? value) * alpha);
}
