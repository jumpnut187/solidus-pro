import { useEffect, useMemo, useState } from 'react';
import {
  Aperture,
  Box,
  Camera,
  ChevronDown,
  CircleStop,
  Layers3,
  Moon,
  Pause,
  Play,
  Sun,
  Upload
} from 'lucide-react';
import type { ReactNode } from 'react';
import { BabylonViewport } from './components/BabylonViewport';
import { PartsManagerDialog } from './components/PartsManagerDialog';
import { api, type MaterialInfo, type ProjectPart, type ResultFieldsPayload } from './lib/api';
import { defaultSettings, resultModes } from './lib/defaults';

type Theme = 'dark' | 'light';

export function App() {
  const [theme, setTheme] = useState<Theme>('dark');
  const [settings, setSettings] = useState(defaultSettings);
  const [projectId, setProjectId] = useState<string | null>(null);
  const [materials, setMaterials] = useState<MaterialInfo[]>([]);
  const [parts, setParts] = useState<ProjectPart[]>([]);
  const [selectedPartId, setSelectedPartId] = useState<string | null>(null);
  const [status, setStatus] = useState('Ready');
  const [progress, setProgress] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [results, setResults] = useState<Record<string, unknown> | null>(null);
  const [resultMode, setResultMode] = useState('fraction_liquid');
  const [threshold, setThreshold] = useState(15);
  const [timePercent, setTimePercent] = useState(0);
  const [isPlayingTimeline, setIsPlayingTimeline] = useState(false);
  const [modelOpacity, setModelOpacity] = useState(34);
  const [showModel, setShowModel] = useState(true);
  const [resultFields, setResultFields] = useState<ResultFieldsPayload | null>(null);
  const [meshEstimate, setMeshEstimate] = useState<string>('No mesh estimate');
  const [cameraCommand, setCameraCommand] = useState<string | null>(null);
  const [projectionMode, setProjectionMode] = useState<'perspective' | 'orthographic'>('orthographic');
  const [showPartsDialog, setShowPartsDialog] = useState(false);

  const selectedPart = parts.find((part) => part.id === selectedPartId) ?? null;
  const material = materials.find((item) => item.name === settings.alloy);
  const resultsReady = Boolean(results && resultFields);
  const activeResultCount = useMemo(
    () => countActiveResults(resultFields, resultMode, threshold, timePercent),
    [resultFields, resultMode, threshold, timePercent]
  );

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    api.materials().then(setMaterials).catch((error) => setStatus(error.message));
  }, []);

  useEffect(() => {
    if (!isPlayingTimeline || resultMode !== 'fraction_liquid' || !resultsReady) return;
    const interval = window.setInterval(() => {
      setTimePercent((current) => (current >= 100 ? 0 : Math.min(100, current + 1)));
    }, 90);
    return () => window.clearInterval(interval);
  }, [isPlayingTimeline, resultMode, resultsReady]);

  useEffect(() => {
    if (resultMode !== 'fraction_liquid') {
      setIsPlayingTimeline(false);
    }
  }, [resultMode]);

  async function ensureProject() {
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
        name: file.filename.replace(/\.stl$/i, ''),
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
    if (!selectedPart || !projectId) return;
    const next = { ...selectedPart, ...patch };
    setParts((current) => current.map((part) => (part.id === next.id ? next : part)));
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

  function arrangeGrid() {
    if (!parts.length || !projectId) return;
    const count = parts.length;
    const cols = Math.ceil(Math.sqrt(count));
    const spacing = Math.max(
      5,
      parts.reduce((max, p) => Math.max(max, p.meshData?.bounds?.x[1] ?? 0, p.meshData?.bounds?.y[1] ?? 0), 0) * 2
    );
    const updates: Promise<unknown>[] = [];
    parts.forEach((part, index) => {
      const row = Math.floor(index / cols);
      const col = index % cols;
      const x = (col - (cols - 1) / 2) * spacing;
      const y = (row - (Math.ceil(count / cols) - 1) / 2) * spacing;
      const patch = { ...part, position_x: x, position_y: y };
      setParts((current) => current.map((p) => (p.id === part.id ? patch : p)));
      updates.push(api.updatePart(projectId, patch));
    });
    Promise.all(updates).catch(() => {});
  }

  function arrangeLine() {
    if (!parts.length || !projectId) return;
    const spacing = Math.max(
      5,
      parts.reduce((max, p) => Math.max(max, p.meshData?.bounds?.x[1] ?? 0, p.meshData?.bounds?.y[1] ?? 0), 0) * 2.5
    );
    const updates: Promise<unknown>[] = [];
    parts.forEach((part, index) => {
      const x = (index - (parts.length - 1) / 2) * spacing;
      const patch = { ...part, position_x: x, position_y: 0 };
      setParts((current) => current.map((p) => (p.id === part.id ? patch : p)));
      updates.push(api.updatePart(projectId, patch));
    });
    Promise.all(updates).catch(() => {});
  }

  async function updateMeshPreview() {
    const nextProjectId = await ensureProject();
    try {
      const preview = await api.meshPreview(nextProjectId, settings.voxel_size);
      setMeshEstimate(`${preview.estimated_voxels.toLocaleString()} voxels, ${preview.estimated_memory_mb.toFixed(0)} MB`);
      if (!preview.can_proceed) setStatus(preview.message);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Mesh preview failed');
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
        const fields = await api.resultFields(id);
        setResultFields(fields);
        setResultMode('fraction_liquid');
        setTimePercent(0);
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

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <header className="brand">
          <div className="brand-mark"><Aperture size={24} /></div>
          <div>
            <strong>Solidus Flow</strong>
            <span>React Babylon fork</span>
          </div>
          <button className="icon-button" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} title="Theme">
            {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
          </button>
        </header>

        <Panel title="Parts Assembly" icon={<Layers3 size={17} />}>
          <label
            className="drop-zone"
            onDragOver={(event) => event.preventDefault()}
            onDrop={(event) => {
              event.preventDefault();
              void handleFiles(event.dataTransfer.files);
            }}
          >
            <Upload size={22} />
            <span>Upload or drop CAD files (STL or STEP)</span>
            <input type="file" multiple accept=".stl,.step,.stp" onChange={(event) => event.target.files && void handleFiles(event.target.files)} />
          </label>
          <div className="part-counts">
            <span>{parts.filter((p) => p.part_type === 'cast').length} Cast</span>
            <span>{parts.filter((p) => p.part_type === 'riser').length} Riser</span>
            <span>{parts.filter((p) => p.part_type === 'riser_post').length} Post</span>
            <span>{parts.filter((p) => p.part_type === 'chill').length} Chill</span>
          </div>
          <button
            className="run-button"
            disabled={!parts.length}
            onClick={() => setShowPartsDialog(true)}
            style={{ marginTop: 8 }}
          >
            <Box size={16} /> Open Parts Manager...
          </button>
        </Panel>

        <Panel title="Material">
          <label>Alloy
            <select value={settings.alloy} onChange={(e) => setSettings({ ...settings, alloy: e.target.value })}>
              {materials.map((item) => <option key={item.name} value={item.name}>{item.name} - {item.full_name}</option>)}
            </select>
          </label>
          <div className="stats-card">
            <span>Liquidus <strong>{material?.liquidus_temp ?? '--'} C</strong></span>
            <span>Solidus <strong>{material?.solidus_temp ?? '--'} C</strong></span>
            <span>Shrinkage <strong>{material?.volumetric_shrinkage ?? '--'}%</strong></span>
            <span>Niyama <strong>{material?.niyama_threshold ?? '--'}</strong></span>
          </div>
        </Panel>

        <Panel title="Analysis Options">
          <Check label="Shrinkage / Porosity" checked={Boolean(settings.analysis_options.shrinkage)} onChange={(v) => setAnalysis('shrinkage', v)} />
          <Check label="Hot Spots" checked={Boolean(settings.analysis_options.hotspot)} onChange={(v) => setAnalysis('hotspot', v)} />
          <Check label="Niyama" checked={Boolean(settings.analysis_options.niyama)} onChange={(v) => setAnalysis('niyama', v)} />
          <Check label="Fraction Liquid" checked={Boolean(settings.analysis_options.fraction_liquid)} onChange={(v) => setAnalysis('fraction_liquid', v)} />
          <Check label="Hot Tear Risk" checked={Boolean(settings.analysis_options.hot_tear)} onChange={(v) => setAnalysis('hot_tear', v)} />
          <Check label="Sand Burn-In" checked={Boolean(settings.analysis_options.sand_burn_in)} onChange={(v) => setAnalysis('sand_burn_in', v)} />
          <Check label="Feeding Path" checked={Boolean(settings.analysis_options.feeding_path)} onChange={(v) => setAnalysis('feeding_path', v)} />
          <Check label="Stress / Distortion" checked={Boolean(settings.analysis_options.stress_analysis)} onChange={(v) => setAnalysis('stress_analysis', v)} />
          <Check label="Microstructure / SDAS" checked={Boolean(settings.analysis_options.microstructure)} onChange={(v) => setAnalysis('microstructure', v)} />
          <label>Sand restraint {(Number(settings.analysis_options.sand_restraint_index) || 0).toFixed(2)}
            <input type="range" min="0" max="1" step="0.05" value={Number(settings.analysis_options.sand_restraint_index)}
              onChange={(e) => setAnalysis('sand_restraint_index', Number(e.target.value))} />
          </label>
        </Panel>

        <Panel title="Shrinkage Parameters">
          {(() => {
            const feedingParams = settings.solidification_params.feeding ?? {};
            const rawShrinkScalar = Number(feedingParams.shrink_scalar ?? 1);
            const shrinkScalar = Math.min(4, Math.max(0.25, Number.isFinite(rawShrinkScalar) ? rawShrinkScalar : 1));
            return (
              <label>Shrink Scalar {shrinkScalar.toFixed(2)}
                <input type="range" min="0.25" max="4" step="0.05" value={shrinkScalar}
                  onChange={(e) => setSolidification('feeding', 'shrink_scalar', Number(e.target.value))} />
              </label>
            );
          })()}
        </Panel>

        <Panel title="Hot Tear Parameters">
          {(() => {
            const hotTearParams = settings.solidification_params.hot_tear ?? {};
            return <>
          <label>Mold Restraint {(Number(hotTearParams.mold_restraint_index) || 0).toFixed(2)}
            <input type="range" min="0" max="1" step="0.05" value={Number(hotTearParams.mold_restraint_index)}
              onChange={(e) => setSolidification('hot_tear', 'mold_restraint_index', Number(e.target.value))} />
          </label>
          <label>Metal/Mold Friction {(Number(hotTearParams.metal_mold_friction_index) || 0).toFixed(2)}
            <input type="range" min="0" max="1" step="0.05" value={Number(hotTearParams.metal_mold_friction_index)}
              onChange={(e) => setSolidification('hot_tear', 'metal_mold_friction_index', Number(e.target.value))} />
          </label>
          <label>Core Restraint {(Number(hotTearParams.core_restraint_index) || 0).toFixed(2)}
            <input type="range" min="0" max="1" step="0.05" value={Number(hotTearParams.core_restraint_index)}
              onChange={(e) => setSolidification('hot_tear', 'core_restraint_index', Number(e.target.value))} />
          </label>
          <label>Impurity Index {(Number(hotTearParams.low_melting_impurity_index) || 0).toFixed(2)}
            <input type="range" min="0" max="1" step="0.05" value={Number(hotTearParams.low_melting_impurity_index)}
              onChange={(e) => setSolidification('hot_tear', 'low_melting_impurity_index', Number(e.target.value))} />
          </label>
          <label>Threshold {Number(hotTearParams.hot_tear_zone_threshold).toFixed(2)} (0=adaptive)
            <input type="range" min="0" max="1" step="0.05" value={Number(hotTearParams.hot_tear_zone_threshold)}
              onChange={(e) => setSolidification('hot_tear', 'hot_tear_zone_threshold', Number(e.target.value))} />
          </label>
          <label>Healing Relief {(Number(hotTearParams.healing_relief_strength) || 0).toFixed(2)}
            <input type="range" min="0" max="1" step="0.05" value={Number(hotTearParams.healing_relief_strength)}
              onChange={(e) => setSolidification('hot_tear', 'healing_relief_strength', Number(e.target.value))} />
          </label>
          <label>Surface Depth {Number(hotTearParams.surface_restraint_depth_inches).toFixed(2)} in
            <input type="range" min="0.1" max="3" step="0.1" value={Number(hotTearParams.surface_restraint_depth_inches)}
              onChange={(e) => setSolidification('hot_tear', 'surface_restraint_depth_inches', Number(e.target.value))} />
          </label>
          <label>Path Search Depth {Number(hotTearParams.solid_path_search_depth)}
            <input type="range" min="2" max="20" step="1" value={Number(hotTearParams.solid_path_search_depth)}
              onChange={(e) => setSolidification('hot_tear', 'solid_path_search_depth', Number(e.target.value))} />
          </label>
            </>;
          })()}
        </Panel>

        <Panel title="Mesh and Performance">
          <label>Voxel size (in)<input type="number" min="0.01" step="0.01" value={settings.voxel_size} onChange={(e) => setSettings({ ...settings, voxel_size: Number(e.target.value) })} /></label>
          <label>Refinement
            <select value={settings.refinement_level} onChange={(e) => setSettings({ ...settings, refinement_level: Number(e.target.value) })}>
              <option value={1}>1 - Coarse</option>
              <option value={2}>2 - Medium</option>
              <option value={3}>3 - Fine</option>
              <option value={4}>4 - Very fine</option>
            </select>
          </label>
          <label>Memory mode
            <select value={settings.memory_mode} onChange={(e) => setSettings({ ...settings, memory_mode: e.target.value })}>
              <option value="auto">Auto</option>
              <option value="ram">RAM</option>
              <option value="disk">Disk</option>
            </select>
          </label>
          <Check label="Chunked voxelization" checked={Boolean(settings.performance_options.use_chunked_voxelization)} onChange={(v) => setPerformance('use_chunked_voxelization', v)} />
          <Check label="Incremental solidification" checked={Boolean(settings.performance_options.use_incremental)} onChange={(v) => setPerformance('use_incremental', v)} />
          <button className="ghost-button" onClick={() => void updateMeshPreview()}>Estimate Mesh</button>
          <p className="muted">{meshEstimate}</p>
        </Panel>

        <Panel title="Simulation">
          <button className="run-button" disabled={isRunning || !parts.length} onClick={() => void runSimulation()}><Play size={16} /> Run Simulation</button>
          <button className="ghost-button" disabled={!isRunning} onClick={() => void cancelSimulation()}><CircleStop size={16} /> Cancel</button>
          <div className="progress"><span style={{ width: `${progress}%` }} /></div>
          <p className="status">{status}</p>
        </Panel>
      </aside>

      <main className="stage">
        <div className="stage-toolbar">
          <div>
            <strong>3D Solidification Canvas</strong>
            <span>{parts.length ? `${parts.length} assembly part(s)` : 'Upload CAD files to begin'}</span>
          </div>
          <div className="toolbar-actions">
            <button onClick={() => setCameraCommand('iso')}><Camera size={15} /> Iso</button>
            <button onClick={() => setCameraCommand('front')}>Front</button>
            <button onClick={() => setCameraCommand('top')}>Top</button>
            <button onClick={() => setCameraCommand('right')}>Right</button>
            <span className="toolbar-separator" />
            <button
              className={projectionMode === 'orthographic' ? 'active' : ''}
              onClick={() => setProjectionMode(projectionMode === 'orthographic' ? 'perspective' : 'orthographic')}
              title="Toggle projection mode"
            >
              {projectionMode === 'orthographic' ? 'Orthographic' : 'Perspective'}
            </button>
          </div>
        </div>
        <BabylonViewport
          parts={parts}
          selectedPartId={selectedPartId}
          resultFields={resultFields}
          resultMode={resultMode}
          threshold={threshold}
          timePercent={timePercent}
          modelOpacity={modelOpacity / 100}
          showModel={showModel}
          theme={theme}
          cameraCommand={cameraCommand}
          onCameraHandled={() => setCameraCommand(null)}
          projectionMode={projectionMode}
          onPartSelected={(id) => setSelectedPartId(id)}
        />
        <section className="result-console">
          <label>Result
            <select
              value={resultMode}
              disabled={!resultsReady}
              onChange={(e) => {
                setIsPlayingTimeline(false);
                setResultMode(e.target.value);
              }}
            >
              {resultModes.map((mode) => (
                <option key={mode.id} value={mode.id} disabled={resultFields ? !resultFields.modes.includes(mode.id) : false}>
                  {mode.label}
                </option>
              ))}
            </select>
          </label>
          {resultMode !== 'fraction_liquid' ? (
            <label>{resultMode === 'shrinkage' ? 'X-Ray Level' : resultMode === 'feed_modulus' ? 'Feed Modulus Range' : 'Threshold'} {threshold}%
              <input type="range" min="1" max="100" value={threshold} disabled={!resultsReady} onChange={(e) => setThreshold(Number(e.target.value))} />
            </label>
          ) : (
            <div className="timeline-control">
              <button className="mini-button" disabled={!resultsReady} onClick={() => setIsPlayingTimeline((playing) => !playing)}>
                {isPlayingTimeline ? <Pause size={14} /> : <Play size={14} />}
                {isPlayingTimeline ? 'Pause' : 'Play'}
              </button>
              <button
                className="mini-button"
                disabled={!resultsReady}
                onClick={() => {
                  setIsPlayingTimeline(false);
                  setTimePercent(0);
                }}
              >
                Liquidus
              </button>
              <label>Liquidus to solidus {timePercent}%
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={timePercent}
                  disabled={!resultsReady}
                  onChange={(e) => {
                    setIsPlayingTimeline(false);
                    setTimePercent(Number(e.target.value));
                  }}
                />
              </label>
            </div>
          )}
          <label>Model opacity {modelOpacity}%
            <input type="range" min="8" max="80" value={modelOpacity} onChange={(e) => setModelOpacity(Number(e.target.value))} />
          </label>
          <label className="check"><input type="checkbox" checked={showModel} onChange={(e) => setShowModel(e.target.checked)} /> Show translucent model</label>
          <div className="result-stats">
            <span>Voxels <strong>{typeof results?.voxel_count === 'number' ? results.voxel_count.toLocaleString() : '--'}</strong></span>
            <span>Volume <strong>{typeof results?.total_volume === 'number' ? results.total_volume.toFixed(2) : '--'} in3</strong></span>
            <span>Active <strong>{activeResultCount?.toLocaleString() ?? '--'}</strong></span>
          </div>
        </section>
      </main>

      {showPartsDialog && (
        <PartsManagerDialog
          parts={parts}
          selectedPartId={selectedPartId}
          onSelectPart={(id) => setSelectedPartId(id)}
          onUpdatePart={(patch) => void updateSelectedPart(patch)}
          onAddParts={(files) => void handleFiles(files)}
          onRemovePart={(id) => void removePart(id)}
          onClearAll={() => void clearAllParts()}
          onDuplicatePart={(id) => void duplicatePart(id)}
          onArrangeGrid={arrangeGrid}
          onArrangeLine={arrangeLine}
          onClose={() => setShowPartsDialog(false)}
        />
      )}
    </div>
  );

  function setAnalysis(key: string, value: boolean | number) {
    setSettings({ ...settings, analysis_options: { ...settings.analysis_options, [key]: value } });
  }

  function setPerformance(key: string, value: boolean | number | string) {
    setSettings({ ...settings, performance_options: { ...settings.performance_options, [key]: value } });
  }

  function setSolidification(group: string, key: string, value: number) {
    setSettings({
      ...settings,
      solidification_params: {
        ...settings.solidification_params,
        [group]: {
          ...(settings.solidification_params[group] ?? {}),
          [key]: value
        }
      }
    });
  }
}

function Panel({ title, icon, children }: { title: string; icon?: ReactNode; children: ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);
  return (
    <section className="panel">
      <div className="panel-title" onClick={() => setCollapsed((c) => !c)} style={{ cursor: 'pointer' }}>
        {icon}<span>{title}</span><ChevronDown size={15} style={{ transform: collapsed ? 'rotate(-90deg)' : 'rotate(0deg)', transition: 'transform 0.2s ease' }} />
      </div>
      {!collapsed && <div className="panel-body">{children}</div>}
    </section>
  );
}

function Check({ label, checked, onChange }: { label: string; checked: boolean; onChange: (checked: boolean) => void }) {
  return <label className="check"><input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} /> {label}</label>;
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
