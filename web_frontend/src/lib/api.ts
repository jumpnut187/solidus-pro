import type { CutFeature, ExtrudeFeature, ReferencePlane } from './modeling/types';

export type PartType =
  | 'cast'
  | 'riser'
  | 'riser_post'
  | 'chill'
  | 'core'
  | 'pour_cup'
  | 'sprue'
  | 'runner'
  | 'filter'
  | 'gate'
  | 'mold';

export type SandInfo = {
  code: number;
  label: string;
  uses: string;
  notes: string;
  bulk_density: number | null;
};

export type PreviewMesh = {
  vertices: number[];
  faces: number[];
  vertex_count: number;
  face_count: number;
};

export type MoldPreview = {
  mold: PreviewMesh[];
  cores: PreviewMesh[];
  summary: {
    pitch_in: number;
    mold_cells: number;
    graded_mold_cells?: number;
    max_cell_span?: number;
    auto_core_cells: number;
    shaft_cells: number;
    gap_cells: number;
    wall_in: number;
    box_lo_in: [number, number, number];
    box_hi_in: [number, number, number];
  } | null;
};

export type MeshData = {
  vertices: number[];
  faces: number[];
  bounds: { x: [number, number]; y: [number, number]; z: [number, number] };
  volume: number;
  surface_area: number;
  vertex_count: number;
  face_count: number;
};

export type UploadedFile = {
  file_id: string;
  filename: string;
  display_name?: string;
  filepath: string;
  source_file?: string;
  body_name?: string;
  part_type: PartType;
  dimensions: string;
  mesh_data: MeshData;
};

export type PourLocationPayload = {
  schema_version: number;
  point_in: [number, number, number];
  normal: [number, number, number] | null;
  part_id: string | null;
};

export type RiserSleeve = 'insulating' | 'adiabatic' | 'none';

export type ProjectPart = {
  id: string;
  name: string;
  part_type: PartType;
  filepath: string;
  position_x: number;
  position_y: number;
  position_z: number;
  rotation_x: number;
  rotation_y: number;
  rotation_z: number;
  scale: number;
  visible: boolean;
  is_open_top: boolean;
  riser_sleeve: RiserSleeve;
  meshData?: MeshData;
  feature?: ExtrudeFeature | null;
  base_filepath?: string | null;
};

export type MaterialInfo = {
  name: string;
  full_name: string;
  liquidus_temp: number;
  solidus_temp: number;
  freezing_range: number;
  pouring_temp: number;
  volumetric_shrinkage: number;
  linear_shrinkage: number;
  feeding_multiplier: number;
  niyama_threshold: number;
  hot_tear_susceptibility: number;
  sand_burn_in_susceptibility: number;
  gas_sensitivity: number;
  thermal_conductivity: number;
  specific_heat: number;
  latent_heat: number;
  thermal_diffusivity: number;
  density_liquid: number;
  density_solid: number;
  critical_fraction_solid: number;
  dendrite_arm_spacing: number;
  surface_tension: number;
};

export type AnalysisOptions = Record<string, boolean | number | string>;
export type PerformanceOptions = Record<string, boolean | number | string>;
export type SolidificationParams = Record<string, Record<string, number | Record<string, number>>>;

export type ProjectSettings = {
  name?: string;
  alloy: string;
  voxel_size: number;
  refinement_level: number;
  memory_mode: string;
  analysis_options: AnalysisOptions;
  performance_options: PerformanceOptions;
  solidification_params: SolidificationParams;
  ui_state?: Record<string, unknown>;
  pour_location?: PourLocationPayload | null;
};

export type PartingLine = {
  z: number;
  visible: boolean;
  locked_to_casting: boolean;
};

export type PartingLineInfo = {
  parting_line: PartingLine | null;
  part_id: string | null;
  suggestions: { max_section: number; bottom: number; top: number; center: number };
  split_z: number;
  split: { cope_depth: number; drag_depth: number; cope_fraction: number; drag_fraction: number };
  bounds: [[number, number, number], [number, number, number]];
};

export function parsePartingLine(raw: unknown): PartingLine | null {
  if (typeof raw !== 'object' || raw === null) return null;
  const record = raw as Record<string, unknown>;
  const z = Number(record.z);
  if (record.z === null || record.z === undefined || !Number.isFinite(z)) return null;
  return {
    z,
    visible: record.visible !== false,
    locked_to_casting: record.locked_to_casting !== false
  };
}

export type ProjectSummary = {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
  updated_at: string;
  alloy: string;
  voxel_size: number;
  refinement_level: number;
  memory_mode: string;
  status: string;
  summary: Record<string, unknown>;
};

export type ProjectRecord = ProjectSettings & {
  id: string;
  status: string;
  created_at: string;
  updated_at: string;
  description: string | null;
  summary: Record<string, unknown>;
};

export type FeatureBodyRequest = {
  name: string;
  part_type: PartType;
  is_open_top: boolean;
  visible: boolean;
  feature: ExtrudeFeature;
  mesh: { vertices: number[]; faces: number[] };
  cut_mesh?: { vertices: number[]; faces: number[] } | null;
};

type StoredPart = Omit<ProjectPart, 'meshData'> & { mesh_data?: MeshData };

export type JobStatus = {
  status: string;
  progress: number;
  message?: string;
  queue_position?: number;
};

export type MeshPreview = {
  estimated_voxels: number;
  estimated_memory_mb: number;
  can_proceed: boolean;
  message: string;
  recommended_voxel_size: number;
};

export type MlMeshRecommendation = {
  base_voxel_size: number;
  refinement_level: number;
  confidence: number;
  reason: string;
  used_model: boolean;
  warning?: string;
};

export type HardwareInfo = {
  cpu_brand: string;
  cpu_vendor: string;
  physical_cores: number | null;
  logical_cores: number | null;
  performance_cores: number | null;
  efficiency_cores: number | null;
  core_tiers: { name: string; cores: number }[];
  total_memory_gb: number | null;
  available_memory_gb: number | null;
  numba_threads: number | null;
  gpu_backend: string;
  gpu_available: boolean;
  metal_available: boolean;
  neural_engine_available: boolean;
  detection_error: string | null;
};

export type AlloyGuideRow = {
  name: string;
  full_name: string;
  liquidus_temp: number;
  solidus_temp: number;
  freezing_range: number;
  pouring_temp: number;
  volumetric_shrinkage: number;
  feeding_multiplier: number;
  niyama_threshold: number;
};

export type ReportFormat = {
  id: 'html' | 'markdown' | 'pdf';
  label: string;
  available: boolean;
  reason: string | null;
};

export type ReportSectionToggles = {
  settings: boolean;
  parts: boolean;
  statistics: boolean;
  screenshot: boolean;
};

export type JobLog = {
  lines: string[];
  next: number;
  dropped: number;
  available: boolean;
};

export type DownloadedFile = { blob: Blob; filename: string };

function filenameFrom(response: Response, fallback: string): string {
  const disposition = response.headers.get('Content-Disposition') ?? '';
  const match = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(disposition);
  return match ? decodeURIComponent(match[1]) : fallback;
}

async function requestFile(url: string, init: RequestInit, fallback: string): Promise<DownloadedFile> {
  const response = await fetch(url, init);
  if (!response.ok) {
    const text = await response.text().catch(() => '');
    let message = `Request failed: ${response.status}`;
    try {
      const parsed: unknown = JSON.parse(text);
      if (typeof parsed === 'object' && parsed !== null && typeof (parsed as { error?: unknown }).error === 'string') {
        message = (parsed as { error: string }).error;
      }
    } catch {
    }
    throw new Error(message);
  }
  return { blob: await response.blob(), filename: filenameFrom(response, fallback) };
}

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  const data: unknown = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message =
      typeof data === 'object' && data !== null && typeof (data as { error?: unknown }).error === 'string'
        ? (data as { error: string }).error
        : `Request failed: ${response.status}`;
    throw new Error(message);
  }
  return data as T;
}

function jsonBody(payload: unknown): RequestInit {
  return {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  };
}

function adoptPart(part: StoredPart): ProjectPart {
  const { mesh_data: meshData, ...rest } = part;
  return meshData ? { ...rest, meshData } : { ...rest };
}

export const api = {
  materials: () => requestJson<MaterialInfo[]>('/api/materials'),

  listProjects: () => requestJson<ProjectSummary[]>('/api/projects'),
  createProject: (settings: ProjectSettings) =>
    requestJson<{ id: string; project: ProjectRecord }>('/api/projects', jsonBody(settings)),
  getProject: async (projectId: string, options?: { includeMesh?: boolean }) => {
    const query = options?.includeMesh ? '?include_mesh=1' : '';
    const payload = await requestJson<{ project: ProjectRecord; parts: StoredPart[] }>(
      `/api/projects/${projectId}${query}`
    );
    return { project: payload.project, parts: payload.parts.map(adoptPart) };
  },
  updateProject: (projectId: string, settings: ProjectSettings) =>
    requestJson<{ project: ProjectRecord }>(`/api/projects/${projectId}`, {
      ...jsonBody(settings),
      method: 'PUT'
    }),
  renameProject: (projectId: string, name: string) =>
    requestJson<{ project: ProjectRecord }>(`/api/projects/${projectId}/rename`, jsonBody({ name })),
  duplicateProject: (projectId: string, name?: string) =>
    requestJson<{ id: string; project: ProjectRecord }>(
      `/api/projects/${projectId}/duplicate`,
      jsonBody(name ? { name } : {})
    ),
  deleteProject: (projectId: string) =>
    requestJson<{ status: string }>(`/api/projects/${projectId}`, { method: 'DELETE' }),

  uploadFiles: (files: File[], voxelSize: number) => {
    const form = new FormData();
    for (const file of files) form.append('files', file);
    form.append('voxel_size_in', String(voxelSize));
    return requestJson<{ files: UploadedFile[] }>('/api/uploads', { method: 'POST', body: form });
  },
  addPart: async (projectId: string, part: Omit<ProjectPart, 'id'>) => {
    const payload = await requestJson<{ id: string; part: StoredPart }>(
      `/api/projects/${projectId}/parts`,
      jsonBody(part)
    );
    return { id: payload.id, part: adoptPart(payload.part) };
  },
  updatePart: async (projectId: string, part: ProjectPart) => {
    const payload = await requestJson<{ part: StoredPart }>(`/api/projects/${projectId}/parts/${part.id}`, {
      ...jsonBody(part),
      method: 'PUT'
    });
    return { part: adoptPart(payload.part) };
  },
  deletePart: (projectId: string, partId: string) =>
    requestJson<{ status: string }>(`/api/projects/${projectId}/parts/${partId}`, { method: 'DELETE' }),
  createFeatureBody: async (projectId: string, body: FeatureBodyRequest) => {
    const payload = await requestJson<{ id: string; part: StoredPart }>(
      `/api/projects/${projectId}/features`,
      jsonBody(body)
    );
    return { id: payload.id, part: adoptPart(payload.part) };
  },
  updateFeatureBody: async (projectId: string, partId: string, body: FeatureBodyRequest) => {
    const payload = await requestJson<{ part: StoredPart }>(`/api/projects/${projectId}/features/${partId}`, {
      ...jsonBody(body),
      method: 'PUT'
    });
    return { part: adoptPart(payload.part) };
  },
  saveReferencePlanes: (projectId: string, planes: ReferencePlane[]) =>
    requestJson<{ planes: ReferencePlane[] }>(`/api/projects/${projectId}/reference-planes`, {
      ...jsonBody({ planes }),
      method: 'PUT'
    }),
  saveCutFeatures: (projectId: string, cuts: CutFeature[]) =>
    requestJson<{ cuts: CutFeature[] }>(`/api/projects/${projectId}/cuts`, {
      ...jsonBody({ cuts }),
      method: 'PUT'
    }),
  storeCutBody: async (projectId: string, partId: string, mesh: { vertices: number[]; faces: number[] } | null) => {
    const payload = await requestJson<{ part: StoredPart }>(`/api/projects/${projectId}/parts/${partId}/cut`, {
      ...jsonBody({ mesh }),
      method: 'PUT'
    });
    return { part: adoptPart(payload.part) };
  },
  partBaseMesh: (projectId: string, partId: string) =>
    requestJson<{ mesh_data: MeshData }>(`/api/projects/${projectId}/parts/${partId}/base-mesh`).then(
      (payload) => payload.mesh_data
    ),
  partMesh: (projectId: string, partId: string) =>
    requestJson<{ mesh_data: MeshData }>(`/api/projects/${projectId}/parts/${partId}/mesh`).then(
      (payload) => payload.mesh_data
    ),

  partingLineInfo: (
    projectId: string,
    query: { partId?: string; rotation?: number[]; position?: number[]; scale?: number; z?: number | null },
    signal?: AbortSignal
  ) => {
    const params = new URLSearchParams();
    if (query.partId) params.set('part_id', query.partId);
    if (query.rotation) params.set('rotation', query.rotation.join(','));
    if (query.position) params.set('position', query.position.join(','));
    if (query.scale !== undefined) params.set('scale', String(query.scale));
    if (query.z !== undefined && query.z !== null) params.set('z', String(query.z));
    return requestJson<PartingLineInfo>(`/api/projects/${projectId}/parting-line?${params.toString()}`, { signal });
  },
  savePartingLine: (projectId: string, line: PartingLine | null) =>
    requestJson<{ parting_line: PartingLine | null }>(`/api/projects/${projectId}/parting-line`, {
      ...jsonBody(line ?? { parting_line: null }),
      method: 'PUT'
    }),

  meshPreview: (projectId: string, voxelSize: number) =>
    requestJson<MeshPreview>(`/api/projects/${projectId}/mesh-preview`, jsonBody({ voxel_size: voxelSize })),

  sands: () => requestJson<SandInfo[]>('/api/sands'),
  moldPreview: (projectId: string, mold: Record<string, number>, voxelSize: number) =>
    requestJson<MoldPreview>(`/api/projects/${projectId}/mold-preview`, jsonBody({ mold, voxel_size: voxelSize })),
  mlMesh: (projectId: string, targetAccuracy?: number) =>
    requestJson<MlMeshRecommendation>(
      `/api/projects/${projectId}/ml-mesh`,
      jsonBody(targetAccuracy === undefined ? {} : { target_accuracy: targetAccuracy })
    ),

  runProject: (projectId: string) =>
    requestJson<{ status: string; sim_id: string }>(`/api/projects/${projectId}/run`, { method: 'POST' }),
  cancelProject: (projectId: string) =>
    requestJson<{ status: string }>(`/api/projects/${projectId}/cancel`, { method: 'POST' }),
  status: (projectId: string) => requestJson<JobStatus>(`/api/jobs/${projectId}/status`),

  projectLog: (projectId: string, since = 0) =>
    requestJson<JobLog>(`/api/projects/${projectId}/log?since=${since}`),

  reportFormats: () => requestJson<{ formats: ReportFormat[] }>('/api/report/formats'),
  generateReport: (
    projectId: string,
    body: { format: string; sections: ReportSectionToggles; screenshot?: string }
  ) => requestFile(`/api/projects/${projectId}/report`, jsonBody(body), 'report.html'),

  exportProjectFile: (projectId: string) =>
    requestFile(`/api/projects/${projectId}/export`, { method: 'GET' }, 'project.solidus-project.json'),
  importProjectFile: (projectFile: File, cadFiles: File[]) => {
    const form = new FormData();
    form.append('project', projectFile);
    for (const file of cadFiles) form.append('files', file);
    return requestJson<{
      id: string;
      project: ProjectRecord;
      missing_parts: string[];
      schema_version: number;
    }>('/api/projects/import', { method: 'POST', body: form });
  },

  hardware: () => requestJson<HardwareInfo>('/api/hardware'),
  helpAlloys: () => requestJson<{ alloys: AlloyGuideRow[] }>('/api/help/alloys')
};

export function saveFile({ blob, filename }: DownloadedFile): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
