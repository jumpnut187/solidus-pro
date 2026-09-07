/** Typed client for the Solidus Pro HTTP API. */

/** The ten part types the engine understands (`core/part_manager.py`). */
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
  | 'gate';

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
  /** Original file (or CAD body) stem, free of the storage prefix. */
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
  meshData?: MeshData;
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
  // The pour point clicked on the 3D assembly, or null when none has been set
  // yet. The mold-filling analysis is gated on this being non-null. Persisted
  // server-side inside the project's existing `ui_state` JSON blob.
  pour_location?: PourLocationPayload | null;
};

/** A row of `GET /api/projects`. */
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

/** `GET /api/projects/<id>`: the stored project row. */
export type ProjectRecord = ProjectSettings & {
  id: string;
  status: string;
  created_at: string;
  updated_at: string;
  description: string | null;
  summary: Record<string, unknown>;
};

/** A part as the server stores it — `mesh_data` only when it was requested. */
type StoredPart = Omit<ProjectPart, 'meshData'> & { mesh_data?: MeshData };

export type ResultPayload = {
  data: {
    voxel_count?: number;
    total_volume?: number;
    available_modes?: string[];
    shrinkage_stats?: { max: number; mean: number; affected_count: number };
    hotspot_stats?: { max: number; mean: number; count: number };
    raw_ranges?: Record<string, { min: number; max: number; mean: number }>;
  };
  visualization?: unknown;
};

export type JobStatus = {
  status: string;
  progress: number;
  message?: string;
  queue_position?: number;
};

export type SurfacePayload = {
  mode: string;
  empty: boolean;
  positions: number[];
  normals?: number[];
  indices: number[];
  active_count: number;
  vertex_count: number;
  triangle_count: number;
  reason?: string;
};

export type VolumePayload = {
  mode: string;
  empty: boolean;
  centers: number[];
  sizes: number[];
  values: number[];
  source_count?: number;
  sample_count?: number;
};

export type ResultField = {
  values: number[];
  empty: boolean;
  min: number;
  max: number;
  mean: number;
};

export type ResultFieldsPayload = {
  version: number;
  empty: boolean;
  centers: number[];
  sizes: number[];
  modes: string[];
  fields: Record<string, ResultField>;
  solidification: {
    normalized_times: number[];
    min_time: number;
    max_time: number;
    front_width: number;
  };
  fraction_liquid?: {
    frames: number[][];
    frame_percents: number[];
  };
  /**
   * Fill Time sweep field: values are pre-transformed to
   * `max_finite_time_s - arrival_time` so the shared "value >= cutoff"
   * active-voxel filter sweeps later arrivals in as the slider advances.
   * `misrun[i]` flags a voxel the fill never reached -- always rendered,
   * regardless of the sweep position. Null/absent when no filling stage ran.
   */
  fill_time?: {
    values: number[];
    misrun: boolean[];
    max_finite_time_s: number;
    filled_voxels: number;
    unfilled_voxels: number;
  } | null;
  source_count: number;
  sample_count: number;
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

/** Move the server's `mesh_data` onto the camelCase key the viewport reads. */
function adoptPart(part: StoredPart): ProjectPart {
  const { mesh_data: meshData, ...rest } = part;
  return meshData ? { ...rest, meshData } : { ...rest };
}

export const api = {
  materials: () => requestJson<MaterialInfo[]>('/api/materials'),

  // --- Projects ------------------------------------------------------------
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

  // --- Parts ---------------------------------------------------------------
  uploadFiles: (files: File[]) => {
    const form = new FormData();
    for (const file of files) form.append('files', file);
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
  /** Triangle geometry for a stored part, used to restore a reloaded project. */
  partMesh: (projectId: string, partId: string) =>
    requestJson<{ mesh_data: MeshData }>(`/api/projects/${projectId}/parts/${partId}/mesh`).then(
      (payload) => payload.mesh_data
    ),

  // --- Mesh ----------------------------------------------------------------
  meshPreview: (projectId: string, voxelSize: number) =>
    requestJson<MeshPreview>(`/api/projects/${projectId}/mesh-preview`, jsonBody({ voxel_size: voxelSize })),
  mlMesh: (projectId: string, targetAccuracy?: number) =>
    requestJson<MlMeshRecommendation>(
      `/api/projects/${projectId}/ml-mesh`,
      jsonBody(targetAccuracy === undefined ? {} : { target_accuracy: targetAccuracy })
    ),

  // --- Runs ----------------------------------------------------------------
  runProject: (projectId: string) =>
    requestJson<{ status: string; sim_id: string }>(`/api/projects/${projectId}/run`, { method: 'POST' }),
  cancelProject: (projectId: string) =>
    requestJson<{ status: string }>(`/api/projects/${projectId}/cancel`, { method: 'POST' }),
  status: (projectId: string) => requestJson<JobStatus>(`/api/jobs/${projectId}/status`),
  results: (projectId: string) => requestJson<ResultPayload>(`/api/projects/${projectId}/results?summary_only=1`),
  resultFields: (projectId: string, options?: { maxPoints?: number }) => {
    const params = new URLSearchParams({ max_points: String(options?.maxPoints ?? 120000) });
    return requestJson<ResultFieldsPayload>(`/api/projects/${projectId}/result-fields?${params.toString()}`);
  },
  surface: (projectId: string, mode: string, threshold: number, timePercent: number) => {
    const params = new URLSearchParams({
      mode,
      threshold: threshold.toFixed(4),
      time_percent: timePercent.toFixed(1)
    });
    return requestJson<SurfacePayload>(`/api/projects/${projectId}/surface?${params.toString()}`);
  },
  volume: (projectId: string, mode: string, timePercent: number) => {
    const params = new URLSearchParams({ mode, time_percent: timePercent.toFixed(1) });
    return requestJson<VolumePayload>(`/api/projects/${projectId}/volume?${params.toString()}`);
  },

  // --- Environment ---------------------------------------------------------
  hardware: () => requestJson<HardwareInfo>('/api/hardware'),
  helpAlloys: () => requestJson<{ alloys: AlloyGuideRow[] }>('/api/help/alloys')
};
