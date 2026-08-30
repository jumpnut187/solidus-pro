export type PartType = 'cast' | 'riser' | 'riser_post' | 'chill' | 'core';

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
  filepath: string;
  part_type: PartType;
  dimensions: [number, number, number];
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
  volumetric_shrinkage: number;
  feeding_multiplier: number;
  niyama_threshold: number;
  hot_tear_susceptibility: number;
  sand_burn_in_susceptibility: number;
};

export type ProjectSettings = {
  name?: string;
  alloy: string;
  voxel_size: number;
  refinement_level: number;
  memory_mode: string;
  analysis_options: Record<string, boolean | number | string>;
  performance_options: Record<string, boolean | number | string>;
  solidification_params: Record<string, Record<string, number | Record<string, number>>>;
  ui_state?: Record<string, unknown>;
  // The MAGMASOFT-style pour point clicked on the 3D assembly, or null when
  // none has been set yet. The mold-filling analysis is gated on this being
  // non-null -- see AnalysisDrawer. Persisted server-side inside the
  // project's existing `ui_state` JSON blob (see web/repository.py).
  pour_location?: PourLocationPayload | null;
};

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
  source_count: number;
  sample_count: number;
};

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || `Request failed: ${response.status}`);
  }
  return data as T;
}

export const api = {
  materials: () => requestJson<MaterialInfo[]>('/api/materials'),
  createProject: (settings: ProjectSettings) =>
    requestJson<{ id: string; project: unknown }>('/api/projects', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings)
    }),
  updateProject: (projectId: string, settings: ProjectSettings) =>
    requestJson<{ project: unknown }>(`/api/projects/${projectId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings)
    }),
  uploadFiles: async (files: File[]) => {
    const form = new FormData();
    for (const file of files) form.append('files', file);
    return requestJson<{ files: UploadedFile[] }>('/api/uploads', {
      method: 'POST',
      body: form
    });
  },
  addPart: (projectId: string, part: Omit<ProjectPart, 'id'>) =>
    requestJson<{ id: string; part: ProjectPart }>(`/api/projects/${projectId}/parts`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(part)
    }),
  updatePart: (projectId: string, part: ProjectPart) =>
    requestJson<{ part: ProjectPart }>(`/api/projects/${projectId}/parts/${part.id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(part)
    }),
  deletePart: (projectId: string, partId: string) =>
    requestJson<{ status: string }>(`/api/projects/${projectId}/parts/${partId}`, {
      method: 'DELETE'
    }),
  meshPreview: (projectId: string, voxelSize: number) =>
    requestJson<{
      estimated_voxels: number;
      estimated_memory_mb: number;
      can_proceed: boolean;
      message: string;
      recommended_voxel_size: number;
    }>(`/api/projects/${projectId}/mesh-preview`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ voxel_size: voxelSize })
    }),
  mlMesh: (projectId: string, targetAccuracy?: number) =>
    requestJson<{
      base_voxel_size: number;
      refinement_level: number;
      confidence: number;
      reason: string;
      used_model: boolean;
      warning?: string;
    }>(`/api/projects/${projectId}/ml-mesh`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(targetAccuracy === undefined ? {} : { target_accuracy: targetAccuracy })
    }),
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
  }
};
