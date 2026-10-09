/** Typed client for the Solidus Pro HTTP API. */
import type { CutFeature, ExtrudeFeature, ReferencePlane } from './modeling/types';

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
  /** The sketch and extrude that made this body, for a body modeled in the browser. */
  feature?: ExtrudeFeature | null;
  /**
   * The body's STL before any extrude cut, kept by the server while a cut is
   * applied so editing or deleting the cut can restore material. Null when
   * the body is uncut.
   */
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
  // The pour point clicked on the 3D assembly, or null when none has been set
  // yet. The mold-filling analysis is gated on this being non-null. Persisted
  // server-side inside the project's existing `ui_state` JSON blob.
  pour_location?: PourLocationPayload | null;
};

/**
 * The flat parting line: the horizontal plane `z` (inches, world), cope above
 * and drag below. Stored in the project's `ui_state.parting_line`; a project
 * without it has no parting line yet.
 */
export type PartingLine = {
  z: number;
  visible: boolean;
  /** Move the plane with the casting when the casting is raised or lowered. */
  locked_to_casting: boolean;
};

/** `GET /api/projects/<id>/parting-line`: suggestions and the cope/drag split for the casting. */
export type PartingLineInfo = {
  parting_line: PartingLine | null;
  part_id: string | null;
  suggestions: { max_section: number; bottom: number; top: number; center: number };
  split_z: number;
  split: { cope_depth: number; drag_depth: number; cope_fraction: number; drag_fraction: number };
  bounds: [[number, number, number], [number, number, number]];
};

/** A stored `ui_state.parting_line`, or null when absent or malformed. */
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

/** What the modeling routes take: the body's properties, its feature and its triangles. */
export type FeatureBodyRequest = {
  name: string;
  part_type: PartType;
  is_open_top: boolean;
  visible: boolean;
  feature: ExtrudeFeature;
  /** The body as extruded; its pre-cut base when `cut_mesh` is sent. */
  mesh: { vertices: number[]; faces: number[] };
  /** The body after the extrude cuts that apply to it. */
  cut_mesh?: { vertices: number[]; faces: number[] } | null;
};

/** A part as the server stores it — `mesh_data` only when it was requested. */
type StoredPart = Omit<ProjectPart, 'meshData'> & { mesh_data?: MeshData };

export type JobStatus = {
  status: string;
  progress: number;
  message?: string;
  queue_position?: number;
};

export type VoxelMeshPreview = {
  mesh_type?: undefined;
  estimated_voxels: number;
  estimated_memory_mb: number;
  can_proceed: boolean;
  message: string;
  recommended_voxel_size: number;
};

/**
 * `web/routes.py::tet_mesh_estimate`; `recommended_voxel_size` is null under the node limit.
 * `can_proceed` false with `falls_back_to_voxel` true: the run uses the voxel mesh instead.
 */
export type TetMeshPreview = {
  mesh_type: 'tet';
  estimated_nodes: number;
  max_nodes: number;
  edge_length_in: number;
  can_proceed: boolean;
  falls_back_to_voxel: boolean;
  message: string;
  recommended_voxel_size: number | null;
};

export type MeshPreview = VoxelMeshPreview | TetMeshPreview;

export type MlMeshRecommendation = {
  base_voxel_size: number;
  refinement_level: number;
  confidence: number;
  reason: string;
  used_model: boolean;
  warning?: string;
  // Present only when the request asked for mesh_type 'tet'.
  mesh_type?: 'tet';
  estimated_nodes?: number;
  max_nodes?: number;
  edge_length_in?: number;
  can_proceed?: boolean;
};

export type HardwareInfo = {
  cpu_brand: string;
  cpu_vendor: string;
  physical_cores: number | null;
  logical_cores: number | null;
  performance_cores: number | null;
  efficiency_cores: number | null;
  /** CPU core tiers under the names the OS reports, e.g. Performance / Efficiency. */
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

// --- Reports ---------------------------------------------------------------

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

// --- Job log ---------------------------------------------------------------

export type JobLog = {
  lines: string[];
  /** Monotonic cursor to pass as `since` on the next poll. */
  next: number;
  /** Lines that rolled off the ring buffer before the client asked for them. */
  dropped: number;
  available: boolean;
};

/** A file the server sent as an attachment, with the name it chose. */
export type DownloadedFile = { blob: Blob; filename: string };

function filenameFrom(response: Response, fallback: string): string {
  const disposition = response.headers.get('Content-Disposition') ?? '';
  const match = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(disposition);
  return match ? decodeURIComponent(match[1]) : fallback;
}

async function requestFile(url: string, init: RequestInit, fallback: string): Promise<DownloadedFile> {
  const response = await fetch(url, init);
  if (!response.ok) {
    // An error comes back as JSON even on a download route, so read it as
    // text and surface the server's reason rather than "500".
    const text = await response.text().catch(() => '');
    let message = `Request failed: ${response.status}`;
    try {
      const parsed: unknown = JSON.parse(text);
      if (typeof parsed === 'object' && parsed !== null && typeof (parsed as { error?: unknown }).error === 'string') {
        message = (parsed as { error: string }).error;
      }
    } catch {
      /* not JSON; keep the status message */
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
  // voxelSize (inches) sets the server's STEP tessellation tolerance.
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
  /** Store a body modeled in the browser: its triangles and the feature that made them. */
  createFeatureBody: async (projectId: string, body: FeatureBodyRequest) => {
    const payload = await requestJson<{ id: string; part: StoredPart }>(
      `/api/projects/${projectId}/features`,
      jsonBody(body)
    );
    return { id: payload.id, part: adoptPart(payload.part) };
  },
  /** Replace a modeled body after its sketch or extrude changed. */
  updateFeatureBody: async (projectId: string, partId: string, body: FeatureBodyRequest) => {
    const payload = await requestJson<{ part: StoredPart }>(`/api/projects/${projectId}/features/${partId}`, {
      ...jsonBody(body),
      method: 'PUT'
    });
    return { part: adoptPart(payload.part) };
  },
  /** Replace the project's reference planes (kept in its ui_state). */
  saveReferencePlanes: (projectId: string, planes: ReferencePlane[]) =>
    requestJson<{ planes: ReferencePlane[] }>(`/api/projects/${projectId}/reference-planes`, {
      ...jsonBody({ planes }),
      method: 'PUT'
    }),
  /** Replace the project's extrude cut definitions (kept in its ui_state). */
  saveCutFeatures: (projectId: string, cuts: CutFeature[]) =>
    requestJson<{ cuts: CutFeature[] }>(`/api/projects/${projectId}/cuts`, {
      ...jsonBody({ cuts }),
      method: 'PUT'
    }),
  /**
   * Store a body as its extrude cuts left it (in its own frame), or with
   * `null` point it back at its pre-cut base.
   */
  storeCutBody: async (projectId: string, partId: string, mesh: { vertices: number[]; faces: number[] } | null) => {
    const payload = await requestJson<{ part: StoredPart }>(`/api/projects/${projectId}/parts/${partId}/cut`, {
      ...jsonBody({ mesh }),
      method: 'PUT'
    });
    return { part: adoptPart(payload.part) };
  },
  /** A cut body's geometry before any cut. */
  partBaseMesh: (projectId: string, partId: string) =>
    requestJson<{ mesh_data: MeshData }>(`/api/projects/${projectId}/parts/${partId}/base-mesh`).then(
      (payload) => payload.mesh_data
    ),
  /** Triangle geometry for a stored part, used to restore a reloaded project. */
  partMesh: (projectId: string, partId: string) =>
    requestJson<{ mesh_data: MeshData }>(`/api/projects/${projectId}/parts/${partId}/mesh`).then(
      (payload) => payload.mesh_data
    ),

  // --- Orient & Parting Line -------------------------------------------------
  /**
   * Parting-line suggestions and the cope/drag split for a body (the casting by
   * default), measured at the placement passed rather than the stored one, so
   * the answer never lags a part write still in flight.
   */
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
  /** Store the parting line, or remove it with null. */
  savePartingLine: (projectId: string, line: PartingLine | null) =>
    requestJson<{ parting_line: PartingLine | null }>(`/api/projects/${projectId}/parting-line`, {
      ...jsonBody(line ?? { parting_line: null }),
      method: 'PUT'
    }),

  // --- Mesh ----------------------------------------------------------------
  meshPreview: (projectId: string, voxelSize: number, meshType: string) =>
    requestJson<MeshPreview>(
      `/api/projects/${projectId}/mesh-preview`,
      jsonBody({ voxel_size: voxelSize, mesh_type: meshType })
    ),
  mlMesh: (projectId: string, meshType: string, targetAccuracy?: number) =>
    requestJson<MlMeshRecommendation>(
      `/api/projects/${projectId}/ml-mesh`,
      jsonBody(
        targetAccuracy === undefined
          ? { mesh_type: meshType }
          : { mesh_type: meshType, target_accuracy: targetAccuracy }
      )
    ),

  // --- Runs ----------------------------------------------------------------
  runProject: (projectId: string) =>
    requestJson<{ status: string; sim_id: string }>(`/api/projects/${projectId}/run`, { method: 'POST' }),
  cancelProject: (projectId: string) =>
    requestJson<{ status: string }>(`/api/projects/${projectId}/cancel`, { method: 'POST' }),
  status: (projectId: string) => requestJson<JobStatus>(`/api/jobs/${projectId}/status`),

  // There is no results fetcher here: the viewport reads dense grids from the
  // binary sidecar through `lib/gridApi.ts`, and the statistics panel reads
  // `GridMeta.stats`. The sampled-field, surface and volume routes that these
  // used to call have been removed from the server as well.

  /** Progress lines for a run; `since` returns only what is new. */
  projectLog: (projectId: string, since = 0) =>
    requestJson<JobLog>(`/api/projects/${projectId}/log?since=${since}`),

  // --- Reports -------------------------------------------------------------
  reportFormats: () => requestJson<{ formats: ReportFormat[] }>('/api/report/formats'),
  generateReport: (
    projectId: string,
    body: { format: string; sections: ReportSectionToggles; screenshot?: string }
  ) => requestFile(`/api/projects/${projectId}/report`, jsonBody(body), 'report.html'),

  // --- Project files -------------------------------------------------------
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

  // --- Environment ---------------------------------------------------------
  hardware: () => requestJson<HardwareInfo>('/api/hardware'),
  helpAlloys: () => requestJson<{ alloys: AlloyGuideRow[] }>('/api/help/alloys')
};

/** Hand a fetched attachment to the browser as a download. */
export function saveFile({ blob, filename }: DownloadedFile): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  // Revoking synchronously can beat the download in some browsers.
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
