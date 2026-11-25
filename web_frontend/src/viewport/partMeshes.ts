import { Material } from '@babylonjs/core/Materials/material';
import { Color3 } from '@babylonjs/core/Maths/math.color';
import type { Plane } from '@babylonjs/core/Maths/math.plane';
import { LinesMesh } from '@babylonjs/core/Meshes/linesMesh';
import { Mesh } from '@babylonjs/core/Meshes/mesh';
import { VertexData } from '@babylonjs/core/Meshes/mesh.vertexData';
import { PBRMaterial } from '@babylonjs/core/Materials/PBR/pbrMaterial';
import type { Scene } from '@babylonjs/core/scene';
import '@babylonjs/core/Rendering/outlineRenderer';

import type { ProjectPart } from '../lib/api';
import { buildCreasedGeometry } from './creaseGeometry';
import { FieldPaintPluginMaterial } from './fieldPaintPlugin';
import { applyPartTransform } from './partTransform';

export const PART_COLORS: Record<string, [number, number, number]> = {
  cast: [0.7, 0.75, 0.8],
  riser: [0.9, 0.6, 0.2],
  riser_post: [0.8, 0.5, 0.3],
  chill: [0.3, 0.3, 0.5],
  core: [0.3, 0.78, 0.4],
  mold: [0.62, 0.4, 0.86],
  pour_cup: [0.45, 0.8, 0.75],
  sprue: [0.3, 0.68, 0.7],
  runner: [0.2, 0.55, 0.65],
  filter: [0.85, 0.85, 0.35],
  gate: [0.15, 0.45, 0.6]
};

const PART_FINISH: Record<string, { metallic: number; roughness: number }> = {
  cast: { metallic: 0.0, roughness: 0.62 },
  riser: { metallic: 0.0, roughness: 0.7 },
  riser_post: { metallic: 0.0, roughness: 0.7 },
  chill: { metallic: 0.0, roughness: 0.55 },
  core: { metallic: 0.0, roughness: 0.9 },
  mold: { metallic: 0.0, roughness: 0.95 },
  pour_cup: { metallic: 0.0, roughness: 0.75 },
  sprue: { metallic: 0.0, roughness: 0.7 },
  runner: { metallic: 0.0, roughness: 0.7 },
  filter: { metallic: 0.0, roughness: 0.9 },
  gate: { metallic: 0.0, roughness: 0.7 }
};

const EDGE_COLOR = new Color3(0.06, 0.07, 0.08);
const EDGE_ALPHA_OPAQUE = 0.85;
const EDGE_ALPHA_GHOST = 0.35;

const SELECTION_COLOR = new Color3(1, 0.78, 0.2);

export const SAND_ALPHA = 0.28;

export const AUTO_SAND_PREFIX = '__auto_';

function isSand(part: ProjectPart): boolean {
  return part.part_type === 'mold' || part.part_type === 'core';
}

const GHOST_COLOR = new Color3(0.74, 0.76, 0.79);
const GHOST_FINISH = { metallic: 0.1, roughness: 0.75 };

export function partColor(partType: string): Color3 {
  const rgb = PART_COLORS[partType] ?? PART_COLORS.cast;
  return new Color3(rgb[0], rgb[1], rgb[2]);
}

export type PartEntry = {
  mesh: Mesh;
  material: PBRMaterial;
  plugin: FieldPaintPluginMaterial;
  edges: LinesMesh | null;
  geometryKey: string;
};

function geometryKey(part: ProjectPart): string {
  return `${part.filepath}:${part.meshData?.vertex_count ?? 0}:${part.meshData?.face_count ?? 0}`;
}

export function syncPartMeshes(
  scene: Scene,
  parts: ProjectPart[],
  entries: Map<string, PartEntry>,
  createPlugin: (material: PBRMaterial) => FieldPaintPluginMaterial
): void {
  const wanted = new Map(parts.map((part) => [part.id, geometryKey(part)]));
  for (const [id, entry] of entries) {
    const key = wanted.get(id);
    if (key === undefined || key !== entry.geometryKey) {
      entry.mesh.dispose();
      entry.material.dispose();
      entries.delete(id);
    }
  }

  for (const part of parts) {
    if (!entries.has(part.id)) {
      const created = createPartMesh(scene, part, createPlugin);
      if (created) entries.set(part.id, created);
    }
  }
}

function createPartMesh(
  scene: Scene,
  part: ProjectPart,
  createPlugin: (material: PBRMaterial) => FieldPaintPluginMaterial
): PartEntry | null {
  const positions = part.meshData?.vertices;
  const indices = part.meshData?.faces;
  if (!positions?.length || !indices?.length) return null;

  const mesh = new Mesh(`part-${part.id}`, scene);
  mesh.isPickable = !part.id.startsWith(AUTO_SAND_PREFIX);
  const creased = buildCreasedGeometry(positions, indices);
  const vertexData = new VertexData();
  vertexData.positions = creased.positions;
  vertexData.indices = creased.indices;
  mesh.sideOrientation = Material.CounterClockWiseSideOrientation;
  vertexData.normals = creased.normals;
  vertexData.applyToMesh(mesh);

  const material = new PBRMaterial(`part-mat-${part.id}`, scene);
  material.backFaceCulling = true;
  material.enableSpecularAntiAliasing = true;
  material.zOffset = 1;
  const plugin = createPlugin(material);
  mesh.material = material;
  mesh.outlineColor = SELECTION_COLOR;
  mesh.outlineWidth = 0.02;

  let edges: LinesMesh | null = null;
  if (creased.edgeSegments.length > 0) {
    edges = new LinesMesh(`edges-${part.id}`, scene, mesh);
    const segmentCount = creased.edgeSegments.length / 6;
    const lineIndices = new Uint32Array(segmentCount * 2);
    for (let i = 0; i < lineIndices.length; i += 1) lineIndices[i] = i;
    const lineData = new VertexData();
    lineData.positions = creased.edgeSegments;
    lineData.indices = lineIndices;
    lineData.applyToMesh(edges);
    edges.color = EDGE_COLOR;
    edges.alpha = EDGE_ALPHA_OPAQUE;
    edges.isPickable = false;
  }

  return { mesh, material, plugin, edges, geometryKey: geometryKey(part) };
}

export function setPartClipPlanes(
  entry: PartEntry,
  x: Plane | null,
  y: Plane | null,
  z: Plane | null
): void {
  entry.material.clipPlane = x;
  entry.material.clipPlane2 = y;
  entry.material.clipPlane3 = z;
  const lineMaterial = entry.edges?.material;
  if (lineMaterial) {
    lineMaterial.clipPlane = x;
    lineMaterial.clipPlane2 = y;
    lineMaterial.clipPlane3 = z;
  }
}

export type PartAppearance = {
  selectedPartId: string | null;
  opacity: number;
  showModel: boolean;
  assemblySize: number;
  depthPrePass: boolean;
  neutralGhost: boolean;
  resultsShowing: boolean;
};

export function applyPartAppearance(
  parts: ProjectPart[],
  entries: Map<string, PartEntry>,
  appearance: PartAppearance
): void {
  const opacity = Math.min(Math.max(appearance.opacity, 0), 1);
  const translucent = opacity < 0.999;

  for (const part of parts) {
    const entry = entries.get(part.id);
    if (!entry) continue;
    const { mesh, material } = entry;
    const finish = PART_FINISH[part.part_type] ?? PART_FINISH.cast;
    const sand = isSand(part);
    const ghost = appearance.neutralGhost && !sand;
    const alpha = sand ? Math.min(opacity, SAND_ALPHA) : opacity;

    material.albedoColor = ghost ? GHOST_COLOR : partColor(part.part_type);
    material.metallic = ghost ? GHOST_FINISH.metallic : finish.metallic;
    material.roughness = ghost ? GHOST_FINISH.roughness : finish.roughness;
    material.alpha = alpha;

    if (sand) {
      material.transparencyMode = PBRMaterial.PBRMATERIAL_ALPHABLEND;
      material.needDepthPrePass = false;
      material.separateCullingPass = true;
    } else if (translucent) {
      material.transparencyMode = PBRMaterial.PBRMATERIAL_ALPHABLEND;
      material.needDepthPrePass = appearance.depthPrePass;
      material.separateCullingPass = appearance.depthPrePass;
    } else {
      material.transparencyMode = PBRMaterial.PBRMATERIAL_OPAQUE;
      material.needDepthPrePass = false;
      material.separateCullingPass = false;
    }

    applyPartTransform(mesh, part);
    mesh.setEnabled(appearance.showModel && part.visible !== false && !(sand && appearance.resultsShowing));
    if (entry.edges) {
      entry.edges.alpha = translucent || sand ? EDGE_ALPHA_GHOST : EDGE_ALPHA_OPAQUE;
    }

    mesh.renderOutline = part.id === appearance.selectedPartId;
    mesh.outlineWidth = Math.max(appearance.assemblySize * 0.004, 0.01);
  }
}
