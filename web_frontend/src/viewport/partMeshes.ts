/**
 * Part geometry and its materials.
 *
 * Parts are opaque by default. The previous renderer capped model opacity at
 * 80 percent, and at 22 percent whenever results existed, on the theory that
 * you had to see through the casting to see the result inside it. With the
 * field painted on the surface that trade is gone: the casting is a solid
 * object again, and translucency is something the user asks for rather than
 * something the result mode imposes.
 */
import { Material } from '@babylonjs/core/Materials/material';
import { Color3 } from '@babylonjs/core/Maths/math.color';
import type { Plane } from '@babylonjs/core/Maths/math.plane';
import { LinesMesh } from '@babylonjs/core/Meshes/linesMesh';
import { Mesh } from '@babylonjs/core/Meshes/mesh';
import { VertexData } from '@babylonjs/core/Meshes/mesh.vertexData';
import { PBRMaterial } from '@babylonjs/core/Materials/PBR/pbrMaterial';
import type { Scene } from '@babylonjs/core/scene';
// Registers the scene component behind `mesh.renderOutline`, the selection
// outline. Without it the property is set and nothing draws.
import '@babylonjs/core/Rendering/outlineRenderer';

import type { ProjectPart } from '../lib/api';
import { buildCreasedGeometry } from './creaseGeometry';
import { FieldPaintPluginMaterial } from './fieldPaintPlugin';
import { applyPartTransform } from './partTransform';

/**
 * Surface color per part type, mirroring `core.part_manager.PART_TYPE_COLORS`
 * so the browser and the desktop viewer name the same body the same color.
 */
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

/**
 * Roughness and metallic per part type, CAD style.
 *
 * Every part is a DIELECTRIC with a satin finish, the way SolidWorks and the
 * other CAD viewers draw a model: colour comes from the diffuse term and the
 * only highlight is a soft one from the headlight. The previous metallic
 * 0.75-0.85 at roughness 0.28-0.35 made the casting a mirror of the studio
 * environment -- a metal has almost no diffuse term, so what reached the eye
 * was reflections, not the part -- and read as chrome. A sand core is still
 * rougher than a machined chill, so the finish still says which body is which.
 */
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

/** Feature-edge lines ("shaded with edges"): near-black, like a CAD viewer. */
const EDGE_COLOR = new Color3(0.06, 0.07, 0.08);
const EDGE_ALPHA_OPAQUE = 0.85;
const EDGE_ALPHA_GHOST = 0.35;

const SELECTION_COLOR = new Color3(1, 0.78, 0.2);

/**
 * Sand bodies -- mold and core parts and the automatic mold and core preview
 * -- are see-through while setting up, so the casting inside them stays in
 * view, and are not drawn while a result is on screen: the result is about
 * the casting, its riser posts and its risers.
 */
export const SAND_ALPHA = 0.28;

/** Id prefix of the automatic mold and core preview bodies (display only, not pickable). */
export const AUTO_SAND_PREFIX = '__auto_';

function isSand(part: ProjectPart): boolean {
  return part.part_type === 'mold' || part.part_type === 'core';
}

/**
 * The casting when it is only there to say where the result *is*.
 *
 * One neutral light grey, and a dull one: in `envelope` the shell is context,
 * not content, so it carries neither the part-type colour nor the field. A
 * riser tinted its own orange or a shell painted blue at the low end of the
 * ramp both read as answers, and the only answer on screen should be the thing
 * inside.
 */
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
  /** Feature edges, parented to `mesh`; null when the part has none. */
  edges: LinesMesh | null;
  /** Identifies the triangles currently uploaded, so a swap is detectable. */
  geometryKey: string;
};

/** What makes two uploads of a part different geometry rather than the same. */
function geometryKey(part: ProjectPart): string {
  return `${part.filepath}:${part.meshData?.vertex_count ?? 0}:${part.meshData?.face_count ?? 0}`;
}

/**
 * Create or update the mesh set so it matches `parts`.
 *
 * Meshes are keyed by part id and reused: rebuilding them on every render
 * would re-upload the STL's vertex buffers whenever a checkbox moved. A part
 * whose id survives but whose *file* changed -- reloading a project, swapping
 * an STL for a revised one -- is rebuilt, because keeping the old triangles
 * under the new name would show the previous revision indefinitely.
 */
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
  // The automatic preview is not a part: nothing to select or edit.
  mesh.isPickable = !part.id.startsWith(AUTO_SAND_PREFIX);
  // Crease-angle normals, not ComputeNormals' all-faces average: see
  // creaseGeometry.ts for the starburst shading the average produced on flat
  // faces that meet a bore or a chamfer.
  const creased = buildCreasedGeometry(positions, indices);
  const vertexData = new VertexData();
  vertexData.positions = creased.positions;
  vertexData.indices = creased.indices;
  // STL triangles wind counter-clockwise seen from outside, and the scene is
  // right-handed. Babylon's defaults assume the opposite on both counts: it
  // treats clockwise as the front face and computes left-handed normals. With
  // those defaults the near, outward faces were culled and the far walls were
  // drawn from the inside with inward normals, so the casting read as an
  // x-ray of itself -- risers with no tops, and the colour on screen came
  // from the far wall while a probe read the near one. The crease normals
  // point out of a counter-clockwise face, as `ComputeNormals` does with
  // `useRightHandedSystem`.
  mesh.sideOrientation = Material.CounterClockWiseSideOrientation;
  vertexData.normals = creased.normals;
  vertexData.applyToMesh(mesh);

  const material = new PBRMaterial(`part-mat-${part.id}`, scene);
  material.backFaceCulling = true;
  material.enableSpecularAntiAliasing = true;
  // Pushed back a hair in depth so the feature edges drawn exactly on it win
  // the depth test instead of stitching in and out of the surface.
  material.zOffset = 1;
  const plugin = createPlugin(material);
  mesh.material = material;
  mesh.outlineColor = SELECTION_COLOR;
  mesh.outlineWidth = 0.02;

  // Named outside the `part-` prefix on purpose: picking identifies a part
  // by that prefix and does not consult `isPickable`.
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

/** Section planes on a part's surface AND its edges, so a cut cuts both. */
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
  /** 0..1. 1 is fully opaque. */
  opacity: number;
  showModel: boolean;
  /** Longest edge of the assembly, used to scale the selection outline. */
  assemblySize: number;
  /**
   * Whether a translucent shell may write depth before anything else draws.
   *
   * It normally should: without a pre-pass a see-through casting blends with
   * its own back faces and reads as a bag of triangles. But the pre-pass
   * writes depth for the *whole* shell, so anything drawn afterwards and
   * further away is rejected -- and when the shell is a ghost with an
   * isosurface inside it, that is precisely the thing the user is looking
   * at. The fill front came out shot through with holes where the shell had
   * claimed the depth buffer first.
   */
  depthPrePass: boolean;
  /**
   * Strip the part-type colour and show the casting as a neutral ghost.
   *
   * Set in `envelope`, where the shell exists to locate the result rather than
   * to describe itself. The field paint on top of it is suppressed separately,
   * by the plugin, so that the molten liquid body still comes through.
   */
  neutralGhost: boolean;
  /** A result is on screen: sand bodies are hidden. */
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
      // No depth pre-pass: a sand shell around the casting would otherwise
      // hide the casting it surrounds.
      material.transparencyMode = PBRMaterial.PBRMATERIAL_ALPHABLEND;
      material.needDepthPrePass = false;
      material.separateCullingPass = true;
    } else if (translucent) {
      material.transparencyMode = PBRMaterial.PBRMATERIAL_ALPHABLEND;
      // Without a depth pre-pass a translucent casting shows its own back
      // faces through its front ones and reads as a bag of triangles rather
      // than as a solid seen through -- but see `depthPrePass` above for when
      // that is the wrong trade.
      material.needDepthPrePass = appearance.depthPrePass;
      material.separateCullingPass = appearance.depthPrePass;
    } else {
      material.transparencyMode = PBRMaterial.PBRMATERIAL_OPAQUE;
      material.needDepthPrePass = false;
      material.separateCullingPass = false;
    }

    // The engine's static-XYZ rotation, as a quaternion; see partTransform.ts.
    applyPartTransform(mesh, part);
    mesh.setEnabled(appearance.showModel && part.visible !== false && !(sand && appearance.resultsShowing));
    if (entry.edges) {
      entry.edges.alpha = translucent || sand ? EDGE_ALPHA_GHOST : EDGE_ALPHA_OPAQUE;
    }

    // Selection reads as an outline rather than as an emissive tint: tinting
    // the surface is indistinguishable from a hot region of the result field,
    // which is the one thing the color on this surface has to mean.
    mesh.renderOutline = part.id === appearance.selectedPartId;
    mesh.outlineWidth = Math.max(appearance.assemblySize * 0.004, 0.01);
  }
}
