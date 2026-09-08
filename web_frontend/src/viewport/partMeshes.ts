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
import { Color3, Mesh, PBRMaterial, VertexData } from '@babylonjs/core';
import type { Scene } from '@babylonjs/core';

import type { ProjectPart } from '../lib/api';
import { FieldPaintPluginMaterial } from './fieldPaintPlugin';

/**
 * Surface color per part type, mirroring `core.part_manager.PART_TYPE_COLORS`
 * so the browser and the desktop viewer name the same body the same color.
 */
export const PART_COLORS: Record<string, [number, number, number]> = {
  cast: [0.7, 0.75, 0.8],
  riser: [0.9, 0.6, 0.2],
  riser_post: [0.8, 0.5, 0.3],
  chill: [0.3, 0.3, 0.5],
  core: [0.6, 0.5, 0.4],
  pour_cup: [0.45, 0.8, 0.75],
  sprue: [0.3, 0.68, 0.7],
  runner: [0.2, 0.55, 0.65],
  filter: [0.85, 0.85, 0.35],
  gate: [0.15, 0.45, 0.6]
};

/** Roughness and metallic per part type: a sand core should not look like steel. */
const PART_FINISH: Record<string, { metallic: number; roughness: number }> = {
  cast: { metallic: 0.75, roughness: 0.35 },
  riser: { metallic: 0.35, roughness: 0.6 },
  riser_post: { metallic: 0.4, roughness: 0.55 },
  chill: { metallic: 0.85, roughness: 0.28 },
  core: { metallic: 0.0, roughness: 0.85 },
  pour_cup: { metallic: 0.2, roughness: 0.7 },
  sprue: { metallic: 0.35, roughness: 0.55 },
  runner: { metallic: 0.35, roughness: 0.55 },
  filter: { metallic: 0.0, roughness: 0.9 },
  gate: { metallic: 0.4, roughness: 0.5 }
};

const SELECTION_COLOR = new Color3(1, 0.78, 0.2);

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
  const vertexData = new VertexData();
  vertexData.positions = positions;
  vertexData.indices = indices;
  const normals: number[] = [];
  VertexData.ComputeNormals(positions, indices, normals);
  vertexData.normals = normals;
  vertexData.applyToMesh(mesh);

  const material = new PBRMaterial(`part-mat-${part.id}`, scene);
  material.backFaceCulling = true;
  material.enableSpecularAntiAliasing = true;
  const plugin = createPlugin(material);
  mesh.material = material;
  mesh.outlineColor = SELECTION_COLOR;
  mesh.outlineWidth = 0.02;

  return { mesh, material, plugin, geometryKey: geometryKey(part) };
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

    material.albedoColor = appearance.neutralGhost ? GHOST_COLOR : partColor(part.part_type);
    material.metallic = appearance.neutralGhost ? GHOST_FINISH.metallic : finish.metallic;
    material.roughness = appearance.neutralGhost ? GHOST_FINISH.roughness : finish.roughness;
    material.alpha = opacity;

    if (translucent) {
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

    mesh.position.set(part.position_x, part.position_y, part.position_z);
    mesh.rotation.set(
      (part.rotation_x * Math.PI) / 180,
      (part.rotation_y * Math.PI) / 180,
      (part.rotation_z * Math.PI) / 180
    );
    const scale = part.scale || 1;
    mesh.scaling.set(scale, scale, scale);
    mesh.setEnabled(appearance.showModel && part.visible !== false);

    // Selection reads as an outline rather than as an emissive tint: tinting
    // the surface is indistinguishable from a hot region of the result field,
    // which is the one thing the color on this surface has to mean.
    mesh.renderOutline = part.id === appearance.selectedPartId;
    mesh.outlineWidth = Math.max(appearance.assemblySize * 0.004, 0.01);
  }
}
