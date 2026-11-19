/**
 * Section planes and the capped faces they expose.
 *
 * Cutting a mesh with a clip plane leaves it hollow: you see through the wall
 * into the inside of a shell, which reads as a modelling error rather than as
 * a cross-section. The fix is a cap on the cut face.
 *
 * The cap here is not built from the geometry. It is a single quad lying in
 * the section plane whose fragment shader discards wherever the occupancy grid
 * says there is no metal, and paints the field everywhere else -- the same
 * shader path the outer surface uses. That is worth doing for three reasons:
 * the cut face of a casting is not a shape anyone modelled, so there is
 * nothing to cap *with*; the grid already knows exactly where the metal is,
 * including internal cavities and separate bodies; and a quad that only ever
 * moves cannot flicker while the position slider is dragged, whereas a cap
 * rebuilt from intersected triangles rebuilds its topology every frame.
 *
 * The cap for an axis is deliberately not clipped by its own plane -- it sits
 * exactly on it, where the sign of the clip distance is whatever rounding
 * says.
 */
import { PBRMaterial } from '@babylonjs/core/Materials/PBR/pbrMaterial';
import { Color3 } from '@babylonjs/core/Maths/math.color';
import { Mesh } from '@babylonjs/core/Meshes/mesh';
import { MeshBuilder } from '@babylonjs/core/Meshes/meshBuilder';
import type { Plane } from '@babylonjs/core/Maths/math.plane';
import type { AbstractMesh } from '@babylonjs/core/Meshes/abstractMesh';
import type { Scene } from '@babylonjs/core/scene';

import type { Bounds } from './cameraController';
import { FieldPaintPluginMaterial } from './fieldPaintPlugin';

export type SectionAxis = 'x' | 'y' | 'z';

export const SECTION_AXES: readonly SectionAxis[] = ['x', 'y', 'z'];

const CAP_NAME_PREFIX = 'section-cap-';

/** Whether a picked mesh is one of the cut faces rather than a part. */
export function isSectionCap(mesh: AbstractMesh | null | undefined): boolean {
  return Boolean(mesh?.name.startsWith(CAP_NAME_PREFIX));
}

export type CapEntry = {
  mesh: Mesh;
  material: PBRMaterial;
  plugin: FieldPaintPluginMaterial;
};

/**
 * A quad per axis, created once and afterwards only moved.
 *
 * Rebuilding the mesh when the slider moves is the difference between a cap
 * that slides and a cap that strobes.
 */
export class SectionCaps {
  private readonly _caps = new Map<SectionAxis, CapEntry>();

  constructor(
    private readonly _scene: Scene,
    private readonly _createPlugin: (material: PBRMaterial) => FieldPaintPluginMaterial
  ) {}

  entries(): CapEntry[] {
    return [...this._caps.values()];
  }

  private _ensure(axis: SectionAxis): CapEntry {
    const existing = this._caps.get(axis);
    if (existing) return existing;

    const mesh = MeshBuilder.CreatePlane(
      `${CAP_NAME_PREFIX}${axis}`,
      { size: 1, sideOrientation: Mesh.DOUBLESIDE },
      this._scene
    );
    // Pickable because the cut face is what a probe on a section is for: it
    // is the only place the interior field is on screen. The quad is far
    // larger than the metal it shows, so a hit on it only counts where the
    // occupancy grid says the shader drew it -- see `ResultsViewport`.
    mesh.isPickable = true;

    const material = new PBRMaterial(`section-cap-${axis}-mat`, this._scene);
    material.backFaceCulling = false;
    material.twoSidedLighting = true;
    material.metallic = 0.1;
    material.roughness = 0.65;
    // The interior of a casting is not a polished face; keeping it matte
    // stops a specular highlight from being mistaken for a hot region.
    material.albedoColor = new Color3(0.5, 0.52, 0.55);
    const plugin = this._createPlugin(material);
    plugin.state.discardOutside = true;
    mesh.material = material;

    const entry = { mesh, material, plugin };
    this._caps.set(axis, entry);
    return entry;
  }

  /**
   * Place the caps for the enabled axes and hide the rest.
   *
   * `planes` are the clip planes for all three axes; each cap takes the planes
   * of the *other* two so a cut in X is itself cut by an enabled cut in Y.
   */
  layout(
    section: Record<SectionAxis, { enabled: boolean; position: number; flip: boolean }>,
    caps: boolean,
    bounds: Bounds | null,
    planes: Record<SectionAxis, Plane | null>
  ): void {
    for (const axis of SECTION_AXES) {
      const state = section[axis];
      if (!caps || !state.enabled || !bounds) {
        this._caps.get(axis)?.mesh.setEnabled(false);
        continue;
      }

      const entry = this._ensure(axis);
      const index = SECTION_AXES.indexOf(axis);
      const min = bounds.min.asArray()[index];
      const max = bounds.max.asArray()[index];
      const position = min + (max - min) * (Math.min(Math.max(state.position, 0), 100) / 100);

      // Comfortably larger than the assembly so the quad always spans the cut.
      const size = bounds.size * 1.6;
      entry.mesh.scaling.set(size, size, 1);
      entry.mesh.position.copyFrom(bounds.center);
      entry.mesh.rotation.set(0, 0, 0);

      if (axis === 'x') {
        entry.mesh.position.x = position;
        entry.mesh.rotation.y = Math.PI / 2;
      } else if (axis === 'y') {
        entry.mesh.position.y = position;
        entry.mesh.rotation.x = Math.PI / 2;
      } else {
        entry.mesh.position.z = position;
      }

      // Its own plane would cut it; the other two should.
      entry.material.clipPlane = axis === 'x' ? null : planes.x;
      entry.material.clipPlane2 = axis === 'y' ? null : planes.y;
      entry.material.clipPlane3 = axis === 'z' ? null : planes.z;
      entry.mesh.setEnabled(true);
    }
  }

  /** Face the cap's normal along the kept side, so it lights like a surface. */
  orient(section: Record<SectionAxis, { flip: boolean }>): void {
    for (const axis of SECTION_AXES) {
      const entry = this._caps.get(axis);
      if (!entry) continue;
      // Two-sided lighting handles the rest; this only picks a nominal facing.
      const flip = section[axis].flip ? -1 : 1;
      entry.mesh.scaling.z = flip;
    }
  }

  dispose(): void {
    for (const entry of this._caps.values()) {
      entry.mesh.dispose();
      entry.material.dispose();
    }
    this._caps.clear();
  }
}
