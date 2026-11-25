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

export function isSectionCap(mesh: AbstractMesh | null | undefined): boolean {
  return Boolean(mesh?.name.startsWith(CAP_NAME_PREFIX));
}

export type CapEntry = {
  mesh: Mesh;
  material: PBRMaterial;
  plugin: FieldPaintPluginMaterial;
};

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
    mesh.isPickable = true;

    const material = new PBRMaterial(`section-cap-${axis}-mat`, this._scene);
    material.backFaceCulling = false;
    material.twoSidedLighting = true;
    material.metallic = 0.1;
    material.roughness = 0.65;
    material.albedoColor = new Color3(0.5, 0.52, 0.55);
    const plugin = this._createPlugin(material);
    plugin.state.discardOutside = true;
    mesh.material = material;

    const entry = { mesh, material, plugin };
    this._caps.set(axis, entry);
    return entry;
  }

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

      entry.material.clipPlane = axis === 'x' ? null : planes.x;
      entry.material.clipPlane2 = axis === 'y' ? null : planes.y;
      entry.material.clipPlane3 = axis === 'z' ? null : planes.z;
      entry.mesh.setEnabled(true);
    }
  }

  orient(section: Record<SectionAxis, { flip: boolean }>): void {
    for (const axis of SECTION_AXES) {
      const entry = this._caps.get(axis);
      if (!entry) continue;
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
