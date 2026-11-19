/**
 * Probe and measurement markers.
 *
 * Both tools answer a question about a specific place on the casting, so the
 * answer belongs at that place rather than only in a panel: a probe reading
 * with no marker leaves you guessing which spot you actually hit, and a
 * distance with no line leaves you guessing what you measured between.
 *
 * Labels are billboarded planes carrying a `DynamicTexture`. That is heavier
 * than an HTML overlay, but an overlay has to be re-projected every frame from
 * React and drifts behind the camera during an orbit; a billboard is placed
 * once and tracks exactly.
 */
import { StandardMaterial } from '@babylonjs/core/Materials/standardMaterial';
import { DynamicTexture } from '@babylonjs/core/Materials/Textures/dynamicTexture';
import { Color3 } from '@babylonjs/core/Maths/math.color';
import { Vector3 } from '@babylonjs/core/Maths/math.vector';
import { Mesh } from '@babylonjs/core/Meshes/mesh';
import { MeshBuilder } from '@babylonjs/core/Meshes/meshBuilder';
import type { LinesMesh } from '@babylonjs/core/Meshes/linesMesh';
import type { Scene } from '@babylonjs/core/scene';

const PROBE_COLOR = new Color3(1, 0.86, 0.3);
const MEASURE_COLOR = new Color3(0.35, 0.85, 1);

const LABEL_WIDTH = 512;
const LABEL_HEIGHT = 128;

type Label = {
  plane: Mesh;
  texture: DynamicTexture;
  material: StandardMaterial;
};

function createLabel(scene: Scene, name: string, worldHeight: number): Label {
  const plane = MeshBuilder.CreatePlane(
    name,
    { width: worldHeight * (LABEL_WIDTH / LABEL_HEIGHT), height: worldHeight },
    scene
  );
  plane.isPickable = false;
  plane.billboardMode = Mesh.BILLBOARDMODE_ALL;
  // Ahead of the casting in the render order: a reading hidden inside the
  // part it describes is worse than no reading.
  plane.renderingGroupId = 2;

  const texture = new DynamicTexture(
    `${name}-texture`,
    { width: LABEL_WIDTH, height: LABEL_HEIGHT },
    scene,
    false
  );
  texture.hasAlpha = true;

  const material = new StandardMaterial(`${name}-mat`, scene);
  material.diffuseTexture = texture;
  material.opacityTexture = texture;
  material.emissiveColor = Color3.White();
  material.disableLighting = true;
  material.backFaceCulling = false;
  material.disableDepthWrite = true;
  plane.material = material;

  return { plane, texture, material };
}

function drawLabel(label: Label, text: string, color: string): void {
  const context = label.texture.getContext();
  context.clearRect(0, 0, LABEL_WIDTH, LABEL_HEIGHT);
  // A rounded plate behind the text, so a reading stays legible over a bright
  // region of the colormap.
  context.fillStyle = 'rgba(8, 11, 16, 0.82)';
  context.fillRect(8, 24, LABEL_WIDTH - 16, LABEL_HEIGHT - 48);
  label.texture.update();
  // drawText is what Babylon's canvas abstraction supports across backends;
  // the raw context lacks fillText on some of them.
  label.texture.drawText(text, null, 88, 'bold 54px Helvetica, Arial, sans-serif', color, null, true, true);
}

/** A single point marker plus its label. */
class Marker {
  readonly sphere: Mesh;
  private readonly _label: Label;
  private readonly _material: StandardMaterial;

  constructor(scene: Scene, name: string, size: number, color: Color3, labelColor: string) {
    const diameter = Math.max(size * 0.02, 0.06);
    this.sphere = MeshBuilder.CreateSphere(name, { diameter, segments: 12 }, scene);
    this.sphere.isPickable = false;
    this.sphere.renderingGroupId = 2;

    this._material = new StandardMaterial(`${name}-mat`, scene);
    this._material.diffuseColor = color;
    this._material.emissiveColor = color.scale(0.85);
    this._material.specularColor = Color3.Black();
    this._material.disableLighting = true;
    this.sphere.material = this._material;

    this._label = createLabel(scene, `${name}-label`, Math.max(size * 0.085, 0.35));
    this._label.plane.parent = this.sphere;
    this._label.plane.position.z = Math.max(size * 0.07, 0.3);
    this._labelColor = labelColor;
  }

  private readonly _labelColor: string;

  place(point: Vector3, text: string | null): void {
    this.sphere.position.copyFrom(point);
    this.sphere.setEnabled(true);
    if (text === null) {
      this._label.plane.setEnabled(false);
      return;
    }
    drawLabel(this._label, text, this._labelColor);
    this._label.plane.setEnabled(true);
  }

  hide(): void {
    this.sphere.setEnabled(false);
    this._label.plane.setEnabled(false);
  }

  dispose(): void {
    this._label.plane.dispose();
    this._label.texture.dispose();
    this._label.material.dispose();
    this.sphere.dispose();
    this._material.dispose();
  }
}

/**
 * Everything the probe and measure tools draw into the scene.
 *
 * Created lazily and sized to the assembly, so a marker on a two-inch riser
 * and one on a six-foot casting both read the same.
 */
export class Annotations {
  private readonly _scene: Scene;
  private _size: number;
  private _probe: Marker | null = null;
  private _measureStart: Marker | null = null;
  private _measureEnd: Marker | null = null;
  private _measureLine: LinesMesh | null = null;

  constructor(scene: Scene, assemblySize: number) {
    this._scene = scene;
    this._size = assemblySize;
  }

  setAssemblySize(size: number): void {
    this._size = size;
  }

  showProbe(point: Vector3, text: string): void {
    if (!this._probe) {
      this._probe = new Marker(this._scene, 'probe-marker', this._size, PROBE_COLOR, '#ffdc4d');
    }
    this._probe.place(point, text);
  }

  clearProbe(): void {
    this._probe?.hide();
  }

  /** The first click of a measurement: a marker with no distance yet. */
  showMeasureStart(point: Vector3): void {
    if (!this._measureStart) {
      this._measureStart = new Marker(this._scene, 'measure-a', this._size, MEASURE_COLOR, '#8fe4ff');
    }
    this._measureStart.place(point, null);
    this._measureEnd?.hide();
    this._measureLine?.setEnabled(false);
  }

  /** The completed measurement: both markers, the line, and the distance. */
  showMeasurement(start: Vector3, end: Vector3, distanceIn: number): void {
    if (!this._measureStart) {
      this._measureStart = new Marker(this._scene, 'measure-a', this._size, MEASURE_COLOR, '#8fe4ff');
    }
    if (!this._measureEnd) {
      this._measureEnd = new Marker(this._scene, 'measure-b', this._size, MEASURE_COLOR, '#8fe4ff');
    }
    this._measureStart.place(start, null);
    this._measureEnd.place(end, `${distanceIn.toFixed(3)} in`);

    // The line is rebuilt rather than moved: `updatable` lines keep their
    // original vertex count, and two points is two points.
    this._measureLine?.dispose();
    this._measureLine = MeshBuilder.CreateLines(
      'measure-line',
      { points: [start, end] },
      this._scene
    );
    this._measureLine.color = MEASURE_COLOR;
    this._measureLine.isPickable = false;
    this._measureLine.renderingGroupId = 2;
    this._measureLine.setEnabled(true);
  }

  clearMeasurements(): void {
    this._measureStart?.hide();
    this._measureEnd?.hide();
    this._measureLine?.dispose();
    this._measureLine = null;
  }

  clearAll(): void {
    this.clearProbe();
    this.clearMeasurements();
  }

  dispose(): void {
    this._probe?.dispose();
    this._measureStart?.dispose();
    this._measureEnd?.dispose();
    this._measureLine?.dispose();
    this._probe = null;
    this._measureStart = null;
    this._measureEnd = null;
    this._measureLine = null;
  }
}
