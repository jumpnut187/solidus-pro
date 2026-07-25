import { useEffect, useRef } from 'react';
import {
  ArcRotateCamera,
  Camera,
  Color3,
  Color4,
  DirectionalLight,
  DynamicTexture,
  Engine,
  GlowLayer,
  HemisphericLight,
  Matrix,
  Mesh,
  MeshBuilder,
  PBRMaterial,
  ParticleSystem,
  PointerEventTypes,
  Quaternion,
  Scene,
  StandardMaterial,
  Vector3,
  VertexData
} from '@babylonjs/core';
import type { ProjectPart, ResultFieldsPayload } from '../lib/api';

type Props = {
  parts: ProjectPart[];
  selectedPartId: string | null;
  resultFields: ResultFieldsPayload | null;
  resultMode: string;
  threshold: number;
  timePercent: number;
  modelOpacity: number;
  showModel: boolean;
  theme: 'dark' | 'light';
  cameraCommand: string | null;
  onCameraHandled: () => void;
  projectionMode: 'perspective' | 'orthographic';
  onPartSelected?: (partId: string | null) => void;
};

const partColors: Record<string, Color3> = {
  cast: new Color3(0.62, 0.71, 0.74),
  riser: new Color3(1.0, 0.56, 0.12),
  riser_post: new Color3(0.92, 0.38, 0.12),
  chill: new Color3(0.18, 0.44, 0.92),
  core: new Color3(0.65, 0.45, 0.25)
};

const resultColors: Record<string, Color3> = {
  shrinkage: new Color3(1, 0.1, 0.04),
  feed_modulus: new Color3(1, 0.82, 0.1),
  hotspot: new Color3(1, 0.45, 0.02),
  niyama_raw: new Color3(0.15, 0.75, 1),
  niyama: new Color3(0.72, 0.24, 1),
  combined_feed_risk: new Color3(1, 0.08, 0.28),
  fraction_liquid: new Color3(1, 0.23, 0.02),
  hot_tear: new Color3(0.05, 0.78, 1),
  sand_burn_in: new Color3(0.74, 0.42, 0.18),
  feeding_quality: new Color3(1, 0.05, 0.2),
  thermal_stress: new Color3(0.2, 1, 0.85),
  distortion: new Color3(0.55, 1, 0.1),
  crack_susceptibility: new Color3(1, 0.25, 0.68),
  sdas: new Color3(1, 0.78, 0.12),
  predicted_uts: new Color3(0.95, 0.08, 0.08)
};

type ActiveVoxelSet = {
  centers: number[];
  sizes: number[];
  values: number[];
  indices: number[];
  count: number;
  baseSize: number;
};

export function BabylonViewport({
  parts,
  selectedPartId,
  resultFields,
  resultMode,
  threshold,
  timePercent,
  modelOpacity,
  showModel,
  theme,
  cameraCommand,
  onCameraHandled,
  projectionMode,
  onPartSelected
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const sceneRef = useRef<Scene | null>(null);
  const engineRef = useRef<Engine | null>(null);
  const cameraRef = useRef<ArcRotateCamera | null>(null);
  const partMeshesRef = useRef<Map<string, Mesh>>(new Map());
  const resultMeshRef = useRef<Mesh | null>(null);
  const particleRef = useRef<ParticleSystem | null>(null);
  const onPartSelectedRef = useRef(onPartSelected);
  onPartSelectedRef.current = onPartSelected;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const engine = new Engine(canvas, true, { antialias: true, stencil: true, preserveDrawingBuffer: true });
    const scene = new Scene(engine);
    scene.clearColor = new Color4(0.015, 0.019, 0.023, 1);
    scene.useRightHandedSystem = true;
    scene.imageProcessingConfiguration.contrast = 1.35;
    scene.imageProcessingConfiguration.exposure = 1.1;

    const camera = new ArcRotateCamera('camera', -Math.PI / 4, Math.PI / 3, 80, Vector3.Zero(), scene);
    camera.upVector = new Vector3(0, 0, 1);
    camera.lowerRadiusLimit = 0.5;
    camera.upperRadiusLimit = 4000;
    camera.wheelDeltaPercentage = 0.015;
    camera.minZ = 0.01;
    camera.maxZ = 20000;
    camera.attachControl(canvas, true);

    const hemi = new HemisphericLight('shop-softbox', new Vector3(0.2, -0.4, 1), scene);
    hemi.intensity = 0.5;
    const key = new DirectionalLight('molten-key', new Vector3(-0.55, 0.35, -0.75), scene);
    key.intensity = 2.1;
    const rim = new DirectionalLight('rim', new Vector3(0.45, -0.8, 0.25), scene);
    rim.intensity = 0.85;

    const glow = new GlowLayer('molten-glow', scene);
    glow.intensity = 0.72;

    const grid = MeshBuilder.CreateGround('shop-floor', { width: 240, height: 240, subdivisions: 48 }, scene);
    const gridMat = new StandardMaterial('floor-mat', scene);
    gridMat.diffuseColor = new Color3(0.035, 0.045, 0.045);
    gridMat.specularColor = Color3.Black();
    grid.material = gridMat;
    grid.position.z = -0.03;

    const axes = [
      [new Vector3(0, 0, 0), new Vector3(16, 0, 0), new Color4(1, 0.1, 0.08, 1)],
      [new Vector3(0, 0, 0), new Vector3(0, 16, 0), new Color4(0.1, 0.95, 0.25, 1)],
      [new Vector3(0, 0, 0), new Vector3(0, 0, 16), new Color4(0.2, 0.42, 1, 1)]
    ] as const;
    axes.forEach(([start, end, color], index) => {
      MeshBuilder.CreateLines(`axis-${index}`, { points: [start, end], colors: [color, color] }, scene);
    });

    const pointerObserver = scene.onPointerObservable.add((pointerInfo) => {
      if (pointerInfo.type === PointerEventTypes.POINTERUP) {
        const pickResult = scene.pick(scene.pointerX, scene.pointerY);
        if (pickResult.hit && pickResult.pickedMesh) {
          const meshName = pickResult.pickedMesh.name;
          if (meshName.startsWith('part-')) {
            const partId = meshName.slice(5);
            onPartSelectedRef.current?.(partId);
          }
        }
      }
    });

    engine.runRenderLoop(() => scene.render());
    const resize = () => engine.resize();
    window.addEventListener('resize', resize);

    sceneRef.current = scene;
    engineRef.current = engine;
    cameraRef.current = camera;
    const partMeshes = partMeshesRef.current;

    return () => {
      window.removeEventListener('resize', resize);
      scene.onPointerObservable.remove(pointerObserver);
      engine.dispose();
      scene.dispose();
      sceneRef.current = null;
      engineRef.current = null;
      cameraRef.current = null;
      partMeshes.clear();
    };
  }, []);

  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    scene.clearColor = theme === 'dark' ? new Color4(0.015, 0.019, 0.023, 1) : new Color4(0.91, 0.93, 0.91, 1);
  }, [theme]);

  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    const existing = partMeshesRef.current;
    const currentIds = new Set(parts.map((part) => part.id));
    for (const [id, mesh] of existing.entries()) {
      if (!currentIds.has(id)) {
        mesh.dispose();
        existing.delete(id);
      }
    }
    for (const part of parts) {
      let mesh = existing.get(part.id);
      if (!mesh) {
        mesh = createPartMesh(scene, part);
        existing.set(part.id, mesh);
      }
      const color = partColors[part.part_type] ?? partColors.cast;
      const mat = mesh.material as PBRMaterial;
      mat.albedoColor = color;
      mat.alpha = resultFields ? Math.min(modelOpacity, 0.22) : modelOpacity;
      mat.emissiveColor = part.id === selectedPartId ? color.scale(0.25) : Color3.Black();
      mesh.position = new Vector3(part.position_x, part.position_y, part.position_z);
      mesh.rotation = new Vector3(
        (part.rotation_x * Math.PI) / 180,
        (part.rotation_y * Math.PI) / 180,
        (part.rotation_z * Math.PI) / 180
      );
      mesh.scaling = new Vector3(part.scale, part.scale, part.scale);
      mesh.renderingGroupId = 1;
      mesh.setEnabled(showModel && part.visible);
    }
    fitCamera(parts, cameraRef.current);
  }, [parts, selectedPartId, modelOpacity, showModel, resultFields]);

  useEffect(() => {
    const isLocalDebug = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
    if (!isLocalDebug || !resultFields || !parts.length) return;
    const modelBounds = boundsFromParts(parts);
    const resultBounds = boundsFromFlatCenters(resultFields.centers);
    if (!modelBounds || !resultBounds) return;
    const modelSize = modelBounds.max.subtract(modelBounds.min).length();
    const centerDelta = resultBounds.center.subtract(modelBounds.center).length();
    if (centerDelta > Math.max(modelSize * 0.2, 0.5)) {
      console.warn('Result/model bounds diverge', {
        model: boundsToDebugPayload(modelBounds),
        result: boundsToDebugPayload(resultBounds),
        centerDelta
      });
    }
  }, [parts, resultFields]);

  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    let mesh = resultMeshRef.current;
    if (!mesh) {
      mesh = MeshBuilder.CreateBox('result-voxel-instance-source', { size: 1 }, scene);
      mesh.renderingGroupId = 2;
      resultMeshRef.current = mesh;
    }
    const activeSet = deriveActiveVoxels(resultFields, resultMode, threshold, timePercent);
    if (!activeSet || !activeSet.count) {
      mesh.thinInstanceSetBuffer('matrix', new Float32Array(), 16);
      mesh.setEnabled(false);
      return;
    }
    mesh.setEnabled(true);
    const matrices = new Float32Array(activeSet.count * 16);
    const rotation = Quaternion.Identity();
    for (let i = 0; i < activeSet.indices.length; i += 1) {
      const sourceIndex = activeSet.indices[i];
      const centerOffset = sourceIndex * 3;
      const size = (resultFields?.sizes[sourceIndex] ?? activeSet.baseSize) * 0.86;
      const matrix = Matrix.Compose(
        new Vector3(size, size, size),
        rotation,
        new Vector3(activeSet.centers[centerOffset] ?? 0, activeSet.centers[centerOffset + 1] ?? 0, activeSet.centers[centerOffset + 2] ?? 0)
      );
      matrix.copyToArray(matrices, i * 16);
    }
    mesh.thinInstanceSetBuffer('matrix', matrices, 16, true);
    const mat = (mesh.material as PBRMaterial | null) ?? new PBRMaterial('result-cache-mat', scene);
    const color = resultColors[resultMode] ?? resultColors.shrinkage;
    mat.albedoColor = color;
    mat.emissiveColor = color.scale(resultMode === 'fraction_liquid' ? 0.62 : 0.18);
    mat.metallic = 0.05;
    mat.roughness = 0.22;
    mat.alpha = resultMode === 'fraction_liquid' ? 0.82 : 0.76;
    mat.backFaceCulling = false;
    mat.disableDepthWrite = false;
    mat.forceDepthWrite = true;
    mat.needDepthPrePass = false;
    mesh.material = mat;
  }, [resultFields, resultMode, threshold, timePercent]);

  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    if (particleRef.current) {
      particleRef.current.dispose();
      particleRef.current = null;
    }
    const activeSet = deriveActiveVoxels(resultFields, resultMode, threshold, timePercent);
    if (!activeSet || !activeSet.count || resultMode !== 'fraction_liquid') return;
    const particleSystem = new ParticleSystem('liquid-flow', 1400, scene);
    particleSystem.renderingGroupId = 3;
    particleSystem.particleTexture = createLiquidParticleTexture(scene);
    const liquidBounds = boundsFromActiveVoxels(activeSet);
    particleSystem.emitter = liquidBounds.center;
    particleSystem.minEmitBox = liquidBounds.min.subtract(liquidBounds.center);
    particleSystem.maxEmitBox = liquidBounds.max.subtract(liquidBounds.center);
    particleSystem.color1 = new Color4(1, 0.18, 0.02, 0.28);
    particleSystem.color2 = new Color4(1, 0.72, 0.16, 0.2);
    particleSystem.minSize = 0.035;
    particleSystem.maxSize = 0.14;
    particleSystem.minLifeTime = 0.35;
    particleSystem.maxLifeTime = 1.4;
    particleSystem.emitRate = 180;
    particleSystem.blendMode = ParticleSystem.BLENDMODE_ADD;
    particleSystem.gravity = new Vector3(0, 0, -0.02);
    particleSystem.direction1 = new Vector3(-0.6, -0.25, 0.1);
    particleSystem.direction2 = new Vector3(0.7, 0.3, 0.45);
    particleSystem.minAngularSpeed = 0;
    particleSystem.maxAngularSpeed = Math.PI;
    particleSystem.minEmitPower = 0.15;
    particleSystem.maxEmitPower = 0.65;
    particleSystem.updateSpeed = 0.012;
    particleSystem.start();
    particleRef.current = particleSystem;
  }, [resultFields, resultMode, threshold, timePercent]);

  useEffect(() => {
    if (!cameraCommand || !cameraRef.current) return;
    const camera = cameraRef.current;
    const distance = camera.radius;
    if (cameraCommand === 'top') {
      camera.alpha = -Math.PI / 2;
      camera.beta = 0.03;
    } else if (cameraCommand === 'front') {
      camera.alpha = -Math.PI / 2;
      camera.beta = Math.PI / 2;
    } else if (cameraCommand === 'right') {
      camera.alpha = 0;
      camera.beta = Math.PI / 2;
    } else if (cameraCommand === 'iso') {
      camera.alpha = -Math.PI / 4;
      camera.beta = Math.PI / 3;
    }
    camera.radius = distance;
    onCameraHandled();
  }, [cameraCommand, onCameraHandled]);

  useEffect(() => {
    const camera = cameraRef.current;
    const engine = engineRef.current;
    if (!camera || !engine) return;
    const bounds = boundsFromParts(parts);
    if (projectionMode === 'orthographic') {
      camera.mode = Camera.ORTHOGRAPHIC_CAMERA;
      if (bounds) {
        const size = bounds.max.subtract(bounds.min);
        const maxDim = Math.max(size.x, size.y, size.z, 1);
        const aspect = engine.getAspectRatio(camera);
        const margin = 1.2;
        if (aspect > 1) {
          camera.orthoTop = (maxDim * margin) / 2;
          camera.orthoBottom = -(maxDim * margin) / 2;
          camera.orthoRight = (maxDim * margin * aspect) / 2;
          camera.orthoLeft = -(maxDim * margin * aspect) / 2;
        } else {
          camera.orthoRight = (maxDim * margin) / 2;
          camera.orthoLeft = -(maxDim * margin) / 2;
          camera.orthoTop = (maxDim * margin) / aspect / 2;
          camera.orthoBottom = -(maxDim * margin) / aspect / 2;
        }
      }
    } else {
      camera.mode = Camera.PERSPECTIVE_CAMERA;
    }
  }, [projectionMode, parts]);

  return <canvas ref={canvasRef} className="viewport-canvas" />;
}

function createLiquidParticleTexture(scene: Scene): DynamicTexture {
  const texture = new DynamicTexture('liquid-particle-texture', { width: 64, height: 64 }, scene, false);
  const context = texture.getContext();
  const gradient = context.createRadialGradient(32, 32, 0, 32, 32, 30);
  gradient.addColorStop(0, 'rgba(255, 245, 188, 1)');
  gradient.addColorStop(0.38, 'rgba(255, 112, 28, 0.86)');
  gradient.addColorStop(1, 'rgba(255, 58, 12, 0)');
  context.clearRect(0, 0, 64, 64);
  context.fillStyle = gradient;
  context.fillRect(0, 0, 64, 64);
  texture.update(false);
  return texture;
}

function deriveActiveVoxels(
  resultFields: ResultFieldsPayload | null,
  resultMode: string,
  threshold: number,
  timePercent: number
): ActiveVoxelSet | null {
  if (!resultFields || resultFields.empty || !resultFields.centers.length) return null;
  const values = valuesForMode(resultFields, resultMode, timePercent);
  if (!values.length) return null;
  const cutoff = thresholdForMode(resultFields, resultMode, threshold);
  const active: Array<{ index: number; value: number }> = [];
  for (let i = 0; i < values.length; i += 1) {
    const value = values[i] ?? 0;
    if (Number.isFinite(value) && value >= cutoff) {
      active.push({ index: i, value });
    }
  }
  if (!active.length) return null;
  const maxVoxels = resultMode === 'fraction_liquid' ? 120000 : 8000;
  if (active.length > maxVoxels) {
    if (resultMode === 'fraction_liquid') {
      active.splice(0, active.length, ...evenlySampleActiveVoxels(active, maxVoxels));
    } else {
      active.sort((a, b) => b.value - a.value);
      active.length = maxVoxels;
    }
  }
  const sizes = resultFields.sizes.length ? resultFields.sizes : [0.1];
  const sortedSizes = [...sizes].sort((a, b) => a - b);
  return {
    centers: resultFields.centers,
    sizes,
    values,
    indices: active.map((item) => item.index),
    count: active.length,
    baseSize: sortedSizes[Math.floor(sortedSizes.length / 2)] ?? 0.1
  };
}

function evenlySampleActiveVoxels(active: Array<{ index: number; value: number }>, maxVoxels: number) {
  if (active.length <= maxVoxels) return active;
  const sampled: Array<{ index: number; value: number }> = [];
  const stride = active.length / maxVoxels;
  for (let i = 0; i < maxVoxels; i += 1) {
    sampled.push(active[Math.floor(i * stride)]);
  }
  return sampled;
}

function valuesForMode(
  resultFields: ResultFieldsPayload,
  resultMode: string,
  timePercent: number
): number[] {
  if (resultMode !== 'fraction_liquid') {
    return resultFields.fields[resultMode]?.values ?? [];
  }
  const frames = resultFields.fraction_liquid?.frames ?? [];
  const framePercents = resultFields.fraction_liquid?.frame_percents ?? [];
  if (frames.length) {
    return interpolateFractionLiquidFrame(frames, framePercents, timePercent);
  }
  const progress = Math.min(1, Math.max(0, timePercent / 100));
  const frontWidth = resultFields.solidification.front_width || 0.075;
  return resultFields.solidification.normalized_times.map((time) => Math.max(0, Math.min(1, (time - progress) / frontWidth)));
}

function thresholdForMode(resultFields: ResultFieldsPayload, resultMode: string, threshold: number): number {
  if (resultMode === 'fraction_liquid' || resultMode === 'niyama_raw') return 0.01;
  const fraction = Math.min(1, Math.max(0, threshold / 100));
  if (resultMode === 'feed_modulus') {
    const field = resultFields.fields.feed_modulus;
    if (!field) return 0;
    return field.min + ((field.max - field.min) * fraction);
  }
  return fraction;
}

function interpolateFractionLiquidFrame(frames: number[][], framePercents: number[], timePercent: number): number[] {
  if (!frames.length) return [];
  if (frames.length === 1 || !framePercents.length) return frames[0];
  const target = Math.min(100, Math.max(0, timePercent));
  if (target <= framePercents[0]) return frames[0];
  const lastIndex = frames.length - 1;
  if (target >= framePercents[lastIndex]) return frames[lastIndex];
  let after = 1;
  while (after < framePercents.length && framePercents[after] < target) {
    after += 1;
  }
  const before = Math.max(0, after - 1);
  const beforePercent = framePercents[before] ?? 0;
  const afterPercent = framePercents[after] ?? 100;
  const span = Math.max(afterPercent - beforePercent, 1e-6);
  const alpha = (target - beforePercent) / span;
  const beforeFrame = frames[before] ?? [];
  const afterFrame = frames[after] ?? beforeFrame;
  return beforeFrame.map((value, index) => value * (1 - alpha) + (afterFrame[index] ?? value) * alpha);
}

function boundsFromActiveVoxels(activeSet: ActiveVoxelSet) {
  const centers = activeSet.centers;
  const values = activeSet.values;
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let minZ = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  let maxZ = Number.NEGATIVE_INFINITY;
  let activeCount = 0;
  for (const sourceIndex of activeSet.indices) {
    const value = values[sourceIndex] ?? 0;
    if (value <= 0.01) continue;
    const i = sourceIndex * 3;
    const x = centers[i] ?? 0;
    const y = centers[i + 1] ?? 0;
    const z = centers[i + 2] ?? 0;
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    minZ = Math.min(minZ, z);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
    maxZ = Math.max(maxZ, z);
    activeCount += 1;
  }
  if (!activeCount) {
    return {
      center: Vector3.Zero(),
      min: new Vector3(-1, -1, -1),
      max: new Vector3(1, 1, 1)
    };
  }
  const padding = Math.max((maxX - minX + maxY - minY + maxZ - minZ) / 90, 0.2);
  const min = new Vector3(minX - padding, minY - padding, minZ - padding);
  const max = new Vector3(maxX + padding, maxY + padding, maxZ + padding);
  return {
    center: min.add(max).scale(0.5),
    min,
    max
  };
}

function createPartMesh(scene: Scene, part: ProjectPart): Mesh {
  const mesh = new Mesh(`part-${part.id}`, scene);
  const vertexData = new VertexData();
  vertexData.positions = part.meshData?.vertices ?? [];
  vertexData.indices = part.meshData?.faces ?? [];
  const normals: number[] = [];
  VertexData.ComputeNormals(vertexData.positions, vertexData.indices, normals);
  vertexData.normals = normals;
  vertexData.applyToMesh(mesh);
  const material = new PBRMaterial(`part-mat-${part.id}`, scene);
  material.albedoColor = partColors[part.part_type] ?? partColors.cast;
  material.metallic = 0.22;
  material.roughness = 0.38;
  material.alpha = 0.34;
  material.backFaceCulling = false;
  material.disableDepthWrite = true;
  material.forceDepthWrite = false;
  material.needDepthPrePass = false;
  material.transparencyMode = PBRMaterial.PBRMATERIAL_ALPHABLEND;
  mesh.material = material;
  return mesh;
}

type BoundsInfo = {
  min: Vector3;
  max: Vector3;
  center: Vector3;
};

function boundsFromParts(parts: ProjectPart[]): BoundsInfo | null {
  const bounds = parts
    .map((part) => part.meshData?.bounds)
    .filter(Boolean) as Array<{ x: [number, number]; y: [number, number]; z: [number, number] }>;
  if (!bounds.length) return null;
  const min = new Vector3(
    Math.min(...bounds.map((b) => b.x[0])),
    Math.min(...bounds.map((b) => b.y[0])),
    Math.min(...bounds.map((b) => b.z[0]))
  );
  const max = new Vector3(
    Math.max(...bounds.map((b) => b.x[1])),
    Math.max(...bounds.map((b) => b.y[1])),
    Math.max(...bounds.map((b) => b.z[1]))
  );
  return { min, max, center: min.add(max).scale(0.5) };
}

function boundsFromFlatCenters(centers: number[]): BoundsInfo | null {
  if (centers.length < 3) return null;
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let minZ = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  let maxZ = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < centers.length; i += 3) {
    const x = centers[i] ?? 0;
    const y = centers[i + 1] ?? 0;
    const z = centers[i + 2] ?? 0;
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    minZ = Math.min(minZ, z);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
    maxZ = Math.max(maxZ, z);
  }
  const min = new Vector3(minX, minY, minZ);
  const max = new Vector3(maxX, maxY, maxZ);
  return { min, max, center: min.add(max).scale(0.5) };
}

function boundsToDebugPayload(bounds: BoundsInfo) {
  return {
    min: bounds.min.asArray(),
    max: bounds.max.asArray(),
    center: bounds.center.asArray()
  };
}

function fitCamera(parts: ProjectPart[], camera: ArcRotateCamera | null): void {
  if (!camera || !parts.length) return;
  const bounds = boundsFromParts(parts);
  if (!bounds) return;
  const sizeVector = bounds.max.subtract(bounds.min);
  const size = Math.max(sizeVector.x, sizeVector.y, sizeVector.z, 1);
  camera.setTarget(bounds.center);
  camera.radius = Math.max(size * 2.15, 8);
}
