export type ContourInit = {
  kind: 'init';
  dims: [number, number, number];
  origin: [number, number, number];
  spacing: number;
  mask: Uint8Array;
};

export type ContourRequest = {
  kind: 'contour';
  id: number;
  frameA: Uint8Array;
  frameB: Uint8Array;
  blend: number;
  iso: number;
  castOnly: boolean;
  invert: boolean;
  blurStrength: number;
};

export type ContourResult = {
  kind: 'contour';
  id: number;
  positions: Float32Array;
  normals: Float32Array;
  indices: Uint32Array;
  scalars: Float32Array;
  vertexCount: number;
  triangleCount: number;
  elapsedMs: number;
};

export type ContourMessage = ContourInit | ContourRequest;
