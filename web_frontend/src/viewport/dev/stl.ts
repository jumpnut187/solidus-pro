/**
 * A minimal binary-STL reader for the viewport bench.
 *
 * The shell keeps a part's triangles in memory from the moment it was
 * uploaded, so the real application never re-reads an STL. The bench starts
 * from a project id and nothing else, so it has to. Fifty lines of
 * `DataView` beats wiring a scene loader in for a development-only page.
 */
import type { MeshData } from '../../lib/api';

const HEADER_BYTES = 80;
const TRIANGLE_BYTES = 50;

export function parseBinaryStl(buffer: ArrayBuffer): MeshData {
  const view = new DataView(buffer);
  const triangleCount = view.getUint32(HEADER_BYTES, true);
  if (buffer.byteLength < HEADER_BYTES + 4 + triangleCount * TRIANGLE_BYTES) {
    throw new Error('Not a binary STL (or truncated)');
  }

  const vertices = new Array<number>(triangleCount * 9);
  const faces = new Array<number>(triangleCount * 3);
  let min = [Infinity, Infinity, Infinity];
  let max = [-Infinity, -Infinity, -Infinity];

  for (let triangle = 0; triangle < triangleCount; triangle += 1) {
    // Skip the facet normal; normals are recomputed from the triangles so a
    // file with wrong or zeroed normals still shades correctly.
    let offset = HEADER_BYTES + 4 + triangle * TRIANGLE_BYTES + 12;
    for (let corner = 0; corner < 3; corner += 1) {
      const base = triangle * 9 + corner * 3;
      for (let axis = 0; axis < 3; axis += 1) {
        const value = view.getFloat32(offset, true);
        offset += 4;
        vertices[base + axis] = value;
        if (value < min[axis]) min[axis] = value;
        if (value > max[axis]) max[axis] = value;
      }
      faces[triangle * 3 + corner] = triangle * 3 + corner;
    }
  }

  if (!Number.isFinite(min[0])) {
    min = [0, 0, 0];
    max = [0, 0, 0];
  }

  return {
    vertices,
    faces,
    bounds: { x: [min[0], max[0]], y: [min[1], max[1]], z: [min[2], max[2]] },
    volume: 0,
    surface_area: 0,
    vertex_count: triangleCount * 3,
    face_count: triangleCount
  };
}
