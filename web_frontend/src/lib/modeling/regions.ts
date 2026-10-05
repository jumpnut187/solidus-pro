/**
 * Sketch regions: the pieces a sketch's closed contours divide the plane
 * into, the things Onshape lets you click to choose what an extrude takes.
 *
 * The regions themselves are computed by the kernel (`sketchRegions`), which
 * has the polygon booleans. What lives here is pure geometry the shell, the
 * viewport and the kernel all need: whether a point is in a region, and a
 * point guaranteed to be inside one. A feature names the regions it uses by
 * such points (seeds), which survive the sketch being edited as long as the
 * region still covers the same spot.
 */
import type { Vec2 } from './vec';

export type SketchRegion = {
  /** The outline and its holes, in sketch coordinates. */
  contours: Vec2[][];
  /** A point inside the region; how a feature names it. */
  seed: Vec2;
  /** Inside an odd number of contours: in the profile when no regions are picked. */
  inDefault: boolean;
  area: number;
  /** Triangles for drawing the region, as xy pairs, three vertices each. */
  triangles: Float32Array;
};

/** Even-odd point-in-polygon over every contour: holes count as outside. */
export function pointInContours(contours: Vec2[][], point: Vec2): boolean {
  let inside = false;
  const [x, y] = point;
  for (const contour of contours) {
    for (let i = 0, j = contour.length - 1; i < contour.length; j = i, i += 1) {
      const a = contour[i];
      const b = contour[j];
      if (a[1] > y !== b[1] > y && x < ((b[0] - a[0]) * (y - a[1])) / (b[1] - a[1]) + a[0]) inside = !inside;
    }
  }
  return inside;
}

/**
 * A point well inside a region: the middle of the widest span a horizontal
 * line through it crosses, trying a few heights so a line that only grazes a
 * thin part does not decide.
 */
export function interiorPoint(contours: Vec2[][]): Vec2 {
  let minY = Infinity;
  let maxY = -Infinity;
  for (const contour of contours) {
    for (const p of contour) {
      minY = Math.min(minY, p[1]);
      maxY = Math.max(maxY, p[1]);
    }
  }
  const height = maxY - minY;
  let best: { point: Vec2; width: number } | null = null;
  for (const fraction of [0.5, 0.37, 0.63, 0.25, 0.75, 0.12, 0.88]) {
    // Off any vertex height, so no crossing is counted twice.
    const y = minY + height * fraction + height * 1.234e-7;
    const xs: number[] = [];
    for (const contour of contours) {
      for (let i = 0, j = contour.length - 1; i < contour.length; j = i, i += 1) {
        const a = contour[i];
        const b = contour[j];
        if (a[1] > y !== b[1] > y) xs.push(a[0] + ((y - a[1]) * (b[0] - a[0])) / (b[1] - a[1]));
      }
    }
    xs.sort((p, q) => p - q);
    for (let k = 0; k + 1 < xs.length; k += 2) {
      const width = xs[k + 1] - xs[k];
      if (!best || width > best.width) best = { point: [(xs[k] + xs[k + 1]) / 2, y], width };
    }
  }
  return best?.point ?? contours[0]?.[0] ?? [0, 0];
}

/**
 * Which of `regions` a feature uses: the ones its seeds fall in, or, with no
 * seeds (`null`), the ones the even-odd rule fills, which is what a sketch
 * extrudes when nothing has been picked.
 */
export function regionSelected(region: SketchRegion, seeds: Vec2[] | null | undefined): boolean {
  if (!seeds) return region.inDefault;
  return seeds.some((seed) => pointInContours(region.contours, seed));
}
