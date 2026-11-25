import type { Vec2 } from './vec';

export type SketchRegion = {
  contours: Vec2[][];
  seed: Vec2;
  inDefault: boolean;
  area: number;
  triangles: Float32Array;
};

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

export function regionSelected(region: SketchRegion, seeds: Vec2[] | null | undefined): boolean {
  if (!seeds) return region.inDefault;
  return seeds.some((seed) => pointInContours(region.contours, seed));
}
