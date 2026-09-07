import type { ColormapId, ResultModeId } from './viewState';

export type ColormapStop = [position: number, rgb: [number, number, number]];

export type ColormapDef = {
  id: ColormapId;
  name: string;
  description: string;
  stops: ColormapStop[];
};

/** The desktop app's eight maps plus the rainbow default used by commercial casting simulators. */
export const COLORMAPS: Record<ColormapId, ColormapDef> = {
  rainbow: {
    id: 'rainbow',
    name: 'Rainbow (Default)',
    description: 'Blue through cyan, green and yellow to red',
    stops: [
      [0.0, [0, 0, 180]],
      [0.15, [0, 80, 255]],
      [0.32, [0, 200, 255]],
      [0.5, [40, 230, 90]],
      [0.66, [200, 255, 0]],
      [0.8, [255, 200, 0]],
      [0.9, [255, 90, 0]],
      [1.0, [190, 0, 0]]
    ]
  },
  red_gradient: {
    id: 'red_gradient',
    name: 'Red Gradient',
    description: 'Classic red gradient for shrinkage',
    stops: [
      [0.0, [50, 0, 0]],
      [0.5, [200, 50, 0]],
      [1.0, [255, 100, 50]]
    ]
  },
  viridis: {
    id: 'viridis',
    name: 'Viridis',
    description: 'Perceptually uniform, colorblind-friendly',
    stops: [
      [0.0, [68, 1, 84]],
      [0.25, [59, 82, 139]],
      [0.5, [33, 145, 140]],
      [0.75, [94, 201, 98]],
      [1.0, [253, 231, 37]]
    ]
  },
  plasma: {
    id: 'plasma',
    name: 'Plasma',
    description: 'Perceptually uniform, high contrast',
    stops: [
      [0.0, [13, 8, 135]],
      [0.25, [126, 3, 168]],
      [0.5, [204, 71, 120]],
      [0.75, [248, 149, 64]],
      [1.0, [240, 249, 33]]
    ]
  },
  coolwarm: {
    id: 'coolwarm',
    name: 'Cool-Warm',
    description: 'Blue-white-red diverging',
    stops: [
      [0.0, [59, 76, 192]],
      [0.5, [221, 221, 221]],
      [1.0, [180, 4, 38]]
    ]
  },
  hot_cold: {
    id: 'hot_cold',
    name: 'Hot-Cold',
    description: 'Temperature-like gradient',
    stops: [
      [0.0, [0, 0, 255]],
      [0.25, [0, 255, 255]],
      [0.5, [0, 255, 0]],
      [0.75, [255, 255, 0]],
      [1.0, [255, 0, 0]]
    ]
  },
  defect_industry: {
    id: 'defect_industry',
    name: 'Industry Standard',
    description: 'Green-yellow-orange-red defect scale',
    stops: [
      [0.0, [0, 100, 0]],
      [0.3, [255, 255, 0]],
      [0.6, [255, 165, 0]],
      [1.0, [255, 0, 0]]
    ]
  },
  niyama: {
    id: 'niyama',
    name: 'Niyama Criterion',
    description: 'Blue-cyan-yellow-red gradient for Niyama defect risk',
    stops: [
      [0.0, [0, 0, 255]],
      [0.33, [0, 255, 255]],
      [0.66, [255, 255, 0]],
      [1.0, [255, 0, 0]]
    ]
  },
  feedmod: {
    id: 'feedmod',
    name: 'Feed Modulus',
    description: 'Industry-standard feed modulus scale in inches',
    stops: [
      [0.0, [0, 210, 230]],
      [0.25, [0, 70, 230]],
      [0.5, [210, 0, 120]],
      [0.68, [255, 45, 0]],
      [0.84, [255, 210, 0]],
      [1.0, [246, 242, 208]]
    ]
  }
};

export function colormapDef(id: ColormapId | string): ColormapDef {
  return COLORMAPS[id as ColormapId] ?? COLORMAPS.rainbow;
}

/** Linear interpolation of the map at t in [0, 1]. */
export function sampleColormap(id: ColormapId | string, t: number): [number, number, number] {
  const stops = colormapDef(id).stops;
  const clamped = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0));
  for (let i = 0; i < stops.length - 1; i += 1) {
    const [p0, c0] = stops[i];
    const [p1, c1] = stops[i + 1];
    if (clamped >= p0 && clamped <= p1) {
      const f = p1 > p0 ? (clamped - p0) / (p1 - p0) : 0;
      return [c0[0] + f * (c1[0] - c0[0]), c0[1] + f * (c1[1] - c0[1]), c0[2] + f * (c1[2] - c0[2])];
    }
  }
  return stops[stops.length - 1][1];
}

/** RGBA lookup table suitable for a 1D texture (size x 1). */
export function colormapLut(id: ColormapId | string, size = 256): Uint8Array {
  const lut = new Uint8Array(size * 4);
  for (let i = 0; i < size; i += 1) {
    const [r, g, b] = sampleColormap(id, i / (size - 1));
    lut[i * 4] = Math.round(r);
    lut[i * 4 + 1] = Math.round(g);
    lut[i * 4 + 2] = Math.round(b);
    lut[i * 4 + 3] = 255;
  }
  return lut;
}

export function gradientCss(id: ColormapId | string, direction = 'to right'): string {
  const parts = colormapDef(id).stops.map(
    ([p, [r, g, b]]) => `rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)}) ${Math.round(p * 100)}%`
  );
  return `linear-gradient(${direction}, ${parts.join(', ')})`;
}

/** The map a mode starts with before the user overrides it. Rainbow everywhere by product decision. */
// eslint-disable-next-line @typescript-eslint/no-unused-vars
export function defaultColormapForMode(_mode: ResultModeId | string): ColormapId {
  return 'rainbow';
}

// ---------------------------------------------------------------------------
// Legacy per-mode stops kept for the old legend component until it is retired.
// ---------------------------------------------------------------------------

export const MODE_COLOR_STOPS: Record<string, string[]> = {
  shrinkage: ['#1c0700', '#8a2408', '#ff4d00', '#ffb35c', '#fff3d6'],
  feed_modulus: ['#0a1c2e', '#145a3e', '#3aa06a', '#ffd166', '#ff5c5c'],
  hotspot: ['#0b0b22', '#2230a8', '#6a2fe0', '#ff5c00', '#ffd21f'],
  niyama_raw: ['#0d1b3e', '#1e5fb4', '#3fa7e8', '#f2c94c', '#e5484d'],
  niyama: ['#0d1b3e', '#1e5fb4', '#3fa7e8', '#f2c94c', '#e5484d'],
  combined_feed_risk: ['#2e7d32', '#a4c639', '#ffeb3b', '#ff9800', '#e53935'],
  fraction_liquid: ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725'],
  fill_time: ['#0d1b6b', '#1e5fb4', '#8a4ec0', '#e0485c', '#ff3b1f'],
  hot_tear: ['#0b1e3d', '#1e5fa8', '#3fd8f0', '#f2e94c', '#e5484d'],
  sand_burn_in: ['#1a0f04', '#6b3a12', '#c08142', '#e8c08a', '#fdf0da'],
  feeding_quality: ['#1c0700', '#8a2408', '#ff4d00', '#ffb35c', '#fff3d6'],
  thermal_stress: ['#06201c', '#0e6f5c', '#22c9a8', '#8ff5df', '#eafff8'],
  distortion: ['#142b0a', '#3f7d1d', '#8fe056', '#d8ff9f', '#f4ffe8'],
  crack_susceptibility: ['#1c0b22', '#5b1e78', '#c23e9f', '#f48fd3', '#ffe8f8'],
  sdas: ['#1a1402', '#66520d', '#c9a520', '#ffd84d', '#fff6c9'],
  predicted_uts: ['#1c0700', '#8a2408', '#e5484d', '#ff9f9f', '#fff0f0']
};

export function colorStopsForMode(resultMode: string): string[] {
  return MODE_COLOR_STOPS[resultMode] ?? MODE_COLOR_STOPS.shrinkage;
}

export function gradientCssForMode(resultMode: string): string {
  return `linear-gradient(to right, ${colorStopsForMode(resultMode).join(', ')})`;
}
