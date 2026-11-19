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
  molten: {
    id: 'molten',
    name: 'Molten Metal',
    description: 'Dark maroon through red and orange to white-hot yellow',
    // Read bottom to top as a cooling curve: metal that has gone solid is the
    // dark maroon of a casting off the line, the mushy zone is red, the
    // free-flowing pool is orange, and metal at pouring temperature is the
    // yellow a foundry sees down a ladle.
    //
    // The top stop stays a saturated yellow rather than the near-white it
    // wants to be. These values are an *albedo*: the painted casting is a lit
    // PBR surface, so the renderer multiplies them by the environment and
    // pushes them through an ACES curve, and a stop that already sits near
    // 255 in all three channels arrives on screen as cream. At the start of a
    // run every voxel is fully liquid, so the whole casting sits on this one
    // stop -- a white-ish tip there gives back exactly the silver casting this
    // map exists to get rid of. White heat is delivered where it belongs, by
    // the emissive front and its bloom, not by the ramp.
    stops: [
      [0.0, [34, 5, 7]],
      [0.13, [92, 11, 9]],
      [0.3, [166, 26, 8]],
      [0.47, [212, 64, 8]],
      [0.63, [240, 114, 8]],
      [0.78, [251, 158, 14]],
      [0.9, [255, 188, 38]],
      [1.0, [255, 214, 70]]
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

/**
 * The map a mode opens with.
 *
 * Rainbow is the convention for every field where the number is what matters,
 * and it stays the default and stays selectable everywhere. The three modes
 * below are the exception: their number *is* a temperature-like state, and a
 * blue-through-green ramp on a pool of liquid steel reads as water. Nobody
 * looking at a filling run wants to translate cyan into "still molten".
 */
const MODE_COLORMAPS: Partial<Record<ResultModeId, ColormapId>> = {
  fraction_liquid: 'molten',
  fill_time: 'molten',
  fill_animation: 'molten'
};

export function defaultColormapForMode(mode: ResultModeId | string): ColormapId {
  return MODE_COLORMAPS[mode as ResultModeId] ?? 'rainbow';
}

/**
 * Whether a colormap is one the user chose rather than one a mode handed them.
 *
 * Switching modes should follow the new mode's default, but only while the
 * colormap on screen is still whatever the *last* mode defaulted to. Once
 * someone has picked viridis on purpose, moving to another field must not
 * silently throw that away.
 */
export function isDefaultColormapFor(mode: ResultModeId | string, colormap: ColormapId): boolean {
  return colormap === defaultColormapForMode(mode);
}

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
