export const MODE_COLOR_STOPS: Record<string, string[]> = {
  shrinkage: ['#1c0700', '#8a2408', '#ff4d00', '#ffb35c', '#fff3d6'],
  feed_modulus: ['#0a1c2e', '#145a3e', '#3aa06a', '#ffd166', '#ff5c5c'],
  hotspot: ['#0b0b22', '#2230a8', '#6a2fe0', '#ff5c00', '#ffd21f'],
  niyama_raw: ['#0d1b3e', '#1e5fb4', '#3fa7e8', '#f2c94c', '#e5484d'],
  niyama: ['#0d1b3e', '#1e5fb4', '#3fa7e8', '#f2c94c', '#e5484d'],
  combined_feed_risk: ['#2e7d32', '#a4c639', '#ffeb3b', '#ff9800', '#e53935'],
  fraction_liquid: ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725'],
  // Blue (fills first) -> red (fills last), the conventional flow-front
  // palette so the ramp reads as a wavefront sweeping through the casting.
  fill_time: ['#0d1b6b', '#1e5fb4', '#8a4ec0', '#e0485c', '#ff3b1f'],
  hot_tear: ['#0b1e3d', '#1e5fa8', '#3fd8f0', '#f2e94c', '#e5484d'],
  sand_burn_in: ['#1a0f04', '#6b3a12', '#c08142', '#e8c08a', '#fdf0da'],
  feeding_quality: ['#1c0700', '#8a2408', '#ff4d00', '#ffb35c', '#fff3d6'],
  thermal_stress: ['#06201c', '#0e6f5c', '#22c9a8', '#8ff5df', '#eafff8'],
  distortion: ['#142b0a', '#3f7d1d', '#8fe056', '#d8ff9f', '#f4ffe8'],
  crack_susceptibility: ['#1c0b22', '#5b1e78', '#c23e9f', '#f48fd3', '#ffe8f8'],
  sdas: ['#1a1402', '#66520d', '#c9a520', '#ffd84d', '#fff6c9'],
  predicted_uts: ['#1c0700', '#8a2408', '#e5484d', '#ff9f9f', '#fff0f0'],
};

export function colorStopsForMode(resultMode: string): string[] {
  return MODE_COLOR_STOPS[resultMode] ?? MODE_COLOR_STOPS.shrinkage;
}

export function gradientCssForMode(resultMode: string): string {
  return `linear-gradient(to right, ${colorStopsForMode(resultMode).join(', ')})`;
}
