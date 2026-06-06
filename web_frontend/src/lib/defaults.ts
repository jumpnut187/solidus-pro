import type { ProjectSettings } from './api';

export const analysisDefaults = {
  shrinkage: true,
  hotspot: true,
  niyama: false,
  hot_tear: false,
  sand_burn_in: false,
  fraction_liquid: true,
  feeding_path: false,
  stress_analysis: false,
  microstructure: false,
  sand_restraint_index: 0.5
};

export const performanceDefaults = {
  use_gpu: false,
  use_incremental: false,
  use_chunked_voxelization: false,
  part_assignment_mode: 'parallel',
  part_assignment_workers: 0,
  part_assignment_batch_size: 2500,
  max_voxels_for_visualization: 90000
};

export const solidificationDefaults = {
  feeding: {
    base_feeding_distance: 6,
    min_influence_factor: 0.6,
    riser_max_reduction: 0.1,
    riser_min_reduction: 1,
    post_contact_reduction: 0.35,
    disconnected_neck_floor: 0.7,
    shrink_scalar: 1
  },
  chill: {
    max_chill_boost: 3,
    shrinkage_radius_multiplier: 1.5,
    shrinkage_max_reduction: 0.15,
    shrinkage_min_reduction: 1,
    thickness_fallback_voxels: 5
  },
  exothermic: {
    ignition_delay: 10,
    burn_duration: 120,
    insulation_only_factor: 0.7,
    full_exothermic_factor: 0.5,
    decay_time_constant: 60
  },
  air_gap: {
    time_constant: 60,
    surface_influence_depth: 0.5,
    max_slowdown: 0.75
  },
  thermal_boundary: {
    mold_temperature: 21,
    chill_temperature: 21,
    htc_cast_interface: 500,
    htc_riser_post: 350,
    htc_riser: 220
  },
  timing: {
    chvorinov_mold_constant: 280,
    safety_factor_with_risers: 2.5,
    safety_factor_no_risers: 1.5,
    min_simulation_time: 120,
    max_simulation_time: 14400,
    sand_saturation_time: 30,
    max_hot_sand_slowdown: 0.6
  },
  hot_tear: {
    mold_restraint_index: 0.5,
    metal_mold_friction_index: 0.5,
    core_restraint_index: 0.0,
    low_melting_impurity_index: 0.0,
    hot_tear_zone_threshold: 0.0,
    healing_relief_strength: 0.65,
    surface_restraint_depth_inches: 0.5,
    solid_path_search_depth: 8
  }
};

export function defaultSettings(): ProjectSettings {
  return {
    name: `Solidus Flow ${new Date().toLocaleString()}`,
    alloy: 'WCB',
    voxel_size: 0.1,
    refinement_level: 3,
    memory_mode: 'auto',
    analysis_options: { ...analysisDefaults },
    performance_options: { ...performanceDefaults },
    solidification_params: structuredClone(solidificationDefaults)
  };
}

export const resultModes = [
  { id: 'shrinkage', label: 'Shrinkage / Porosity', color: '#ff3f24' },
  { id: 'feed_modulus', label: 'Feed Modulus', color: '#ffd21a' },
  { id: 'hotspot', label: 'Hot Spots', color: '#ff9f1c' },
  { id: 'niyama_raw', label: 'Raw Niyama', color: '#38bdf8' },
  { id: 'niyama', label: 'Niyama Risk', color: '#b85cff' },
  { id: 'combined_feed_risk', label: 'Combined Feed Risk', color: '#f43f5e' },
  { id: 'fraction_liquid', label: 'Fraction Liquid', color: '#ff5c1f' },
  { id: 'hot_tear', label: 'Hot Tear Risk', color: '#20d3ff' },
  { id: 'sand_burn_in', label: 'Sand Burn-In', color: '#c08142' },
  { id: 'feeding_quality', label: 'Blocked Feeding', color: '#ff2d55' },
  { id: 'thermal_stress', label: 'Thermal Stress', color: '#5eead4' },
  { id: 'distortion', label: 'Distortion', color: '#a3e635' },
  { id: 'crack_susceptibility', label: 'Crack Susceptibility', color: '#f472b6' },
  { id: 'sdas', label: 'SDAS', color: '#facc15' },
  { id: 'predicted_uts', label: 'Low UTS', color: '#ef4444' }
];
