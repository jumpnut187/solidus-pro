import type { Dispatch, SetStateAction } from 'react';
import { Sparkles } from 'lucide-react';

import { ConfigDrawer } from '../ConfigDrawer';
import type { ProjectSettings } from '../../lib/api';

type Props = {
  settings: ProjectSettings;
  setSettings: Dispatch<SetStateAction<ProjectSettings>>;
  onClose: () => void;
};

type ParamRow = {
  key: string;
  label: string;
  min: number;
  max: number;
  step: number;
  unit?: string;
};

type ParamGroup = {
  key: string;
  title: string;
  params: ParamRow[];
};

const GROUPS: ParamGroup[] = [
  {
    key: 'feeding',
    title: 'Feeding',
    params: [
      { key: 'base_feeding_distance', label: 'Base Feeding Distance', min: 1, max: 12, step: 0.5, unit: 'in' },
      { key: 'min_influence_factor', label: 'Min Influence Factor', min: 0.1, max: 1, step: 0.05 },
      { key: 'riser_max_reduction', label: 'Riser Max Reduction', min: 0, max: 1, step: 0.05 },
      { key: 'riser_min_reduction', label: 'Riser Min Reduction', min: 0, max: 3, step: 0.1 },
      { key: 'post_contact_reduction', label: 'Post-Contact Reduction', min: 0, max: 1, step: 0.05 },
      { key: 'disconnected_neck_floor', label: 'Disconnected Neck Floor', min: 0, max: 1, step: 0.05 },
      { key: 'shrink_scalar', label: 'Shrink Scalar', min: 0.25, max: 4, step: 0.05 },
      { key: 'feeding_zone_vertical_bias', label: 'Feeding Zone Vertical Bias', min: 0, max: 1, step: 0.05 },
    ],
  },
  {
    key: 'chill',
    title: 'Chill',
    params: [
      { key: 'max_chill_boost', label: 'Max Chill Boost', min: 1, max: 6, step: 0.5 },
      { key: 'shrinkage_radius_multiplier', label: 'Shrinkage Radius Multiplier', min: 0.5, max: 3, step: 0.1 },
      { key: 'shrinkage_max_reduction', label: 'Shrinkage Max Reduction', min: 0, max: 1, step: 0.05 },
      { key: 'shrinkage_min_reduction', label: 'Shrinkage Min Reduction', min: 0, max: 3, step: 0.1 },
      { key: 'thickness_fallback_voxels', label: 'Thickness Fallback Voxels', min: 1, max: 20, step: 1 },
    ],
  },
  {
    key: 'exothermic',
    title: 'Exothermic / Insulation',
    params: [
      { key: 'ignition_delay', label: 'Ignition Delay', min: 0, max: 300, step: 5, unit: 's' },
      { key: 'burn_duration', label: 'Burn Duration', min: 30, max: 600, step: 10, unit: 's' },
      { key: 'insulation_only_factor', label: 'Insulation-Only Factor', min: 0.2, max: 1, step: 0.05 },
      { key: 'full_exothermic_factor', label: 'Full Exothermic Factor', min: 0.2, max: 1, step: 0.05 },
      { key: 'decay_time_constant', label: 'Decay Time Constant', min: 10, max: 300, step: 10, unit: 's' },
    ],
  },
  {
    key: 'air_gap',
    title: 'Air Gap',
    params: [
      { key: 'time_constant', label: 'Time Constant', min: 10, max: 300, step: 10, unit: 's' },
      { key: 'surface_influence_depth', label: 'Surface Influence Depth', min: 0.1, max: 2, step: 0.1, unit: 'in' },
      { key: 'max_slowdown', label: 'Max Slowdown', min: 0.1, max: 1, step: 0.05 },
    ],
  },
  {
    key: 'thermal_boundary',
    title: 'Thermal Boundary',
    params: [
      { key: 'mold_temperature', label: 'Mold Temperature', min: 0, max: 500, step: 5, unit: 'C' },
      { key: 'chill_temperature', label: 'Chill Temperature', min: 0, max: 500, step: 5, unit: 'C' },
      { key: 'htc_cast_interface', label: 'HTC Cast Interface', min: 50, max: 2000, step: 50, unit: 'W/m²K' },
      { key: 'htc_riser_post', label: 'HTC Riser Post', min: 50, max: 2000, step: 50, unit: 'W/m²K' },
      { key: 'htc_riser', label: 'HTC Riser', min: 50, max: 2000, step: 50, unit: 'W/m²K' },
      { key: 'htc_gating', label: 'HTC Gating', min: 50, max: 2000, step: 50, unit: 'W/m²K' },
    ],
  },
  {
    key: 'timing',
    title: 'Timing',
    params: [
      { key: 'chvorinov_mold_constant', label: 'Chvorinov Mold Constant', min: 50, max: 800, step: 10, unit: 's/mm²' },
      { key: 'safety_factor_with_risers', label: 'Safety Factor w/ Risers', min: 1, max: 5, step: 0.1 },
      { key: 'safety_factor_no_risers', label: 'Safety Factor w/o Risers', min: 1, max: 5, step: 0.1 },
      { key: 'min_simulation_time', label: 'Min Simulation Time', min: 30, max: 600, step: 15, unit: 's' },
      { key: 'max_simulation_time', label: 'Max Simulation Time', min: 1800, max: 43200, step: 600, unit: 's' },
      { key: 'sand_saturation_time', label: 'Sand Saturation Time', min: 10, max: 300, step: 10, unit: 's' },
      { key: 'max_hot_sand_slowdown', label: 'Max Hot Sand Slowdown', min: 0.1, max: 1, step: 0.05 },
    ],
  },
  {
    key: 'hot_tear',
    title: 'Hot Tear',
    params: [
      { key: 'mold_restraint_index', label: 'Mold Restraint Index', min: 0, max: 1, step: 0.05 },
      { key: 'metal_mold_friction_index', label: 'Metal/Mold Friction Index', min: 0, max: 1, step: 0.05 },
      { key: 'core_restraint_index', label: 'Core Restraint Index', min: 0, max: 1, step: 0.05 },
      { key: 'low_melting_impurity_index', label: 'Low-Melting Impurity Index', min: 0, max: 1, step: 0.05 },
      { key: 'hot_tear_zone_threshold', label: 'Hot Tear Zone Threshold (0=adaptive)', min: 0, max: 1, step: 0.05 },
      { key: 'healing_relief_strength', label: 'Healing Relief Strength', min: 0, max: 1, step: 0.05 },
      { key: 'surface_restraint_depth_inches', label: 'Surface Restraint Depth', min: 0.1, max: 3, step: 0.1, unit: 'in' },
      { key: 'solid_path_search_depth', label: 'Solid Path Search Depth', min: 2, max: 20, step: 1 },
    ],
  },
  {
    key: 'filling',
    title: 'Mold Filling',
    params: [
      { key: 'pour_rate_lb_s', label: 'Pour Rate', min: 0.5, max: 50, step: 0.5, unit: 'lb/s' },
      { key: 'pour_time_s', label: 'Pour Time (0=derive)', min: 0, max: 120, step: 1, unit: 's' },
      { key: 'pour_temperature_c', label: 'Pour Temperature (0=alloy default)', min: 0, max: 1700, step: 10, unit: 'C' },
      { key: 'initial_head_in', label: 'Initial Head', min: 0.5, max: 24, step: 0.5, unit: 'in' },
      { key: 'fill_grid_coarsening', label: 'Fill Grid Coarsening', min: 1, max: 8, step: 1, unit: '× voxel' },
      { key: 'max_fill_time_s', label: 'Max Fill Time', min: 1, max: 600, step: 5, unit: 's' },
      { key: 'smagorinsky_cs', label: 'Smagorinsky Cs', min: 0.05, max: 0.3, step: 0.01 },
      { key: 'lattice_velocity_max', label: 'Lattice Velocity Max', min: 0.005, max: 0.2, step: 0.005 },
      { key: 'wall_htc_multiplier', label: 'Wall HTC Multiplier', min: 0.1, max: 5, step: 0.1 },
      { key: 'inlet_mode_code', label: 'Inlet Mode (0=Head, 1=Velocity)', min: 0, max: 1, step: 1 },
      { key: 'pour_mode_code', label: 'Pour Mode (0=Lip, 1=Bottom)', min: 0, max: 1, step: 1 },
      { key: 'vent_risers', label: 'Vent Risers Enabled (0/1)', min: 0, max: 1, step: 1 },
      { key: 'cold_shut_arrival_gap_s', label: 'Cold Shut Arrival Gap', min: 0.01, max: 10, step: 0.05, unit: 's' },
      { key: 'cold_shut_temp_margin_c', label: 'Cold Shut Temp Margin', min: 0, max: 200, step: 5, unit: 'C' },
      { key: 'entrapment_check_interval', label: 'Entrapment Check Interval', min: 1, max: 1000, step: 10, unit: 'steps' },
      { key: 'porosity', label: 'Filter Porosity', min: 0.5, max: 0.95, step: 0.01 },
      { key: 'pore_diameter_mm', label: 'Filter Pore Diameter', min: 0.5, max: 8, step: 0.1, unit: 'mm' },
      { key: 'ergun_prefactor', label: 'Filter Ergun Prefactor', min: 0.5, max: 10, step: 0.1 },
      { key: 'forchheimer_c_f', label: 'Filter Forchheimer C_f', min: 0.1, max: 2, step: 0.05 },
      { key: 'contact_angle_deg', label: 'Filter Contact Angle', min: 90, max: 160, step: 1, unit: 'deg' },
      { key: 'window_fraction', label: 'Filter Window Fraction', min: 0.2, max: 0.8, step: 0.05 },
      { key: 'rated_velocity_warn_factor', label: 'Filter Rated Velocity Warn Factor', min: 0.5, max: 5, step: 0.1 },
      { key: 'head_fraction_warn', label: 'Filter Head Fraction Warn', min: 0.05, max: 1, step: 0.05 },
      { key: 'priming_enabled', label: 'Filter Priming Enabled (0/1)', min: 0, max: 1, step: 1 },
      { key: 'drag_enabled', label: 'Filter Drag Enabled (0/1)', min: 0, max: 1, step: 1 },
    ],
  },
];

// FilterParams field names. These rows share the "filling" UI group above
// with FillingParams' own fields, but on the backend
// (SolidificationParameters.filling.filter) they live one level deeper.
// Nest/flatten them on write/read so the payload sent to the API matches
// SolidificationParameters.to_dict()'s shape.
const FILTER_KEYS = new Set([
  'porosity', 'pore_diameter_mm', 'ergun_prefactor', 'forchheimer_c_f',
  'contact_angle_deg', 'window_fraction', 'rated_velocity_warn_factor',
  'head_fraction_warn', 'priming_enabled', 'drag_enabled',
]);

export function SolidificationDrawer({ settings, setSettings, onClose }: Props) {
  function setSolidification(group: string, key: string, value: number) {
    if (group === 'filling' && FILTER_KEYS.has(key)) {
      const fillingBucket = (settings.solidification_params.filling ?? {}) as Record<string, unknown>;
      const filterBucket = (fillingBucket.filter ?? {}) as Record<string, number>;
      setSettings({
        ...settings,
        solidification_params: {
          ...settings.solidification_params,
          filling: {
            ...fillingBucket,
            filter: {
              ...filterBucket,
              [key]: value,
            },
          },
        },
      });
      return;
    }
    setSettings({
      ...settings,
      solidification_params: {
        ...settings.solidification_params,
        [group]: {
          ...(settings.solidification_params[group] ?? {}),
          [key]: value,
        },
      },
    });
  }

  function getSolidification(group: string, key: string): number {
    if (group === 'filling' && FILTER_KEYS.has(key)) {
      const fillingBucket = (settings.solidification_params.filling ?? {}) as Record<string, unknown>;
      const filterBucket = (fillingBucket.filter ?? {}) as Record<string, number>;
      return Number(filterBucket[key]);
    }
    return Number((settings.solidification_params[group] ?? {})[key]);
  }

  return (
    <ConfigDrawer title="Solidification Parameters" icon={<Sparkles size={16} />} onClose={onClose}>
      {GROUPS.map((group) => (
        <div key={group.title} className="drawer-section">
          <div className="group-title">{group.title}</div>
          <div className="drawer-section-body">
            {group.params.map((row) => {
              const raw = getSolidification(group.key, row.key);
              const value = Number.isFinite(raw) ? raw : row.min;
              const isRatio = row.min === 0 && row.max === 1 && row.step <= 0.05;
              return isRatio ? (
                <label key={row.key}>
                  {row.label} {value.toFixed(2)}
                  <input
                    type="range"
                    min={row.min}
                    max={row.max}
                    step={row.step}
                    value={value}
                    onChange={(e) => setSolidification(group.key, row.key, Number(e.target.value))}
                  />
                </label>
              ) : (
                <label key={row.key}>
                  {row.label}{row.unit ? ` (${row.unit})` : ''}
                  <input
                    type="number"
                    min={row.min}
                    max={row.max}
                    step={row.step}
                    value={value}
                    onChange={(e) => setSolidification(group.key, row.key, Number(e.target.value))}
                  />
                </label>
              );
            })}
          </div>
        </div>
      ))}
      <p className="muted">Feeding, chill, and hot-tear parameters tune the solidification model for your casting geometry.</p>
    </ConfigDrawer>
  );
}
