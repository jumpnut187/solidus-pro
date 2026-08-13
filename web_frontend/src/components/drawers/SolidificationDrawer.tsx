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
];

export function SolidificationDrawer({ settings, setSettings, onClose }: Props) {
  function setSolidification(group: string, key: string, value: number) {
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

  return (
    <ConfigDrawer title="Solidification Parameters" icon={<Sparkles size={16} />} onClose={onClose}>
      {GROUPS.map((group) => (
        <div key={group.title} className="drawer-section">
          <div className="group-title">{group.title}</div>
          <div className="drawer-section-body">
            {group.params.map((row) => {
              const raw = Number((settings.solidification_params[group.key] ?? {})[row.key]);
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
