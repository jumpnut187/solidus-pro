import { Group, SetupPanel } from '../SetupPanel';
import { formatNumber } from '../../lib/format';
import type { MaterialInfo } from '../../lib/api';

export type MaterialPanelProps = {
  alloy: string;
  materials: MaterialInfo[];
  onSelectAlloy: (alloy: string) => void;
  onClose: () => void;
};

type Row = { label: string; value: string };

/** The read-only property sheet the desktop Material dialog shows, in sections. */
function sections(material: MaterialInfo): { title: string; rows: Row[] }[] {
  return [
    {
      title: 'Temperatures',
      rows: [
        { label: 'Liquidus', value: `${formatNumber(material.liquidus_temp, 0)} °C` },
        { label: 'Solidus', value: `${formatNumber(material.solidus_temp, 0)} °C` },
        { label: 'Freezing range', value: `${formatNumber(material.freezing_range, 0)} °C` },
        { label: 'Pour temperature', value: `${formatNumber(material.pouring_temp, 0)} °C` }
      ]
    },
    {
      title: 'Shrinkage and feeding',
      rows: [
        { label: 'Volumetric shrinkage', value: `${formatNumber(material.volumetric_shrinkage, 2)} %` },
        { label: 'Linear shrinkage', value: `${formatNumber(material.linear_shrinkage, 2)} %` },
        { label: 'Feeding multiplier', value: formatNumber(material.feeding_multiplier, 1) },
        { label: 'Critical solid fraction', value: formatNumber(material.critical_fraction_solid, 2) },
        { label: 'Niyama threshold', value: formatNumber(material.niyama_threshold, 2) }
      ]
    },
    {
      title: 'Thermal',
      rows: [
        { label: 'Conductivity', value: `${formatNumber(material.thermal_conductivity, 1)} W/m·K` },
        { label: 'Specific heat', value: `${formatNumber(material.specific_heat, 0)} J/kg·K` },
        { label: 'Latent heat', value: `${formatNumber(material.latent_heat / 1000, 0)} kJ/kg` },
        { label: 'Diffusivity', value: `${formatNumber(material.thermal_diffusivity * 1e6, 3)} mm²/s` }
      ]
    },
    {
      title: 'Density and structure',
      rows: [
        { label: 'Density (liquid)', value: `${formatNumber(material.density_liquid, 0)} kg/m³` },
        { label: 'Density (solid)', value: `${formatNumber(material.density_solid, 0)} kg/m³` },
        { label: 'Reference SDAS', value: `${formatNumber(material.dendrite_arm_spacing, 0)} µm` },
        { label: 'Surface tension', value: `${formatNumber(material.surface_tension, 2)} N/m` }
      ]
    },
    {
      title: 'Defect susceptibility',
      rows: [
        { label: 'Hot tear', value: formatNumber(material.hot_tear_susceptibility, 2) },
        { label: 'Sand burn-in', value: formatNumber(material.sand_burn_in_susceptibility, 2) },
        { label: 'Gas sensitivity', value: formatNumber(material.gas_sensitivity, 2) }
      ]
    }
  ];
}

export function MaterialPanel({ alloy, materials, onSelectAlloy, onClose }: MaterialPanelProps) {
  const material = materials.find((item) => item.name === alloy) ?? null;
  return (
    <SetupPanel title="Material" onClose={onClose}>
      <label className="field">
        <span className="field-label">Alloy</span>
        <select className="select" value={alloy} onChange={(event) => onSelectAlloy(event.target.value)}>
          {materials.length === 0 ? <option value={alloy}>{alloy}</option> : null}
          {materials.map((item) => (
            <option key={item.name} value={item.name}>
              {item.name} — {item.full_name}
            </option>
          ))}
        </select>
      </label>

      {material ? (
        <>
          <Group title="Properties">
            <div className="property-sheet">
              {sections(material).map((section) => (
                <div key={section.title}>
                  <p className="property-section-title">{section.title}</p>
                  {section.rows.map((row) => (
                    <div className="property-row" key={row.label}>
                      <span className="property-label">{row.label}</span>
                      <span className="property-value">{row.value}</span>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </Group>
          <p className="field-hint">
            Alloy properties come from the simulation engine's material database and are not editable here.
          </p>
        </>
      ) : (
        <p className="status-note">Loading alloy properties...</p>
      )}
    </SetupPanel>
  );
}
