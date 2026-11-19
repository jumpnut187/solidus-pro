import { Sliders } from 'lucide-react';

import { Group, SetupPanel } from '../SetupPanel';
import { PARAM_TABS, paramsForTab } from '../../lib/solidificationFields';
import { changedCount } from '../../lib/solidificationValues';
import type { SolidificationParams } from '../../lib/api';

export type SolidificationPanelProps = {
  params: SolidificationParams;
  onOpenModal: () => void;
  onReset: () => void;
  onClose: () => void;
};

/**
 * A summary of the solver parameters with a way into the full dialog.
 *
 * Forty-five numeric fields do not belong in a 380px panel next to the canvas;
 * they belong in a dialog the user opens deliberately. What is useful at a
 * glance is which groups have been moved off their defaults.
 */
export function SolidificationPanel({ params, onOpenModal, onReset, onClose }: SolidificationPanelProps) {
  const totalChanged = PARAM_TABS.reduce((sum, tab) => sum + changedCount(params, tab.id), 0);
  return (
    <SetupPanel title="Solidification" onClose={onClose} onReset={onReset} resetLabel="Reset all parameters">
      <p className="field-hint">
        Solver constants for feeding, chills, exothermic sleeves, the air gap, thermal boundaries, timing, hot
        tearing and mold filling.
      </p>

      <button type="button" className="btn btn-primary btn-block btn-lg" onClick={onOpenModal}>
        <Sliders size={14} />
        Open parameters
      </button>

      <Group title="Groups">
        <div className="property-sheet">
          {PARAM_TABS.map((tab) => {
            const changed = changedCount(params, tab.id);
            const { main, filter } = paramsForTab(tab.id);
            return (
              <div className="property-row" key={tab.id}>
                <span className="property-label">{tab.label}</span>
                <span className="property-value">
                  {changed > 0 ? (
                    <span className="chip chip-accent">{changed} changed</span>
                  ) : (
                    <span className="text-faint">{main.length + filter.length} defaults</span>
                  )}
                </span>
              </div>
            );
          })}
        </div>
      </Group>

      <p className="status-note status-note-info">
        {totalChanged === 0
          ? 'Every parameter is at its shipped default.'
          : `${totalChanged} parameter${totalChanged === 1 ? '' : 's'} differ from the defaults and will be used for the next run.`}
      </p>
    </SetupPanel>
  );
}
