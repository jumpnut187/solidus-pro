import type { ReactNode } from 'react';
import { RotateCcw, X } from 'lucide-react';

export type SetupPanelProps = {
  title: string;
  onClose: () => void;
  /**
   * Restores this panel's fields to their defaults. Panels apply changes live,
   * so there is no Confirm to press and no Cancel to fake — the honest
   * affordance is "put it back the way it was shipped".
   */
  onReset?: () => void;
  resetLabel?: string;
  children: ReactNode;
  footer?: ReactNode;
};

export function SetupPanel({ title, onClose, onReset, resetLabel, children, footer }: SetupPanelProps) {
  return (
    <section className="setup-panel" aria-label={title}>
      <header className="panel-head">
        <h2 className="panel-title">{title}</h2>
        <div className="spacer" />
        {onReset ? (
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            onClick={onReset}
            title={resetLabel ?? 'Reset to defaults'}
          >
            <RotateCcw size={12} />
            Reset
          </button>
        ) : null}
        <button type="button" className="btn btn-ghost btn-icon" onClick={onClose} aria-label={`Close ${title}`}>
          <X size={15} />
        </button>
      </header>
      <div className="panel-body">{children}</div>
      {footer ? <footer className="panel-foot">{footer}</footer> : null}
    </section>
  );
}

export function Group({ title, children, action }: { title: string; children: ReactNode; action?: ReactNode }) {
  return (
    <div className="group">
      <div className="group-head">
        <span className="group-title">{title}</span>
        <span className="group-rule" />
        {action}
      </div>
      {children}
    </div>
  );
}
