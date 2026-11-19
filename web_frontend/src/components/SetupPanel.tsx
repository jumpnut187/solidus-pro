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

export type CollapsibleGroupProps = {
  title: string;
  /** Stable key the shell persists the open/closed state under. */
  id: string;
  collapsed: boolean;
  onToggle: (collapsed: boolean) => void;
  /** A short summary shown on the header while the section is closed. */
  summary?: ReactNode;
  children: ReactNode;
};

/**
 * A titled section that folds away.
 *
 * The results panel is long enough that at 1366x768 its lower sections sat
 * below the fold; folding the ones you are not using is how a panel that deep
 * stays usable. The open/closed state is persisted, so the panel comes back
 * arranged the way it was left.
 */
export function CollapsibleGroup({
  title,
  id,
  collapsed,
  onToggle,
  summary,
  children
}: CollapsibleGroupProps) {
  const bodyId = `group-${id}`;
  return (
    <div className="group">
      <button
        type="button"
        className="group-head group-toggle"
        aria-expanded={!collapsed}
        aria-controls={bodyId}
        onClick={() => onToggle(!collapsed)}
      >
        <span className={collapsed ? 'group-chevron' : 'group-chevron is-open'} aria-hidden="true" />
        <span className="group-title">{title}</span>
        <span className="group-rule" />
        {collapsed && summary ? <span className="group-summary">{summary}</span> : null}
      </button>
      <div id={bodyId} hidden={collapsed} className="group-body">
        {children}
      </div>
    </div>
  );
}
