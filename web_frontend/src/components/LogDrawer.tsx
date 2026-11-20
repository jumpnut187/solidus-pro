import { useEffect, useRef } from 'react';
import { ChevronDown, ChevronUp, Terminal } from 'lucide-react';

export type LogDrawerProps = {
  open: boolean;
  onToggle: () => void;
  lines: string[];
  available: boolean;
  isRunning: boolean;
  dropped: number;
};

export function LogDrawer({ open, onToggle, lines, available, isRunning, dropped }: LogDrawerProps) {
  const bodyRef = useRef<HTMLDivElement | null>(null);
  const pinnedRef = useRef(true);

  useEffect(() => {
    const body = bodyRef.current;
    if (!body || !open || !pinnedRef.current) return;
    body.scrollTop = body.scrollHeight;
  }, [lines, open]);

  return (
    <div className={open ? 'log-drawer is-open' : 'log-drawer'}>
      <button
        type="button"
        className="log-drawer-head"
        onClick={onToggle}
        aria-expanded={open}
        aria-controls="job-log-body"
      >
        <Terminal size={13} />
        <span className="log-drawer-title">Job log</span>
        {lines.length ? <span className="chip">{lines.length}</span> : null}
        {isRunning ? <span className="chip chip-warning">running</span> : null}
        <span className="spacer" />
        {open ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
      </button>
      {open ? (
        <div className="log-drawer-body" id="job-log-body" ref={bodyRef} onScroll={(event) => {
          const element = event.currentTarget;
          pinnedRef.current = element.scrollHeight - element.scrollTop - element.clientHeight < 24;
        }}>
          {dropped > 0 ? (
            <div className="log-line log-line-muted">
              ... {dropped.toLocaleString()} earlier line{dropped === 1 ? '' : 's'} rolled out of the buffer
            </div>
          ) : null}
          {lines.length ? (
            lines.map((line, index) => (
              <div className="log-line" key={`${index}-${line}`}>
                {line}
              </div>
            ))
          ) : (
            <p className="log-empty">
              {available
                ? 'No progress lines yet.'
                : 'Nothing logged for this project. Run a simulation and the solver reports here.'}
            </p>
          )}
        </div>
      ) : null}
    </div>
  );
}
