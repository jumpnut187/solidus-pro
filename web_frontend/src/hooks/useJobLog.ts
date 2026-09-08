import { useCallback, useEffect, useRef, useState } from 'react';

import { api } from '../lib/api';

const POLL_MS = 1200;

/**
 * The running job's progress log.
 *
 * Polled incrementally: the server keeps the whole ring buffer and hands back
 * only what is new, so a four-minute run does not re-download five hundred
 * lines every second. Polling stops when the drawer is shut, because nobody is
 * reading it — but the buffer lives on the server, so opening it later still
 * shows the whole run.
 */
export function useJobLog(projectId: string | null, active: boolean, isRunning: boolean) {
  const [lines, setLines] = useState<string[]>([]);
  const [available, setAvailable] = useState(false);
  /** Lines the ring buffer discarded before we asked for them. */
  const [dropped, setDropped] = useState(0);
  const cursorRef = useRef(0);

  // A different project has a different log; never show one under the other.
  useEffect(() => {
    setLines([]);
    setAvailable(false);
    setDropped(0);
    cursorRef.current = 0;
  }, [projectId]);

  const fetchOnce = useCallback(async () => {
    if (!projectId) return;
    try {
      const payload = await api.projectLog(projectId, cursorRef.current);
      setAvailable(payload.available);
      if (payload.next < cursorRef.current) {
        // A re-run resets the buffer, so the sequence goes backwards. Start
        // again rather than splicing new lines onto the previous run's log.
        cursorRef.current = payload.next;
        setDropped(0);
        setLines(payload.lines);
        return;
      }
      if (payload.dropped > 0) setDropped((current) => current + payload.dropped);
      if (payload.lines.length) {
        cursorRef.current = payload.next;
        setLines((current) => [...current, ...payload.lines]);
      }
    } catch {
      // A log that cannot be read is not worth interrupting anyone over.
    }
  }, [projectId]);

  useEffect(() => {
    if (!projectId || !active) return;
    void fetchOnce();
    if (!isRunning) return;
    const timer = window.setInterval(() => void fetchOnce(), POLL_MS);
    return () => window.clearInterval(timer);
  }, [projectId, active, isRunning, fetchOnce]);

  return { lines, available, dropped, refresh: fetchOnce };
}
