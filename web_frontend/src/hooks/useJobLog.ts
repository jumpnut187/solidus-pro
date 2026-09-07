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
  const countRef = useRef(0);

  // A different project has a different log; never show one under the other.
  useEffect(() => {
    setLines([]);
    setAvailable(false);
    countRef.current = 0;
  }, [projectId]);

  const fetchOnce = useCallback(async () => {
    if (!projectId) return;
    try {
      const payload = await api.projectLog(projectId, countRef.current);
      setAvailable(payload.available);
      if (payload.total < countRef.current) {
        // The buffer was cleared or rolled over; start again rather than
        // splicing new lines onto a log that no longer matches.
        countRef.current = payload.total;
        setLines(payload.lines);
        return;
      }
      if (payload.lines.length) {
        countRef.current = payload.total;
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

  return { lines, available, refresh: fetchOnce };
}
