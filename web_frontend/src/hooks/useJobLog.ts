import { useCallback, useEffect, useRef, useState } from 'react';

import { api } from '../lib/api';

const POLL_MS = 1200;

export function useJobLog(projectId: string | null, active: boolean, isRunning: boolean) {
  const [lines, setLines] = useState<string[]>([]);
  const [available, setAvailable] = useState(false);
  const [dropped, setDropped] = useState(0);
  const cursorRef = useRef(0);

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
