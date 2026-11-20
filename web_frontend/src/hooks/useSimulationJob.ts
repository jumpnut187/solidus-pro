import { useCallback, useEffect, useRef, useState } from 'react';

import { api } from '../lib/api';

export type JobPhase = 'idle' | 'submitting' | 'queued' | 'running' | 'completed' | 'error' | 'cancelled';

export type JobState = {
  phase: JobPhase;
  progress: number;
  message: string;
  queuePosition: number | null;
};

const IDLE: JobState = { phase: 'idle', progress: 0, message: 'Ready', queuePosition: null };

const POLL_RUNNING_MS = 650;
const POLL_QUEUED_MS = 1500;

export type SimulationJobOptions = {
  onCompleted: (projectId: string) => void;
  onError: (message: string) => void;
};

export function useSimulationJob({ onCompleted, onError }: SimulationJobOptions) {
  const [job, setJob] = useState<JobState>(IDLE);
  const [pollingId, setPollingId] = useState<string | null>(null);
  const callbacks = useRef({ onCompleted, onError });
  callbacks.current = { onCompleted, onError };

  const isBusy = job.phase === 'submitting' || job.phase === 'queued' || job.phase === 'running';

  useEffect(() => {
    if (!pollingId) return;
    let cancelled = false;
    let timer = 0;

    const poll = async () => {
      try {
        const status = await api.status(pollingId);
        if (cancelled) return;
        if (status.status === 'completed') {
          setJob({ phase: 'completed', progress: 100, message: 'Simulation complete', queuePosition: null });
          setPollingId(null);
          callbacks.current.onCompleted(pollingId);
          return;
        }
        if (status.status === 'error' || status.status === 'failed') {
          setJob({
            phase: 'error',
            progress: status.progress ?? 0,
            message: status.message || 'Simulation failed',
            queuePosition: null
          });
          setPollingId(null);
          callbacks.current.onError(status.message || 'Simulation failed');
          return;
        }
        if (status.status === 'cancelled') {
          setJob({ phase: 'cancelled', progress: 0, message: 'Simulation cancelled', queuePosition: null });
          setPollingId(null);
          return;
        }
        const queued = status.status === 'queued';
        setJob({
          phase: queued ? 'queued' : 'running',
          progress: status.progress ?? 0,
          message:
            status.message ||
            (queued
              ? status.queue_position
                ? `Queued, position ${status.queue_position}`
                : 'Queued'
              : 'Running'),
          queuePosition: status.queue_position ?? null
        });
        timer = window.setTimeout(poll, queued ? POLL_QUEUED_MS : POLL_RUNNING_MS);
      } catch (error) {
        if (cancelled) return;
        const message = error instanceof Error ? error.message : 'Status polling failed';
        setJob({ phase: 'error', progress: 0, message, queuePosition: null });
        setPollingId(null);
        callbacks.current.onError(message);
      }
    };

    void poll();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [pollingId]);

  const start = useCallback(async (projectId: string) => {
    setJob({ phase: 'submitting', progress: 0, message: 'Submitting simulation...', queuePosition: null });
    try {
      await api.runProject(projectId);
      setJob({ phase: 'queued', progress: 0, message: 'Queued', queuePosition: null });
      setPollingId(projectId);
      return true;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Simulation failed to start';
      setJob({ phase: 'error', progress: 0, message, queuePosition: null });
      callbacks.current.onError(message);
      return false;
    }
  }, []);

  const cancel = useCallback(async (projectId: string) => {
    setPollingId(null);
    try {
      await api.cancelProject(projectId);
      setJob({ phase: 'cancelled', progress: 0, message: 'Simulation cancelled', queuePosition: null });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unable to cancel the simulation';
      setJob({ phase: 'error', progress: 0, message, queuePosition: null });
      callbacks.current.onError(message);
    }
  }, []);

  const reset = useCallback((message = 'Ready') => {
    setPollingId(null);
    setJob({ ...IDLE, message });
  }, []);

  return { job, isBusy, start, cancel, reset };
}
