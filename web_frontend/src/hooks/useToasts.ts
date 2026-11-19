import { useCallback, useRef, useState } from 'react';

export type ToastKind = 'info' | 'success' | 'warning' | 'error';

export type Toast = {
  id: number;
  kind: ToastKind;
  message: string;
};

/** How long each kind lingers. Errors stay until dismissed. */
const LIFETIME_MS: Record<ToastKind, number> = {
  info: 3200,
  success: 3200,
  warning: 6000,
  error: 0
};

const MAX_VISIBLE = 4;

/**
 * Non-blocking status and error notices.
 *
 * Nothing in the shell is allowed to interrupt the user with a browser
 * `alert()`: a message about a background job must not steal focus from the
 * canvas or a form the user is filling in.
 */
export function useToasts() {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const nextId = useRef(1);
  const timers = useRef(new Map<number, number>());

  const dismiss = useCallback((id: number) => {
    const timer = timers.current.get(id);
    if (timer !== undefined) {
      window.clearTimeout(timer);
      timers.current.delete(id);
    }
    setToasts((current) => current.filter((toast) => toast.id !== id));
  }, []);

  const push = useCallback(
    (kind: ToastKind, message: string) => {
      const id = nextId.current++;
      setToasts((current) => [...current, { id, kind, message }].slice(-MAX_VISIBLE));
      const lifetime = LIFETIME_MS[kind];
      if (lifetime > 0) {
        timers.current.set(
          id,
          window.setTimeout(() => dismiss(id), lifetime)
        );
      }
      return id;
    },
    [dismiss]
  );

  const notify = useCallback((message: string) => push('info', message), [push]);
  const succeed = useCallback((message: string) => push('success', message), [push]);
  const warn = useCallback((message: string) => push('warning', message), [push]);
  const fail = useCallback(
    (error: unknown, fallback = 'Something went wrong') =>
      push('error', error instanceof Error ? error.message : typeof error === 'string' ? error : fallback),
    [push]
  );

  return { toasts, push, notify, succeed, warn, fail, dismiss };
}
