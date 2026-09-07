import { useEffect, useRef } from 'react';

export type ShortcutHandlers = {
  run: () => void;
  stop: () => void;
  addPart: () => void;
  screenshot: () => void;
  saveProject: () => void;
  fitCamera: () => void;
  helpGuide: () => void;
  shortcuts: () => void;
  escape: () => void;
  togglePlayback: () => void;
};

/** Typing in a field must not fire an accelerator. */
function isTextEntry(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  const tag = target.tagName;
  if (tag === 'TEXTAREA' || tag === 'SELECT') return true;
  if (tag !== 'INPUT') return false;
  // A checkbox or a slider is not text entry, and Space on a range is the
  // browser's own no-op, so the transport shortcut may still claim it.
  const type = (target as HTMLInputElement).type;
  return !['checkbox', 'radio', 'range', 'button', 'submit', 'file'].includes(type);
}

/**
 * The desktop accelerators, in the browser: F5 run, Shift+F5 stop,
 * Ctrl/Cmd+O add part, Ctrl/Cmd+S screenshot, Ctrl/Cmd+Shift+S save project,
 * Home fit, F1 help, ? shortcuts, Escape close, Space play/pause.
 *
 * Escape is deliberately left to fire even inside a text field — it is how you
 * back out of an edit — while every other shortcut defers to typing.
 */
export function useKeyboardShortcuts(handlers: ShortcutHandlers): void {
  const ref = useRef(handlers);
  ref.current = handlers;

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const handlersNow = ref.current;
      if (event.key === 'Escape') {
        handlersNow.escape();
        return;
      }
      if (isTextEntry(event.target)) return;

      const modifier = event.metaKey || event.ctrlKey;
      if (event.key === 'F5') {
        event.preventDefault();
        if (event.shiftKey) handlersNow.stop();
        else handlersNow.run();
        return;
      }
      if (modifier && (event.key === 'o' || event.key === 'O')) {
        event.preventDefault();
        handlersNow.addPart();
        return;
      }
      if (modifier && (event.key === 's' || event.key === 'S')) {
        event.preventDefault();
        if (event.shiftKey) handlersNow.saveProject();
        else handlersNow.screenshot();
        return;
      }
      if (modifier) return;
      if (event.key === 'F1') {
        event.preventDefault();
        handlersNow.helpGuide();
        return;
      }
      // '?' is Shift+/ on most layouts, so match the produced character
      // rather than the physical key.
      if (event.key === '?') {
        event.preventDefault();
        handlersNow.shortcuts();
        return;
      }
      if (event.key === 'Home') {
        event.preventDefault();
        handlersNow.fitCamera();
        return;
      }
      if (event.key === ' ' || event.code === 'Space') {
        event.preventDefault();
        handlersNow.togglePlayback();
      }
    }

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);
}
