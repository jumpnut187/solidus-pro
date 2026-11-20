import { useEffect, useRef } from 'react';

export type ShortcutHandlers = {
  run: () => void;
  stop: () => void;
  addPart: () => void;
  screenshot: () => void;
  saveProject: () => void;
  saveProjectAs: () => void;
  fitCamera: () => void;
  helpGuide: () => void;
  shortcuts: () => void;
  escape: () => void;
  togglePlayback: () => void;
  toggleOrient: () => void;
};

function isTextEntry(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  const tag = target.tagName;
  if (tag === 'TEXTAREA' || tag === 'SELECT') return true;
  if (tag !== 'INPUT') return false;
  const type = (target as HTMLInputElement).type;
  return !['checkbox', 'radio', 'range', 'button', 'submit', 'file'].includes(type);
}

const SPACE_OWNERS = [
  'button',
  'a[href]',
  'summary',
  'select',
  'input',
  'textarea',
  '[contenteditable]:not([contenteditable="false"])',
  '[role="button"]',
  '[role="checkbox"]',
  '[role="switch"]',
  '[role="radio"]',
  '[role="tab"]',
  '[role="menuitem"]',
  '[role="menuitemcheckbox"]',
  '[role="menuitemradio"]',
  '[role="option"]',
  '[role="slider"]',
  '[role="combobox"]',
  '[role="listbox"]'
].join(',');

function focusOwnsSpace(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target === document.body || target === document.documentElement) return false;
  if (target instanceof HTMLCanvasElement) return false;
  if (target.closest(SPACE_OWNERS)) return true;
  return target.tabIndex >= 0;
}

function isInDialog(target: EventTarget | null): boolean {
  return target instanceof Element && target.closest('[role="dialog"], [aria-modal="true"]') !== null;
}

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
        if (event.shiftKey) handlersNow.saveProjectAs();
        else handlersNow.saveProject();
        return;
      }
      if (modifier || event.altKey) return;
      if (event.key === 'F1') {
        event.preventDefault();
        handlersNow.helpGuide();
        return;
      }
      if (event.key === '?') {
        event.preventDefault();
        handlersNow.shortcuts();
        return;
      }
      if (event.key === 'p' || event.key === 'P') {
        if (event.repeat || isInDialog(event.target)) return;
        event.preventDefault();
        handlersNow.screenshot();
        return;
      }
      if (event.key === 'o' || event.key === 'O') {
        if (event.repeat || isInDialog(event.target)) return;
        event.preventDefault();
        handlersNow.toggleOrient();
        return;
      }
      if (event.key === 'Home') {
        event.preventDefault();
        handlersNow.fitCamera();
        return;
      }
      if (event.key === ' ' || event.code === 'Space') {
        if (focusOwnsSpace(event.target)) return;
        event.preventDefault();
        handlersNow.togglePlayback();
      }
    }

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);
}
