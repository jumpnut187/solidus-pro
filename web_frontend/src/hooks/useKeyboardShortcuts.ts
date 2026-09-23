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
};

/** Typing in a field must not fire an accelerator. */
function isTextEntry(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  const tag = target.tagName;
  if (tag === 'TEXTAREA' || tag === 'SELECT') return true;
  if (tag !== 'INPUT') return false;
  const type = (target as HTMLInputElement).type;
  return !['checkbox', 'radio', 'range', 'button', 'submit', 'file'].includes(type);
}

/**
 * Controls that give Space a meaning of their own.
 *
 * Space presses a focused button, toggles a checkbox, opens a summary and
 * activates anything with a button-like role. The playback shortcut used to
 * claim it everywhere but text fields, so pressing Space after clicking Save,
 * a speed chip or a panel toggle paused the animation instead of doing what
 * the focused control said.
 */
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

/**
 * Whether Space belongs to the element that has focus.
 *
 * The page itself and the 3D canvas are the two places Space is left to the
 * transport: the canvas is focused by every orbit, so that is where it is
 * pressed most. Anything else that can take focus is presumed to be a control
 * that wants the key, including one this list has not heard of.
 */
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

/**
 * The desktop accelerators, in the browser: F5 run, Shift+F5 stop,
 * Ctrl/Cmd+O add part, Ctrl/Cmd+S save project, Ctrl/Cmd+Shift+S save as,
 * P screenshot, Home fit, F1 help, ? shortcuts, Escape close, Space
 * play/pause. `lib/shortcuts.ts` documents exactly this set.
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
        // Claimed with or without Shift: the browser's own Save Page dialog
        // is never what someone pressing Save in this application wants.
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
      // '?' is Shift+/ on most layouts, so match the produced character
      // rather than the physical key.
      if (event.key === '?') {
        event.preventDefault();
        handlersNow.shortcuts();
        return;
      }
      if (event.key === 'p' || event.key === 'P') {
        // A held key repeats; one press is one PNG. And a bare letter pressed
        // while a dialog has focus is meant for the dialog, not for a capture
        // of the canvas behind it.
        if (event.repeat || isInDialog(event.target)) return;
        event.preventDefault();
        handlersNow.screenshot();
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
