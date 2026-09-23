/**
 * The keyboard shortcuts, in one place.
 *
 * The menu hints and the cheat sheet used to carry their own copies, so the
 * Tools menu promised a hardcoded Command key on a machine that wanted Ctrl.
 * `useKeyboardShortcuts` implements exactly this table; anything added here has
 * to be added there too, and the cheat-sheet test compares the two.
 */

/** Modifier label for the platform the page is actually running on. */
export const MOD_KEY: string =
  typeof navigator !== 'undefined' &&
  /Mac|iPhone|iPad/.test(navigator.userAgent + (navigator.platform ?? ''))
    ? 'Cmd'
    : 'Ctrl';

export type ShortcutSpec = { keys: string; description: string };

export const SHORTCUT_GROUPS: { group: string; items: ShortcutSpec[] }[] = [
  {
    group: 'Simulation',
    items: [
      { keys: 'F5', description: 'Run simulation' },
      { keys: 'Shift + F5', description: 'Stop simulation' }
    ]
  },
  {
    group: 'Project',
    items: [
      { keys: `${MOD_KEY} + O`, description: 'Add CAD parts' },
      // Save is on the key every other application puts it on. It used to
      // take a screenshot, so the one accelerator people press without
      // thinking produced a PNG instead of saving their work.
      { keys: `${MOD_KEY} + S`, description: 'Save project' },
      { keys: `${MOD_KEY} + Shift + S`, description: 'Save project as' },
      // A bare letter rather than a modified one: the browsers reserve most
      // modifier combinations (Cmd/Ctrl+P prints, Cmd/Ctrl+Shift+P opens a
      // private window in Firefox), and a single key is still guarded
      // against typing by the text-entry check.
      { keys: 'P', description: 'Export screenshot' }
    ]
  },
  {
    group: 'View',
    items: [
      { keys: 'Home', description: 'Fit the model to the view' },
      { keys: 'Space', description: 'Play / pause a time mode' },
      { keys: 'Escape', description: 'Close the top panel, dialog or tool' }
    ]
  },
  {
    group: 'Help',
    items: [
      { keys: 'F1', description: 'Help guide' },
      { keys: '?', description: 'This cheat sheet' }
    ]
  }
];

/** The label a menu item shows, looked up by description. */
export function shortcutFor(description: string): string | undefined {
  for (const group of SHORTCUT_GROUPS) {
    const match = group.items.find((item) => item.description === description);
    if (match) return match.keys;
  }
  return undefined;
}
