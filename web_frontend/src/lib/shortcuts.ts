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
      { keys: `${MOD_KEY} + S`, description: 'Save project' },
      { keys: `${MOD_KEY} + Shift + S`, description: 'Save project as' },
      { keys: 'P', description: 'Export screenshot' }
    ]
  },
  {
    group: 'View',
    items: [
      { keys: 'Home', description: 'Fit the model to the view' },
      { keys: 'O', description: 'Orient & Parting Line' },
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

export function shortcutFor(description: string): string | undefined {
  for (const group of SHORTCUT_GROUPS) {
    const match = group.items.find((item) => item.description === description);
    if (match) return match.keys;
  }
  return undefined;
}
