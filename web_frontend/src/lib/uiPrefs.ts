/**
 * Everything the shell remembers between sessions: which project was open,
 * how the workspace was arranged, and the full `ResultViewState`.
 *
 * The project id in particular is the fix for the most damaging bug in the old
 * client — it lived only in React state, so a browser refresh silently threw
 * away the parts, the settings and the results of whatever the user was
 * working on.
 */
import { defaultResultViewState, type ResultViewState, type ViewportTool } from './viewState';
import { deepMerge } from './defaults';

/**
 * The single-axis section shape the legacy renderer takes.
 *
 * Kept for the viewport adapter (`src/viewport/ResultsViewport.tsx`), which
 * flattens the shell's three-axis `SectionState` down to this. The shell
 * itself works in `viewState.SectionState`; do not add fields here.
 */
export type SectionState = {
  enabled: boolean;
  mode: 'single' | 'slab';
  axis: 'x' | 'y' | 'z';
  position: number;
  thickness: number;
  invert: boolean;
};

export type PanelId =
  | 'parts'
  | 'material'
  | 'mesh'
  | 'analysis'
  | 'performance'
  | 'solidification'
  | 'projects';

export type ShellPrefs = {
  /** Which docked setup panel is open, or null for none. */
  activePanel: PanelId | null;
  resultsCollapsed: boolean;
  /** Orthographic projection. The desktop defaults to orthographic. */
  ortho: boolean;
  tool: ViewportTool;
  view: ResultViewState;
};

export const defaultShellPrefs: ShellPrefs = {
  activePanel: 'parts',
  resultsCollapsed: false,
  ortho: true,
  tool: 'select',
  view: defaultResultViewState
};

const SHELL_STORAGE_KEY = 'solidus.shell.v1';
const PROJECT_STORAGE_KEY = 'solidus.projectId.v1';

export function loadShellPrefs(): ShellPrefs {
  if (typeof window === 'undefined') return defaultShellPrefs;
  try {
    const raw = window.localStorage.getItem(SHELL_STORAGE_KEY);
    if (!raw) return defaultShellPrefs;
    const merged = deepMerge(defaultShellPrefs, JSON.parse(raw) as unknown);
    // Playback never survives a reload: coming back to a page that is quietly
    // animating is disorienting, and the frames are not loaded yet anyway.
    return { ...merged, view: { ...merged.view, playing: false } };
  } catch {
    return defaultShellPrefs;
  }
}

export function saveShellPrefs(prefs: ShellPrefs): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(SHELL_STORAGE_KEY, JSON.stringify(prefs));
  } catch {
    /* storage unavailable - ignore */
  }
}

export function loadStoredProjectId(): string | null {
  if (typeof window === 'undefined') return null;
  try {
    return window.localStorage.getItem(PROJECT_STORAGE_KEY);
  } catch {
    return null;
  }
}

export function saveStoredProjectId(projectId: string | null): void {
  if (typeof window === 'undefined') return;
  try {
    if (projectId) window.localStorage.setItem(PROJECT_STORAGE_KEY, projectId);
    else window.localStorage.removeItem(PROJECT_STORAGE_KEY);
  } catch {
    /* storage unavailable - ignore */
  }
}
