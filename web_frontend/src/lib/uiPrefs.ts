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
import { defaultColormapForMode } from './colormaps';
import { deepMerge } from './defaults';

export type PanelId =
  | 'parts'
  | 'model'
  | 'material'
  | 'mesh'
  | 'analysis'
  | 'performance'
  | 'solidification'
  | 'projects';

/** Foldable sections of the results panel, keyed by `CollapsibleGroup` id. */
export type ResultSectionId = 'field' | 'threshold' | 'scale' | 'section' | 'display' | 'statistics';

export type ShellPrefs = {
  /** Which docked setup panel is open, or null for none. */
  activePanel: PanelId | null;
  resultsCollapsed: boolean;
  /** Which results-panel sections are folded away. */
  sectionsCollapsed: Record<ResultSectionId, boolean>;
  /** Orthographic projection. The desktop defaults to orthographic. */
  ortho: boolean;
  tool: ViewportTool;
  /** The job log drawer at the foot of the canvas. */
  logOpen: boolean;
  view: ResultViewState;
  /**
   * Which one-time preference migrations have run against this browser's
   * stored prefs. Bumped only when a *default* changes in a way that a stored
   * value would otherwise mask forever; see `loadShellPrefs`.
   */
  prefsVersion: number;
};

/** Current migration level. See `PREFS_MIGRATIONS` in `loadShellPrefs`. */
const PREFS_VERSION = 1;

export const defaultShellPrefs: ShellPrefs = {
  activePanel: 'parts',
  resultsCollapsed: false,
  // Field and threshold are what you reach for first; scale, section and the
  // statistics table are reference material, so they start folded.
  sectionsCollapsed: {
    field: false,
    threshold: false,
    scale: true,
    section: false,
    display: false,
    statistics: true
  },
  ortho: true,
  tool: 'select',
  logOpen: false,
  view: { ...defaultResultViewState, colormap: defaultColormapForMode(defaultResultViewState.mode) },
  prefsVersion: PREFS_VERSION
};

const SHELL_STORAGE_KEY = 'solidus.shell.v1';
const PROJECT_STORAGE_KEY = 'solidus.projectId.v1';

export function loadShellPrefs(): ShellPrefs {
  if (typeof window === 'undefined') return defaultShellPrefs;
  try {
    const raw = window.localStorage.getItem(SHELL_STORAGE_KEY);
    if (!raw) return defaultShellPrefs;
    const stored = JSON.parse(raw) as unknown;
    // Read the stored level *before* merging: the merge fills the field in
    // from the defaults, which would make every old blob look current.
    const storedVersion =
      typeof stored === 'object' && stored !== null && typeof (stored as ShellPrefs).prefsVersion === 'number'
        ? (stored as ShellPrefs).prefsVersion
        : 0;
    const merged = deepMerge(defaultShellPrefs, stored);
    // Playback never survives a reload: coming back to a page that is quietly
    // animating is disorienting, and the frames are not loaded yet anyway.
    const view = { ...merged.view, playing: false };

    // Migration 1: the time modes gained their own default colormap. Anyone
    // who used the app before that has `rainbow` stored against them, and a
    // stored value beats a default forever -- so the fix would never reach the
    // people who reported it. `rainbow` was the only value those modes could
    // have had, so it cannot stand for a deliberate preference *over* molten;
    // every other stored map is left exactly as it is.
    if (storedVersion < 1 && view.colormap === 'rainbow') {
      view.colormap = defaultColormapForMode(view.mode);
    }

    return { ...merged, view, prefsVersion: PREFS_VERSION };
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
