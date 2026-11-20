import { defaultResultViewState, type ResultViewState, type ViewportTool } from './viewState';
import { defaultColormapForMode } from './colormaps';
import { deepMerge } from './defaults';

export type PanelId =
  | 'parts'
  | 'orient'
  | 'model'
  | 'material'
  | 'mesh'
  | 'analysis'
  | 'performance'
  | 'solidification'
  | 'mold'
  | 'projects';

export type ResultSectionId = 'field' | 'threshold' | 'scale' | 'section' | 'display' | 'statistics';

export type ShellPrefs = {
  activePanel: PanelId | null;
  resultsCollapsed: boolean;
  sectionsCollapsed: Record<ResultSectionId, boolean>;
  ortho: boolean;
  tool: ViewportTool;
  logOpen: boolean;
  view: ResultViewState;
  prefsVersion: number;
};

const PREFS_VERSION = 1;

export const defaultShellPrefs: ShellPrefs = {
  activePanel: 'parts',
  resultsCollapsed: false,
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
    const storedVersion =
      typeof stored === 'object' && stored !== null && typeof (stored as ShellPrefs).prefsVersion === 'number'
        ? (stored as ShellPrefs).prefsVersion
        : 0;
    const merged = deepMerge(defaultShellPrefs, stored);
    const view = { ...merged.view, playing: false };

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
  }
}
