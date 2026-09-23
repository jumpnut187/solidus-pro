import { Camera, FlameKindling, FolderOpen, Play, Plus, Save, Square } from 'lucide-react';

import { Menu } from './Menu';
import { DEFAULT_PROJECT_NAME } from '../lib/defaults';
import { shortcutFor } from '../lib/shortcuts';
import type { JobState } from '../hooks/useSimulationJob';

export type TopBarProps = {
  projectName: string;
  onProjectNameChange: (name: string) => void;
  onNewProject: () => void;
  onOpenProject: () => void;
  onSaveProject: () => void;
  onSaveProjectAs: () => void;
  onRun: () => void;
  onStop: () => void;
  canRun: boolean;
  isBusy: boolean;
  job: JobState;
  onScreenshot: () => void;
  onExport: (
    kind: 'screenshot' | 'animation' | 'report' | 'projectFile' | 'importProjectFile' | 'riserDesign'
  ) => void;
  onHelp: (topic: 'guide' | 'shortcuts' | 'hardware' | 'about') => void;
  /** Entries that need a project or results stay visible but disabled. */
  exportsAvailable: {
    animation: boolean;
    report: boolean;
    projectFile: boolean;
    riserDesign: boolean;
  };
};

export function TopBar({
  projectName,
  onProjectNameChange,
  onNewProject,
  onOpenProject,
  onSaveProject,
  onSaveProjectAs,
  onRun,
  onStop,
  canRun,
  isBusy,
  job,
  onScreenshot,
  onExport,
  onHelp,
  exportsAvailable
}: TopBarProps) {
  const showProgress = isBusy || job.phase === 'completed' || job.phase === 'error';
  return (
    <header className="topbar">
      <div className="brand">
        <span className="brand-mark" aria-hidden="true">
          <FlameKindling size={15} />
        </span>
        <span className="brand-name">Solidus Pro</span>
      </div>

      <div className="topbar-sep" aria-hidden="true" />

      <input
        className="project-name-input"
        value={projectName}
        aria-label="Project name"
        title={projectName}
        placeholder={DEFAULT_PROJECT_NAME}
        onChange={(event) => onProjectNameChange(event.target.value)}
      />

      <div className="topbar-group">
        <button type="button" className="btn btn-ghost btn-sm" onClick={onNewProject}>
          <Plus size={13} />
          New
        </button>
        <button type="button" className="btn btn-ghost btn-sm" onClick={onOpenProject}>
          <FolderOpen size={13} />
          Open
        </button>
        <button
          type="button"
          className="btn btn-ghost btn-sm"
          onClick={onSaveProject}
          title={
            `Save now (${shortcutFor('Save project')}). Settings already save themselves as you ` +
            'change them; this writes the name and settings immediately.'
          }
        >
          <Save size={13} />
          Save
        </button>
        <button
          type="button"
          className="btn btn-ghost btn-sm"
          onClick={onSaveProjectAs}
          title={
            `Save a copy under a new name and switch to it (${shortcutFor('Save project as')}). ` +
            'Parts and settings are copied; results stay with the original.'
          }
        >
          Save As
        </button>
      </div>

      <div className="topbar-sep" aria-hidden="true" />

      <div className="topbar-group">
        <button
          type="button"
          className="btn btn-run btn-sm"
          onClick={onRun}
          disabled={!canRun || isBusy}
          title={`Run simulation (${shortcutFor('Run simulation')})`}
        >
          <Play size={13} />
          Run
        </button>
        <button
          type="button"
          className="btn btn-sm"
          onClick={onStop}
          disabled={!isBusy}
          title={`Stop simulation (${shortcutFor('Stop simulation')})`}
        >
          <Square size={12} />
          Stop
        </button>
      </div>

      <div className="job-status">
        {showProgress ? (
          <div
            className="job-bar"
            role="progressbar"
            aria-valuenow={Math.round(job.progress)}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label="Simulation progress"
          >
            <div className="job-bar-fill" style={{ width: `${Math.min(100, Math.max(0, job.progress))}%` }} />
          </div>
        ) : null}
        <span className={job.phase === 'error' ? 'job-text is-error' : 'job-text'} title={job.message}>
          {job.message}
        </span>
      </div>

      <div className="topbar-group">
        <button type="button" className="btn btn-ghost btn-sm" onClick={onScreenshot} title={`Export screenshot (${shortcutFor('Export screenshot')})`}>
          <Camera size={13} />
          Screenshot
        </button>
        <Menu
          label="Tools"
          items={[
            { label: 'Design', heading: true },
            {
              label: 'Automatic Riser Design',
              onSelect: () => onExport('riserDesign'),
              disabled: !exportsAvailable.riserDesign
            },
            { label: 'Export', heading: true },
            { label: 'Screenshot (PNG)', onSelect: () => onExport('screenshot'), shortcut: shortcutFor('Export screenshot') },
            {
              label: 'Animation (WebM)',
              onSelect: () => onExport('animation'),
              disabled: !exportsAvailable.animation
            },
            { label: 'Report...', onSelect: () => onExport('report'), disabled: !exportsAvailable.report },
            {
              label: 'Project File',
              onSelect: () => onExport('projectFile'),
              disabled: !exportsAvailable.projectFile
            },
            { label: 'Import', heading: true },
            { label: 'Project File...', onSelect: () => onExport('importProjectFile') }
          ]}
        />
        <Menu
          label="Help"
          items={[
            { label: 'Help Guide', onSelect: () => onHelp('guide'), shortcut: shortcutFor('Help guide') },
            { label: 'Keyboard Shortcuts', onSelect: () => onHelp('shortcuts'), shortcut: shortcutFor('This cheat sheet') },
            { label: 'Hardware Info', onSelect: () => onHelp('hardware') },
            { label: 'About Solidus Pro', onSelect: () => onHelp('about') }
          ]}
        />
      </div>
    </header>
  );
}
