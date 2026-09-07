import { FolderOpen, Plus, RefreshCw } from 'lucide-react';

import { Group, SetupPanel } from '../SetupPanel';
import { formatRelativeTime } from '../../lib/format';
import type { ProjectSummary } from '../../lib/api';

export type ProjectsPanelProps = {
  projects: ProjectSummary[];
  currentProjectId: string | null;
  loading: boolean;
  onOpen: (projectId: string) => void;
  onRefresh: () => void;
  onNew: () => void;
  onBrowse: () => void;
  onClose: () => void;
};

export function ProjectsPanel({
  projects,
  currentProjectId,
  loading,
  onOpen,
  onRefresh,
  onNew,
  onBrowse,
  onClose
}: ProjectsPanelProps) {
  const recent = projects.slice(0, 12);
  return (
    <SetupPanel title="Projects" onClose={onClose}>
      <div className="row">
        <button type="button" className="btn btn-sm" onClick={onNew}>
          <Plus size={12} />
          New
        </button>
        <button type="button" className="btn btn-sm" onClick={onBrowse}>
          <FolderOpen size={12} />
          Browse all
        </button>
        <div className="spacer" />
        <button type="button" className="btn btn-ghost btn-icon" onClick={onRefresh} aria-label="Refresh list">
          <RefreshCw size={13} />
        </button>
      </div>

      <Group title={`Recent (${projects.length})`}>
        {loading && projects.length === 0 ? (
          <p className="status-note">Loading projects...</p>
        ) : recent.length === 0 ? (
          <p className="status-note">
            No saved projects yet. Add parts and run a simulation, and the project appears here.
          </p>
        ) : (
          <div className="project-list">
            {recent.map((project) => (
              <button
                type="button"
                key={project.id}
                className="project-card"
                aria-current={project.id === currentProjectId}
                onClick={() => onOpen(project.id)}
              >
                <span className="project-card-body">
                  <span className="project-card-name">{project.name}</span>
                  <span className="project-card-meta">
                    {project.alloy} · {formatRelativeTime(project.updated_at)}
                  </span>
                </span>
                <StatusChip status={project.status} />
              </button>
            ))}
          </div>
        )}
      </Group>
    </SetupPanel>
  );
}

export function StatusChip({ status }: { status: string }) {
  if (status === 'completed') return <span className="chip chip-success">Results</span>;
  if (status === 'running' || status === 'queued') return <span className="chip chip-warning">{status}</span>;
  if (status === 'error' || status === 'failed') return <span className="chip">Failed</span>;
  return <span className="chip">Draft</span>;
}
