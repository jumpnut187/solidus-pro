import { useMemo, useState } from 'react';
import { Copy, FolderOpen, Pencil, RefreshCw, Trash2 } from 'lucide-react';

import { Modal } from './Modal';
import { StatusChip } from '../panels/ProjectsPanel';
import { formatDateTime, formatRelativeTime } from '../../lib/format';
import type { ProjectSummary } from '../../lib/api';

export type OpenProjectModalProps = {
  projects: ProjectSummary[];
  currentProjectId: string | null;
  loading: boolean;
  onOpen: (projectId: string) => void;
  onRename: (project: ProjectSummary) => void;
  onDuplicate: (project: ProjectSummary) => void;
  onDelete: (project: ProjectSummary) => void;
  onRefresh: () => void;
  onClose: () => void;
};

export function OpenProjectModal({
  projects,
  currentProjectId,
  loading,
  onOpen,
  onRename,
  onDuplicate,
  onDelete,
  onRefresh,
  onClose
}: OpenProjectModalProps) {
  const [query, setQuery] = useState('');

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return projects;
    return projects.filter(
      (project) =>
        project.name.toLowerCase().includes(needle) || project.alloy.toLowerCase().includes(needle)
    );
  }, [projects, query]);

  return (
    <Modal
      title="Open Project"
      subtitle={`${projects.length} project${projects.length === 1 ? '' : 's'} on this server`}
      size="lg"
      onClose={onClose}
      flush
      toolbar={
        <div className="row" style={{ padding: 'var(--sp-4) var(--sp-5)' }}>
          <input
            className="input"
            placeholder="Filter by name or alloy"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            aria-label="Filter projects"
          />
          <button type="button" className="btn btn-icon" onClick={onRefresh} aria-label="Refresh list">
            <RefreshCw size={13} />
          </button>
        </div>
      }
      footer={
        <>
          <div className="spacer" />
          <button type="button" className="btn" onClick={onClose}>
            Close
          </button>
        </>
      }
    >
      {loading && projects.length === 0 ? (
        <p className="status-note" style={{ margin: 'var(--sp-5)' }}>
          Loading projects...
        </p>
      ) : filtered.length === 0 ? (
        <p className="status-note" style={{ margin: 'var(--sp-5)' }}>
          {projects.length === 0
            ? 'No projects yet. Close this dialog, add parts and run a simulation.'
            : 'No project matches that filter.'}
        </p>
      ) : (
        <div className="project-table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th>Name</th>
                <th className="cell-tight">Alloy</th>
                <th className="cell-tight">Status</th>
                <th className="cell-tight">Updated</th>
                <th className="cell-tight" aria-label="Actions" />
              </tr>
            </thead>
            <tbody>
              {filtered.map((project) => (
                <tr key={project.id} aria-selected={project.id === currentProjectId}>
                  <td>
                    <span className="project-name-cell">
                      <strong>{project.name}</strong>
                      <span className="text-faint mono" style={{ fontSize: 'var(--fs-xs)' }}>
                        {project.id}
                      </span>
                    </span>
                  </td>
                  <td className="cell-tight">{project.alloy}</td>
                  <td className="cell-tight">
                    <StatusChip status={project.status} />
                  </td>
                  <td className="cell-tight" title={formatDateTime(project.updated_at)}>
                    {formatRelativeTime(project.updated_at)}
                  </td>
                  <td className="cell-tight">
                    <span className="project-actions">
                      <button
                        type="button"
                        className="btn btn-sm"
                        onClick={() => onOpen(project.id)}
                        disabled={project.id === currentProjectId}
                      >
                        <FolderOpen size={12} />
                        {project.id === currentProjectId ? 'Open' : 'Open'}
                      </button>
                      <button
                        type="button"
                        className="btn btn-ghost btn-icon"
                        onClick={() => onRename(project)}
                        aria-label={`Rename ${project.name}`}
                        title="Rename"
                      >
                        <Pencil size={13} />
                      </button>
                      <button
                        type="button"
                        className="btn btn-ghost btn-icon"
                        onClick={() => onDuplicate(project)}
                        aria-label={`Duplicate ${project.name}`}
                        title="Duplicate (parts and settings, not results)"
                      >
                        <Copy size={13} />
                      </button>
                      <button
                        type="button"
                        className="btn btn-ghost btn-icon"
                        onClick={() => onDelete(project)}
                        aria-label={`Delete ${project.name}`}
                        title="Delete"
                      >
                        <Trash2 size={13} />
                      </button>
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Modal>
  );
}
