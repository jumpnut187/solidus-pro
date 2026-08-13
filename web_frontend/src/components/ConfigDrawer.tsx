import type { ReactNode } from 'react';
import { X } from 'lucide-react';

type Props = {
  title: string;
  icon?: ReactNode;
  onClose: () => void;
  children: ReactNode;
};

export function ConfigDrawer({ title, icon, onClose, children }: Props) {
  return (
    <div className="drawer" role="dialog" aria-label={title}>
      <header className="drawer-header">
        <div className="drawer-title">
          {icon}
          <span>{title}</span>
        </div>
        <button className="icon-button" onClick={onClose} title="Close">
          <X size={16} />
        </button>
      </header>
      <div className="drawer-body">{children}</div>
      <footer className="drawer-footer">
        <button className="ghost-button" onClick={onClose}>Cancel</button>
        <button className="confirm-button" onClick={onClose}>Confirm</button>
      </footer>
    </div>
  );
}
