import { useEffect, useRef, useState, type ReactNode } from 'react';
import { ChevronDown } from 'lucide-react';

export type MenuItemSpec = {
  label: string;
  onSelect?: () => void;
  shortcut?: string;
  disabled?: boolean;
  /** Renders a section heading instead of a clickable row. */
  heading?: boolean;
};

/**
 * A dropdown that closes on Escape, on outside click, and after a selection.
 *
 * Menus are plain buttons in a popover rather than a `<select>`: the top bar's
 * Export and Help menus mix actions with headings, which a select cannot do.
 */
export function Menu({
  label,
  icon,
  items,
  align = 'right'
}: {
  label: string;
  icon?: ReactNode;
  items: MenuItemSpec[];
  align?: 'left' | 'right';
}) {
  const [open, setOpen] = useState(false);
  const anchorRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) return;
    function onPointerDown(event: MouseEvent) {
      if (!anchorRef.current?.contains(event.target as Node)) setOpen(false);
    }
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        event.stopPropagation();
        setOpen(false);
      }
    }
    document.addEventListener('mousedown', onPointerDown);
    document.addEventListener('keydown', onKeyDown, true);
    return () => {
      document.removeEventListener('mousedown', onPointerDown);
      document.removeEventListener('keydown', onKeyDown, true);
    };
  }, [open]);

  return (
    <div className="menu-anchor" ref={anchorRef}>
      <button
        type="button"
        className="btn btn-ghost btn-sm"
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        {icon}
        {label}
        <ChevronDown size={12} />
      </button>
      {open ? (
        <div className="menu-pop" role="menu" style={align === 'left' ? { left: 0, right: 'auto' } : undefined}>
          {items.map((item, index) =>
            item.heading ? (
              <p className="menu-label" key={`${item.label}-${index}`}>
                {item.label}
              </p>
            ) : (
              <button
                type="button"
                role="menuitem"
                className="menu-item"
                key={`${item.label}-${index}`}
                disabled={item.disabled}
                onClick={() => {
                  setOpen(false);
                  item.onSelect?.();
                }}
              >
                <span>{item.label}</span>
                {item.shortcut ? <span className="menu-item-key">{item.shortcut}</span> : null}
              </button>
            )
          )}
        </div>
      ) : null}
    </div>
  );
}
