import { useCallback, useEffect, useRef, useState } from 'react';
import { X } from 'lucide-react';
import type { ReactNode } from 'react';

type Props = {
  title: string;
  initialX?: number;
  initialY?: number;
  width?: number;
  height?: number;
  minWidth?: number;
  minHeight?: number;
  children: ReactNode;
  onClose: () => void;
};

export function DraggableDialog({
  title,
  initialX = 120,
  initialY = 90,
  width = 560,
  height = 520,
  minWidth = 420,
  minHeight = 320,
  children,
  onClose
}: Props) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState({ x: initialX, y: initialY });
  const [size, setSize] = useState({ width, height });
  const [isDragging, setIsDragging] = useState(false);
  const [isResizing, setIsResizing] = useState(false);
  const dragOffsetRef = useRef({ x: 0, y: 0 });
  const resizeStartRef = useRef({ width: 0, height: 0, x: 0, y: 0 });

  const handleMouseDown = useCallback((event: React.MouseEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    setIsDragging(true);
    dragOffsetRef.current = {
      x: event.clientX - position.x,
      y: event.clientY - position.y
    };
  }, [position]);

  const handleResizeMouseDown = useCallback((event: React.MouseEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    event.stopPropagation();
    setIsResizing(true);
    resizeStartRef.current = {
      width: size.width,
      height: size.height,
      x: event.clientX,
      y: event.clientY
    };
  }, [size]);

  useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (event: MouseEvent) => {
      const maxX = Math.max(0, window.innerWidth - size.width - 20);
      const maxY = Math.max(0, window.innerHeight - 100);
      setPosition({
        x: Math.max(0, Math.min(maxX, event.clientX - dragOffsetRef.current.x)),
        y: Math.max(0, Math.min(maxY, event.clientY - dragOffsetRef.current.y))
      });
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, size.width]);

  useEffect(() => {
    if (!isResizing) return;

    const handleMouseMove = (event: MouseEvent) => {
      const newWidth = Math.max(minWidth, resizeStartRef.current.width + (event.clientX - resizeStartRef.current.x));
      const newHeight = Math.max(minHeight, resizeStartRef.current.height + (event.clientY - resizeStartRef.current.y));
      const maxWidth = window.innerWidth - position.x - 20;
      const maxHeight = window.innerHeight - position.y - 20;
      setSize({
        width: Math.min(newWidth, maxWidth),
        height: Math.min(newHeight, maxHeight)
      });
    };

    const handleMouseUp = () => {
      setIsResizing(false);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizing, minWidth, minHeight, position.x, position.y]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  return (
    <div
      ref={dialogRef}
      className="draggable-dialog"
      style={{
        left: position.x,
        top: position.y,
        width: size.width,
        height: size.height,
        position: 'fixed',
        zIndex: 100
      }}
    >
      <div
        className="dialog-header"
        onMouseDown={handleMouseDown}
        role="button"
        tabIndex={0}
        aria-label="Drag dialog"
      >
        <span className="dialog-title">{title}</span>
        <button className="dialog-close" onClick={onClose} aria-label="Close dialog">
          <X size={16} />
        </button>
      </div>
      <div className="dialog-body">{children}</div>
      <div
        className="dialog-resize-handle"
        onMouseDown={handleResizeMouseDown}
        role="button"
        tabIndex={0}
        aria-label="Resize dialog"
      />
    </div>
  );
}
