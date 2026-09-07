import { useState } from 'react';

import { Modal } from './Modal';

export type ConfirmRequest = {
  title: string;
  message: string;
  /** Label for the confirming action. Defaults to "Confirm". */
  confirmLabel?: string;
  /** Red-styles the confirming action for a destructive operation. */
  destructive?: boolean;
  onConfirm: () => void;
};

export function ConfirmDialog({ request, onClose }: { request: ConfirmRequest; onClose: () => void }) {
  return (
    <Modal
      title={request.title}
      size="sm"
      onClose={onClose}
      footer={
        <>
          <div className="spacer" />
          <button type="button" className="btn" onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            className={request.destructive ? 'btn btn-danger' : 'btn btn-primary'}
            onClick={() => {
              request.onConfirm();
              onClose();
            }}
          >
            {request.confirmLabel ?? 'Confirm'}
          </button>
        </>
      }
    >
      <p className="text-dim" style={{ lineHeight: 1.6 }}>
        {request.message}
      </p>
    </Modal>
  );
}

export type PromptRequest = {
  title: string;
  label: string;
  initialValue: string;
  confirmLabel?: string;
  hint?: string;
  onSubmit: (value: string) => void;
};

/**
 * Replaces the browser `prompt()` the desktop parity features would otherwise
 * need (arrange spacing, rename, save as). A native prompt is blocking, cannot
 * be styled, and is suppressed outright in some browsers.
 */
export function PromptDialog({ request, onClose }: { request: PromptRequest; onClose: () => void }) {
  const [value, setValue] = useState(request.initialValue);
  const submit = () => {
    const trimmed = value.trim();
    if (!trimmed) return;
    request.onSubmit(trimmed);
    onClose();
  };
  return (
    <Modal
      title={request.title}
      size="sm"
      onClose={onClose}
      footer={
        <>
          <div className="spacer" />
          <button type="button" className="btn" onClick={onClose}>
            Cancel
          </button>
          <button type="button" className="btn btn-primary" onClick={submit} disabled={!value.trim()}>
            {request.confirmLabel ?? 'OK'}
          </button>
        </>
      }
    >
      <label className="field">
        <span className="field-label">{request.label}</span>
        <input
          className="input"
          value={value}
          autoFocus
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') submit();
          }}
        />
        {request.hint ? <span className="field-hint">{request.hint}</span> : null}
      </label>
    </Modal>
  );
}
