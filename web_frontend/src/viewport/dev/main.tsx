/** Mount point for the standalone viewport bench (`/dev.html`). */
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

import { Bench } from './Bench';
import './bench.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Bench />
  </StrictMode>
);
