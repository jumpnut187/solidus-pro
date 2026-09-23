import process from 'node:process';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': `http://127.0.0.1:${process.env.CASTSIM_API_PORT || '5050'}`
    }
  },
  build: {
    outDir: 'dist',
    // Off rather than 'hidden'. Flask serves every file under dist/ as-is
    // (see web/app_factory.py), so a hidden map is still one guessable URL
    // away and publishes the whole frontend source. `vite` dev keeps its
    // own in-memory maps for debugging.
    sourcemap: false
  }
});

