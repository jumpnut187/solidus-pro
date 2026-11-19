import { fileURLToPath } from 'node:url';
import process from 'node:process';
import { defineConfig, searchForWorkspaceRoot } from 'vite';
import react from '@vitejs/plugin-react';

// Result help text shared with the desktop app. It lives outside this project,
// so the dev server must be allowed to serve that one file.
const resultHelpJson = fileURLToPath(new URL('../core/result_help.json', import.meta.url));

export default defineConfig({
  plugins: [react()],
  server: {
    fs: {
      allow: [searchForWorkspaceRoot(process.cwd()), resultHelpJson]
    },
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

