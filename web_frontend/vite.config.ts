import { fileURLToPath } from 'node:url';
import process from 'node:process';
import { defineConfig, searchForWorkspaceRoot } from 'vite';
import react from '@vitejs/plugin-react';

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
    sourcemap: false
  }
});
