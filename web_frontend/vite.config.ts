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
    sourcemap: true
  }
});

