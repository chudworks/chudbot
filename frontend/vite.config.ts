import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Vite dev server config.
//
// Backend runs on 127.0.0.1:1860 (set in config.toml's [web].listen).
// Vite proxies /api/* and the media routes to the backend so we don't
// need CORS handling during development. /api/.../events is an SSE
// stream — Vite's default proxy handles streaming correctly when
// `ws: false` (the default).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:1860',
        changeOrigin: false,
      },
      '/images': 'http://127.0.0.1:1860',
      '/videos': 'http://127.0.0.1:1860',
      '/avatars': 'http://127.0.0.1:1860',
    },
  },
  build: {
    // Vite's default is fine. We don't tweak outDir here — serve.sh
    // copies dist/ to $CHUDBOT_DIR/frontend-build/ explicitly.
    sourcemap: false,
  },
});
