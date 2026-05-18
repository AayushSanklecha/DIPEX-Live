import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Proxy API requests to the FastAPI backend in dev mode.
    // This avoids CORS issues when the frontend runs on :5173 and
    // the backend runs on :8000.
    proxy: {
      '/api': {
        target: 'http://dipex-api:8000',
        changeOrigin: true,
      },
      '/analyst': {
        target: 'http://dipex-api:8000',
        changeOrigin: true,
      },
      '/metrics': {
        target: 'http://dipex-api:8000',
        changeOrigin: true,
      },
      '/ingest': {
        target: 'http://dipex-api:8000',
        changeOrigin: true,
      },
      '/preprocess': {
        target: 'http://dipex-api:8000',
        changeOrigin: true,
      },
      '/stats': {
        target: 'http://dipex-api:8000',
        changeOrigin: true,
      },
      '^/report/': {
        target: 'http://dipex-api:8000',
        changeOrigin: true,
      },
      '/cohort': {
        target: 'http://dipex-api:8000',
        changeOrigin: true,
      },
      '/audit': {
        target: 'http://dipex-api:8000',
        changeOrigin: true,
      },
    },
  },
})

