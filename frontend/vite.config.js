import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 8081,
    allowedHosts: [
      'd3pka9yj6j75yn.cloudfront.net',
      'localhost',
      '127.0.0.1',
      '172.31.16.189'
    ]
  },
  build: {
    rollupOptions: {
      input: {
        main: './index.html',
        enhanced: './enhanced_multi_route.html'
      }
    }
  }
})