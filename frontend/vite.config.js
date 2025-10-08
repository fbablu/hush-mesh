import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 3000,
    allowedHosts: [
      'd3pka9yj6j75yn.cloudfront.net',
      'localhost',
      '127.0.0.1',
      '172.31.16.189'
    ]
  }
})