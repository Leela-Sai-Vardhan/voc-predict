import path from 'path';
import fs from 'fs';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Serve the sibling figures/ folder at /figures/
const figuresDir = path.resolve(__dirname, '../figures');

export default defineConfig({
  plugins: [
    react(),
    {
      name: 'serve-figures',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          if (req.url && req.url.startsWith('/figures/')) {
            const filePath = path.join(figuresDir, req.url.replace('/figures/', ''));
            if (fs.existsSync(filePath)) {
              res.setHeader('Content-Type', 'image/png');
              res.setHeader('Cache-Control', 'public, max-age=31536000');
              res.end(fs.readFileSync(filePath));
              return;
            }
          }
          next();
        });
      },
    },
  ],
  server: {
    port: 5173,
    open: true,
  },
  build: {
    outDir: 'dist',
  },
});