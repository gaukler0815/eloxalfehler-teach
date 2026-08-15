import { defineConfig } from 'vite';
import { VitePWA } from 'vite-plugin-pwa';

// Basis-Pfad: relativ, damit das Spiel auch in Unterordnern
// (z. B. GitHub Pages /dino-abenteuer/) funktioniert.
export default defineConfig({
  base: './',
  server: {
    host: true,
    port: 5173,
  },
  build: {
    target: 'es2020',
    chunkSizeWarningLimit: 1600,
  },
  plugins: [
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['icons/icon.svg', 'icons/apple-touch-icon.png', 'favicon.svg'],
      manifest: {
        id: '/dino-abenteuer/',
        name: 'Linneas Dino-Abenteuer',
        short_name: 'Dino-Abenteuer',
        description:
          'Ein Jump-and-Run mit 30 Leveln, 30 Dinos und Lese- sowie Mathe-Aufgaben fuer Erstleser.',
        lang: 'de',
        dir: 'ltr',
        start_url: './index.html',
        scope: './',
        display: 'standalone',
        display_override: ['standalone', 'fullscreen'],
        orientation: 'landscape',
        background_color: '#0d1b2a',
        theme_color: '#1b7f4b',
        categories: ['games', 'education', 'kids'],
        icons: [
          {
            src: 'icons/icon-192.png',
            sizes: '192x192',
            type: 'image/png',
            purpose: 'any',
          },
          {
            src: 'icons/icon-512.png',
            sizes: '512x512',
            type: 'image/png',
            purpose: 'any',
          },
          {
            src: 'icons/icon-maskable-512.png',
            sizes: '512x512',
            type: 'image/png',
            purpose: 'maskable',
          },
          {
            src: 'icons/icon.svg',
            sizes: 'any',
            type: 'image/svg+xml',
            purpose: 'any',
          },
        ],
      },
      workbox: {
        globPatterns: ['**/*.{js,css,html,svg,png,ico,woff2}'],
        // Phaser ist gross - Limit anheben, damit der Bundle offline gecacht wird.
        maximumFileSizeToCacheInBytes: 6 * 1024 * 1024,
        navigateFallback: 'index.html',
        cleanupOutdatedCaches: true,
      },
      devOptions: {
        enabled: false,
      },
    }),
  ],
});
