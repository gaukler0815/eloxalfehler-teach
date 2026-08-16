import './styles/main.css';
import * as screens from './ui/screens.js';
import { start } from './ui/screens.js';
import { audioFreischalten } from './audio/sfx.js';
import { LEVEL, WELTEN } from './data/welten.js';
import Spielstand from './state/storage.js';

start();

// Zugriff fuer die Browser-Konsole (Debugging und automatische Tests)
window.__quiz = { screens, LEVEL, WELTEN, Spielstand };

// Audio darf erst nach einer Nutzergeste starten (Browser-Vorgabe)
const freischalten = () => {
  audioFreischalten();
  window.removeEventListener('pointerdown', freischalten);
  window.removeEventListener('keydown', freischalten);
};
window.addEventListener('pointerdown', freischalten);
window.addEventListener('keydown', freischalten);

// Doppeltipp-Zoom auf Tablets unterbinden
document.addEventListener('gesturestart', (e) => e.preventDefault());

// Service Worker fuer den Offline-Betrieb
if (import.meta.env.PROD) {
  import('virtual:pwa-register')
    .then(({ registerSW }) => registerSW({ immediate: true }))
    .catch(() => {
      /* ohne Service Worker laeuft die App genauso, nur nicht offline */
    });
}
