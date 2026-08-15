import Phaser from 'phaser';
import './styles/main.css';

import BootScene from './scenes/BootScene.js';
import MenuScene from './scenes/MenuScene.js';
import LevelSelectScene from './scenes/LevelSelectScene.js';
import DinoSelectScene from './scenes/DinoSelectScene.js';
import GameScene from './scenes/GameScene.js';
import MinigameScene from './scenes/MinigameScene.js';
import { GRAVITATION } from './game/levelGenerator.js';
import { audioFreischalten } from './audio/sfx.js';

const spiel = new Phaser.Game({
  type: Phaser.AUTO,
  parent: 'game',
  backgroundColor: '#0d1b2a',
  scale: {
    mode: Phaser.Scale.FIT,
    autoCenter: Phaser.Scale.CENTER_BOTH,
    width: 1280,
    height: 720,
  },
  render: {
    antialias: true,
    roundPixels: false,
  },
  fps: {
    target: 60,
    // Kein Delta-Glaetten: Auf langsamen Geraeten wird das Spiel sonst
    // zur Zeitlupe, weil Phaser die echte Bildzeit "wegglaettet".
    smoothStep: false,
    // Bei weniger als 10 Bildern/s wird der Schritt begrenzt, damit
    // niemand durch duenne Plattformen hindurchrutscht.
    min: 10,
  },
  physics: {
    default: 'arcade',
    arcade: {
      gravity: { y: GRAVITATION },
      debug: false,
      // Zeitbasiert statt fester Schrittweite: Auf langsameren Tablets laeuft
      // das Spiel dann nicht in Zeitlupe, sondern nur mit weniger Bildern.
      fixedStep: false,
    },
  },
  scene: [BootScene, MenuScene, LevelSelectScene, DinoSelectScene, GameScene, MinigameScene],
});

// Zugriff fuer die Browser-Konsole (Debugging, z. B. window.__spiel.scene)
window.__spiel = spiel;

// Kontextmenue und Doppeltipp-Zoom auf Tablets unterbinden
window.addEventListener('contextmenu', (e) => e.preventDefault());
document.addEventListener('gesturestart', (e) => e.preventDefault());

// Audio darf erst nach einer Nutzergeste starten (Browser-Vorgabe)
const freischalten = () => {
  audioFreischalten();
  window.removeEventListener('pointerdown', freischalten);
  window.removeEventListener('keydown', freischalten);
};
window.addEventListener('pointerdown', freischalten);
window.addEventListener('keydown', freischalten);

// Service Worker fuer den Offline-Betrieb registrieren
if (import.meta.env.PROD) {
  import('virtual:pwa-register')
    .then(({ registerSW }) => {
      registerSW({ immediate: true });
    })
    .catch(() => {
      /* PWA ist optional - das Spiel laeuft auch ohne Service Worker. */
    });
}

export default spiel;
