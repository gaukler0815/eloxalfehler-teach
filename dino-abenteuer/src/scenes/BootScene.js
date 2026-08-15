import Phaser from 'phaser';
import { texturenErzeugen, animationenAnlegen } from '../gfx/textures.js';

/**
 * Erzeugt alle Texturen und Animationen und startet dann das Hauptmenue.
 * Es werden keine externen Dateien geladen - alles entsteht auf Canvas.
 */
export default class BootScene extends Phaser.Scene {
  constructor() {
    super('Boot');
  }

  create() {
    texturenErzeugen(this);
    animationenAnlegen(this);

    const splash = document.getElementById('boot-splash');
    if (splash) {
      splash.classList.add('hidden');
      setTimeout(() => splash.remove(), 500);
    }

    this.scene.start('Menu');
  }
}
