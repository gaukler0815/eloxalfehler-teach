/**
 * Parallax-Kulisse: Himmel + drei Ebenen, die unterschiedlich schnell
 * mitscrollen. Die Ebenen haengen an der Kamera (scrollFactor 0); ihre
 * Unterkante wird jedes Bild an die Bodenlinie der Welt gesetzt, damit der
 * Horizont immer dort sitzt, wo der Boden anfaengt - egal wie hoch oder
 * tief die Kamera gerade steht.
 */

import { BODEN_Y } from '../game/levelGenerator.js';

export function kulisseAufbauen(scene, biomId) {
  const cam = scene.cameras.main;
  const b = cam.width;
  const h = cam.height;

  const himmel = scene.add
    .image(0, 0, `himmel_${biomId}`)
    .setOrigin(0, 0)
    .setDisplaySize(b, h)
    .setScrollFactor(0)
    .setDepth(-100);

  const ebene = (key, hoehe, tiefe, alpha) =>
    scene.add
      .tileSprite(0, 0, b, hoehe, key)
      .setOrigin(0, 0)
      .setScrollFactor(0)
      .setDepth(tiefe)
      .setAlpha(alpha);

  // Je naeher am Spielfeld, desto blasser: Der Hintergrund soll ruhig bleiben,
  // damit sich Dino, Gegner und Sammelsachen klar davon abheben.
  const fern = ebene(`fern_${biomId}`, 220, -90, 0.75);
  const mittel = ebene(`mittel_${biomId}`, 200, -80, 0.62);
  const nah = ebene(`nah_${biomId}`, 150, -70, 0.45);

  return {
    himmel,
    fern,
    mittel,
    nah,
    aktualisieren() {
      const x = cam.scrollX;
      fern.tilePositionX = x * 0.12;
      mittel.tilePositionX = x * 0.32;
      nah.tilePositionX = x * 0.6;

      // Wo liegt die Bodenlinie gerade auf dem Bildschirm? Die Ebenen haengen
      // daran, ragen aber nur ein Stueck darueber hinaus - so bleibt der
      // Laufbereich frei und der Dino hebt sich vom Hintergrund ab.
      const bodenAufSchirm = BODEN_Y - cam.scrollY;
      fern.y = bodenAufSchirm - 260;
      mittel.y = bodenAufSchirm - 180;
      nah.y = bodenAufSchirm - 70;
    },
    zerstoeren() {
      [himmel, fern, mittel, nah].forEach((o) => o.destroy());
    },
  };
}
