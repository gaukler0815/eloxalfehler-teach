/**
 * Parallax-Kulisse: Himmel + drei Ebenen, die unterschiedlich schnell
 * mitscrollen. Alle Ebenen haengen an der Kamera (scrollFactor 0) und
 * werden ueber tilePositionX verschoben.
 */

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

  const fern = scene.add
    .tileSprite(0, h - 340, b, 220, `fern_${biomId}`)
    .setOrigin(0, 0)
    .setScrollFactor(0)
    .setDepth(-90)
    .setAlpha(0.9);

  const mittel = scene.add
    .tileSprite(0, h - 260, b, 200, `mittel_${biomId}`)
    .setOrigin(0, 0)
    .setScrollFactor(0)
    .setDepth(-80);

  const nah = scene.add
    .tileSprite(0, h - 150, b, 150, `nah_${biomId}`)
    .setOrigin(0, 0)
    .setScrollFactor(0)
    .setDepth(-70)
    .setAlpha(0.95);

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
      const y = cam.scrollY;
      fern.y = h - 340 - y * 0.05;
      mittel.y = h - 260 - y * 0.1;
      nah.y = h - 150 - y * 0.2;
    },
    zerstoeren() {
      [himmel, fern, mittel, nah].forEach((o) => o.destroy());
    },
  };
}
