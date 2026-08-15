/**
 * Erzeugt die PWA-Icons als echte PNG-Dateien - ohne externe Pakete.
 * Aufruf:  npm run icons
 *
 * Die Dateien liegen anschliessend in public/icons/ und werden vom
 * Manifest (vite.config.js) referenziert.
 */

import { deflateSync } from 'node:zlib';
import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HIER = dirname(fileURLToPath(import.meta.url));
const ZIEL = join(HIER, '..', 'public', 'icons');

// ------------------------------------------------------------ PNG schreiben

const CRC_TABELLE = (() => {
  const t = new Int32Array(256);
  for (let n = 0; n < 256; n += 1) {
    let c = n;
    for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c;
  }
  return t;
})();

function crc32(buf) {
  let c = -1;
  for (let i = 0; i < buf.length; i += 1) c = CRC_TABELLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ -1) >>> 0;
}

function chunk(typ, daten) {
  const laenge = Buffer.alloc(4);
  laenge.writeUInt32BE(daten.length, 0);
  const inhalt = Buffer.concat([Buffer.from(typ, 'ascii'), daten]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(inhalt), 0);
  return Buffer.concat([laenge, inhalt, crc]);
}

function pngSchreiben(pfad, breite, hoehe, rgba) {
  const roh = Buffer.alloc((breite * 4 + 1) * hoehe);
  for (let y = 0; y < hoehe; y += 1) {
    roh[y * (breite * 4 + 1)] = 0; // Filter "None"
    rgba.copy(roh, y * (breite * 4 + 1) + 1, y * breite * 4, (y + 1) * breite * 4);
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(breite, 0);
  ihdr.writeUInt32BE(hoehe, 4);
  ihdr[8] = 8; // Bittiefe
  ihdr[9] = 6; // RGBA
  const datei = Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(roh, { level: 9 })),
    chunk('IEND', Buffer.alloc(0)),
  ]);
  writeFileSync(pfad, datei);
  return datei.length;
}

// ------------------------------------------------------------- Zeichnen

class Bild {
  constructor(groesse) {
    this.g = groesse;
    this.daten = Buffer.alloc(groesse * groesse * 4, 0);
  }

  punkt(x, y, [r, gr, b], alpha = 1) {
    if (x < 0 || y < 0 || x >= this.g || y >= this.g || alpha <= 0) return;
    const i = (y * this.g + x) * 4;
    const a0 = this.daten[i + 3] / 255;
    const a = alpha + a0 * (1 - alpha);
    this.daten[i] = Math.round((r * alpha + this.daten[i] * a0 * (1 - alpha)) / a);
    this.daten[i + 1] = Math.round((gr * alpha + this.daten[i + 1] * a0 * (1 - alpha)) / a);
    this.daten[i + 2] = Math.round((b * alpha + this.daten[i + 2] * a0 * (1 - alpha)) / a);
    this.daten[i + 3] = Math.round(a * 255);
  }

  /** Abgerundetes Quadrat mit senkrechtem Farbverlauf. */
  hintergrund(von, bis, radius) {
    const g = this.g;
    for (let y = 0; y < g; y += 1) {
      const t = y / (g - 1);
      const farbe = von.map((c, i) => Math.round(c + (bis[i] - c) * t));
      for (let x = 0; x < g; x += 1) {
        const dx = Math.max(radius - x, x - (g - 1 - radius), 0);
        const dy = Math.max(radius - y, y - (g - 1 - radius), 0);
        const d = Math.hypot(dx, dy);
        const alpha = d <= radius ? 1 : Math.max(0, 1 - (d - radius));
        this.punkt(x, y, farbe, alpha);
      }
    }
  }

  ellipse(cx, cy, rx, ry, farbe, alpha = 1) {
    for (let y = Math.floor(cy - ry - 1); y <= Math.ceil(cy + ry + 1); y += 1) {
      for (let x = Math.floor(cx - rx - 1); x <= Math.ceil(cx + rx + 1); x += 1) {
        const d = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2;
        if (d <= 1) this.punkt(x, y, farbe, alpha);
        else if (d <= 1.06) this.punkt(x, y, farbe, alpha * (1 - (d - 1) / 0.06));
      }
    }
  }

  linie(x1, y1, x2, y2, dicke, farbe) {
    const schritte = Math.ceil(Math.hypot(x2 - x1, y2 - y1) * 2);
    for (let i = 0; i <= schritte; i += 1) {
      const t = i / schritte;
      this.ellipse(x1 + (x2 - x1) * t, y1 + (y2 - y1) * t, dicke, dicke, farbe);
    }
  }
}

/** Dino-Ei-Motiv. `rand` = zusaetzlicher Sicherheitsrand (maskable). */
function motiv(groesse, { maskable = false } = {}) {
  const bild = new Bild(groesse);
  const s = groesse / 512;
  bild.hintergrund([27, 127, 75], [13, 60, 42], maskable ? 0 : Math.round(96 * s));

  const skala = maskable ? 0.78 : 1;
  const cx = groesse / 2;
  const cy = groesse / 2 + 10 * s;

  // Ei
  bild.ellipse(cx, cy, 150 * s * skala, 185 * s * skala, [255, 246, 221]);
  bild.ellipse(cx - 44 * s * skala, cy - 78 * s * skala, 42 * s * skala, 46 * s * skala, [
    255, 255, 255,
  ], 0.5);

  // Flecken
  const flecken = [
    [-58, -20, 34],
    [46, 22, 40],
    [-16, 82, 30],
    [58, -70, 24],
  ];
  flecken.forEach(([fx, fy, fr]) => {
    bild.ellipse(
      cx + fx * s * skala,
      cy + fy * s * skala,
      fr * s * skala,
      fr * 0.9 * s * skala,
      [255, 179, 63]
    );
  });

  // Zickzack-Sprung im Ei
  const zacken = [
    [-120, -12],
    [-60, 18],
    [-14, -22],
    [36, 20],
    [86, -14],
    [126, 10],
  ];
  for (let i = 0; i < zacken.length - 1; i += 1) {
    bild.linie(
      cx + zacken[i][0] * s * skala,
      cy + zacken[i][1] * s * skala,
      cx + zacken[i + 1][0] * s * skala,
      cy + zacken[i + 1][1] * s * skala,
      5 * s * skala,
      [140, 90, 40]
    );
  }
  return bild;
}

// ---------------------------------------------------------------- Ausgabe

mkdirSync(ZIEL, { recursive: true });

const dateien = [
  ['icon-192.png', 192, {}],
  ['icon-512.png', 512, {}],
  ['icon-maskable-512.png', 512, { maskable: true }],
  ['apple-touch-icon.png', 180, {}],
];

dateien.forEach(([name, groesse, opt]) => {
  const bild = motiv(groesse, opt);
  const bytes = pngSchreiben(join(ZIEL, name), groesse, groesse, bild.daten);
  process.stdout.write(`${name.padEnd(26)} ${groesse}x${groesse}  ${bytes} Bytes\n`);
});

const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#1b7f4b"/><stop offset="1" stop-color="#0d3c2a"/>
    </linearGradient>
  </defs>
  <rect width="512" height="512" rx="96" fill="url(#g)"/>
  <ellipse cx="256" cy="266" rx="150" ry="185" fill="#fff6dd"/>
  <ellipse cx="212" cy="188" rx="42" ry="46" fill="#ffffff" opacity=".5"/>
  <g fill="#ffb33f">
    <ellipse cx="198" cy="246" rx="34" ry="31"/>
    <ellipse cx="302" cy="288" rx="40" ry="36"/>
    <ellipse cx="240" cy="348" rx="30" ry="27"/>
    <ellipse cx="314" cy="196" rx="24" ry="22"/>
  </g>
  <polyline points="136,254 196,284 242,244 292,286 342,252 382,276"
    fill="none" stroke="#8c5a28" stroke-width="10" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
`;
writeFileSync(join(ZIEL, 'icon.svg'), svg);
writeFileSync(join(ZIEL, 'favicon.svg'), svg);
process.stdout.write('icon.svg / favicon.svg geschrieben\n');
