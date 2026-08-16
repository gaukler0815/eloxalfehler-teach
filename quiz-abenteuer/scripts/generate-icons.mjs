/**
 * Erzeugt die PWA-Icons als echte PNG-Dateien - ohne externe Pakete.
 * Aufruf:  npm run icons
 *
 * Motiv: ein leuchtender Stern (fuer Sterne sammeln) auf Nachthimmel,
 * dazu ein paar kleine Sterne - passend zu Quiz und Weltraum.
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
    roh[y * (breite * 4 + 1)] = 0;
    rgba.copy(roh, y * (breite * 4 + 1) + 1, y * breite * 4, (y + 1) * breite * 4);
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(breite, 0);
  ihdr.writeUInt32BE(hoehe, 4);
  ihdr[8] = 8;
  ihdr[9] = 6;
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
        else if (d <= 1.08) this.punkt(x, y, farbe, alpha * (1 - (d - 1) / 0.08));
      }
    }
  }

  /** Fuellt ein beliebiges Polygon (Strahlenverfahren, 2x2 geglaettet). */
  polygon(punkte, farbe, alpha = 1) {
    const xs = punkte.map((p) => p[0]);
    const ys = punkte.map((p) => p[1]);
    const x0 = Math.max(0, Math.floor(Math.min(...xs)));
    const x1 = Math.min(this.g - 1, Math.ceil(Math.max(...xs)));
    const y0 = Math.max(0, Math.floor(Math.min(...ys)));
    const y1 = Math.min(this.g - 1, Math.ceil(Math.max(...ys)));

    const drin = (px, py) => {
      let treffer = false;
      for (let i = 0, j = punkte.length - 1; i < punkte.length; j = i, i += 1) {
        const [xi, yi] = punkte[i];
        const [xj, yj] = punkte[j];
        if (yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) treffer = !treffer;
      }
      return treffer;
    };

    for (let y = y0; y <= y1; y += 1) {
      for (let x = x0; x <= x1; x += 1) {
        let treffer = 0;
        for (const oy of [0.25, 0.75]) {
          for (const ox of [0.25, 0.75]) if (drin(x + ox, y + oy)) treffer += 1;
        }
        if (treffer) this.punkt(x, y, farbe, alpha * (treffer / 4));
      }
    }
  }

  stern(cx, cy, aussen, innen, farbe, alpha = 1, zacken = 5) {
    const punkte = [];
    for (let i = 0; i < zacken * 2; i += 1) {
      const r = i % 2 === 0 ? aussen : innen;
      const w = (Math.PI / zacken) * i - Math.PI / 2;
      punkte.push([cx + Math.cos(w) * r, cy + Math.sin(w) * r]);
    }
    this.polygon(punkte, farbe, alpha);
  }
}

function motiv(groesse, { maskable = false } = {}) {
  const bild = new Bild(groesse);
  const s = groesse / 512;
  bild.hintergrund([47, 60, 120], [24, 28, 62], maskable ? 0 : Math.round(96 * s));

  const skala = maskable ? 0.76 : 1;
  const cx = groesse / 2;
  const cy = groesse / 2;

  // Kleine Sterne im Hintergrund
  [
    [-160, -150, 16],
    [150, -140, 22],
    [-150, 150, 20],
    [160, 145, 14],
  ].forEach(([dx, dy, r]) => {
    bild.stern(cx + dx * s * skala, cy + dy * s * skala, r * s * skala, r * 0.42 * s * skala, [
      255, 255, 255,
    ], 0.8);
  });

  // Sanfter Schein hinter dem grossen Stern
  bild.ellipse(cx, cy, 150 * s * skala, 150 * s * skala, [255, 210, 63], 0.16);
  bild.ellipse(cx, cy, 110 * s * skala, 110 * s * skala, [255, 210, 63], 0.16);

  // Grosser Stern
  bild.stern(cx, cy, 160 * s * skala, 66 * s * skala, [255, 190, 30]);
  bild.stern(cx, cy - 6 * s * skala, 128 * s * skala, 52 * s * skala, [255, 224, 110]);
  return bild;
}

// ---------------------------------------------------------------- Ausgabe

mkdirSync(ZIEL, { recursive: true });

[
  ['icon-192.png', 192, {}],
  ['icon-512.png', 512, {}],
  ['icon-maskable-512.png', 512, { maskable: true }],
  ['apple-touch-icon.png', 180, {}],
].forEach(([name, groesse, opt]) => {
  const bild = motiv(groesse, opt);
  const bytes = pngSchreiben(join(ZIEL, name), groesse, groesse, bild.daten);
  process.stdout.write(`${name.padEnd(26)} ${groesse}x${groesse}  ${bytes} Bytes\n`);
});

const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#2f3c78"/><stop offset="1" stop-color="#181c3e"/>
    </linearGradient>
  </defs>
  <rect width="512" height="512" rx="96" fill="url(#g)"/>
  <g fill="#ffffff" opacity=".8">
    <path d="M96 106 100 118 112 122 100 126 96 138 92 126 80 122 92 118Z"/>
    <path d="M406 116 412 132 428 138 412 144 406 160 400 144 384 138 400 132Z"/>
    <path d="M106 406 111 420 125 425 111 430 106 444 101 430 87 425 101 420Z"/>
    <path d="M416 401 420 411 430 415 420 419 416 429 412 419 402 415 412 411Z"/>
  </g>
  <circle cx="256" cy="256" r="150" fill="#ffd23f" opacity=".16"/>
  <path d="M256 96 292 210 412 210 315 281 352 395 256 324 160 395 197 281 100 210 220 210Z" fill="#ffbe1e"/>
  <path d="M256 130 286 226 386 226 305 285 336 381 256 322 176 381 207 285 126 226 226 226Z" fill="#ffe06e"/>
</svg>
`;
writeFileSync(join(ZIEL, 'icon.svg'), svg);
writeFileSync(join(ZIEL, 'favicon.svg'), svg);
process.stdout.write('icon.svg / favicon.svg geschrieben\n');
