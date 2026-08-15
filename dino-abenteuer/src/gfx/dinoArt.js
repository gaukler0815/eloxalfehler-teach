/**
 * Dino-Zeichner.
 *
 * Alle 31 Dinos werden zur Laufzeit auf ein Canvas gemalt - es gibt
 * bewusst keine externen Bilddateien. Dieselbe Funktion liefert die
 * Phaser-Sprites (4 Frames) und die kleinen Vorschaubilder in der
 * HTML-Dino-Auswahl.
 *
 * Frames: 0 = Stand, 1 = Lauf A, 2 = Lauf B, 3 = Sprung
 */

export const FRAME_W = 72;
export const FRAME_H = 64;
export const FRAMES = 4;

// ---------------------------------------------------------------- Farbhilfen

function mische(hex, ziel, anteil) {
  const a = hexZuRgb(hex);
  const b = hexZuRgb(ziel);
  const r = Math.round(a[0] + (b[0] - a[0]) * anteil);
  const g = Math.round(a[1] + (b[1] - a[1]) * anteil);
  const bl = Math.round(a[2] + (b[2] - a[2]) * anteil);
  return `rgb(${r},${g},${bl})`;
}

function hexZuRgb(hex) {
  const h = hex.replace('#', '');
  const v = h.length === 3 ? h.split('').map((c) => c + c).join('') : h;
  return [parseInt(v.slice(0, 2), 16), parseInt(v.slice(2, 4), 16), parseInt(v.slice(4, 6), 16)];
}

export function dunkler(hex, anteil = 0.3) {
  return mische(hex, '#000000', anteil);
}

export function heller(hex, anteil = 0.3) {
  return mische(hex, '#ffffff', anteil);
}

// ------------------------------------------------------------- Grundformen

function ellipse(ctx, x, y, rx, ry, fill, rot = 0) {
  ctx.beginPath();
  ctx.ellipse(x, y, Math.max(0.5, rx), Math.max(0.5, ry), rot, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.stroke();
}

function poly(ctx, punkte, fill, schliessen = true) {
  ctx.beginPath();
  punkte.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
  if (schliessen) ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.stroke();
}

function rrect(ctx, x, y, w, h, r, fill) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.stroke();
}

/** Ein Bein: Oberschenkel + Fuss, um `winkel` nach vorne/hinten gedreht. */
function bein(ctx, x, y, laenge, breite, winkel, farbe, fussFarbe) {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(winkel);
  rrect(ctx, -breite / 2, 0, breite, laenge, breite / 2, farbe);
  rrect(ctx, -breite / 2 - 1, laenge - 3, breite + 5, 5, 2.5, fussFarbe);
  ctx.restore();
}

function auge(ctx, x, y, r, blickRichtung = 1) {
  ctx.save();
  ctx.lineWidth = 1;
  ellipse(ctx, x, y, r, r, '#ffffff');
  ctx.fillStyle = '#1a1a26';
  ctx.beginPath();
  ctx.arc(x + r * 0.32 * blickRichtung, y, r * 0.52, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

// --------------------------------------------------------- Animationswerte

function animation(frame) {
  switch (frame) {
    case 1:
      return { hebe: -1, vorn: 0.55, hinten: -0.4, schwanz: -0.12, kopf: -1, sprung: false };
    case 2:
      return { hebe: 0, vorn: -0.4, hinten: 0.55, schwanz: 0.12, kopf: 1, sprung: false };
    case 3:
      return { hebe: -3, vorn: -0.75, hinten: 0.7, schwanz: -0.3, kopf: -2, sprung: true };
    default:
      return { hebe: 0, vorn: 0.06, hinten: -0.06, schwanz: 0, kopf: 0, sprung: false };
  }
}

// ------------------------------------------------------------ Hauptzeichner

/**
 * Malt einen Dino nach rechts blickend in ein FRAME_W x FRAME_H Feld.
 * @param {CanvasRenderingContext2D} ctx
 * @param {object} dino Eintrag aus DINOS
 * @param {number} frame 0..3
 * @param {number} offsetX
 * @param {number} offsetY
 */
export function zeichneDino(ctx, dino, frame = 0, offsetX = 0, offsetY = 0) {
  const a = animation(frame);
  const c = dino.colors;
  const linie = dunkler(c.body, 0.55);
  const schatten = dunkler(c.body, 0.22);

  ctx.save();
  ctx.translate(offsetX, offsetY + a.hebe);
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.strokeStyle = linie;
  ctx.lineWidth = 2;

  const zeichner = BAUPLAENE[dino.shape] || BAUPLAENE.theropod;
  zeichner(ctx, { c, linie, schatten, a, dino });

  ctx.restore();
}

// Bodenlinie im Frame
const BODEN = 60;

// --------------------------------------------------------------- Bauplaene

function schwanzDick(ctx, c, spitzeY, dicke = 9, laenge = 22) {
  poly(
    ctx,
    [
      [26, 36 - dicke / 2],
      [26 - laenge, spitzeY - 2],
      [26 - laenge - 3, spitzeY + 2],
      [26, 36 + dicke / 2],
    ],
    c
  );
}

function theropod(ctx, { c, schatten, a }, opt = {}) {
  const kopfX = 50;
  const kopfY = 24 + a.kopf;
  // Schwanz
  schwanzDick(ctx, c.body, 30 + a.schwanz * 18, 11, 24);
  // Hinteres Bein
  bein(ctx, 30, 40, 16, 9, a.hinten * 0.5, schatten, dunkler(c.body, 0.4));
  // Koerper
  ellipse(ctx, 34, 36, 15, 12, c.body);
  ellipse(ctx, 34, 41, 10, 6, c.belly);
  if (opt.segel) {
    poly(
      ctx,
      [
        [22, 30],
        [28, 12],
        [40, 10],
        [46, 28],
      ],
      c.accent
    );
    ellipse(ctx, 34, 36, 15, 12, c.body);
    ellipse(ctx, 34, 41, 10, 6, c.belly);
  }
  // Vorderes Bein
  bein(ctx, 38, 40, 16, 9, a.vorn * 0.5, c.body, dunkler(c.body, 0.35));
  // Aermchen
  poly(
    ctx,
    [
      [42, 34],
      [50, 36 + a.kopf * 0.5],
      [50, 39 + a.kopf * 0.5],
      [42, 37],
    ],
    schatten
  );
  // Hals + Kopf
  poly(
    ctx,
    [
      [38, 28],
      [kopfX - 6, kopfY - 4],
      [kopfX - 4, kopfY + 8],
      [40, 34],
    ],
    c.body
  );
  ellipse(ctx, kopfX, kopfY, opt.kopfR || 11, (opt.kopfR || 11) * 0.82, c.body);
  // Schnauze
  rrect(ctx, kopfX + 2, kopfY - 2, 13, 9, 3, c.body);
  // Zaehne
  ctx.strokeStyle = 'rgba(0,0,0,0)';
  poly(
    ctx,
    [
      [kopfX + 3, kopfY + 7],
      [kopfX + 5, kopfY + 10],
      [kopfX + 7, kopfY + 7],
      [kopfX + 9, kopfY + 10],
      [kopfX + 11, kopfY + 7],
      [kopfX + 13, kopfY + 9],
      [kopfX + 14, kopfY + 7],
    ],
    '#fffdf5'
  );
  ctx.strokeStyle = dunkler(c.body, 0.55);
  if (opt.kaemme) {
    poly(ctx, [[kopfX - 4, kopfY - 8], [kopfX + 1, kopfY - 16], [kopfX + 5, kopfY - 7]], c.accent);
  }
  if (opt.hoerner) {
    poly(ctx, [[kopfX - 1, kopfY - 8], [kopfX + 1, kopfY - 15], [kopfX + 4, kopfY - 7]], c.accent);
  }
  auge(ctx, kopfX + 3, kopfY - 2, 3.4);
}

function raptor(ctx, ctxt) {
  const { c, schatten, a } = ctxt;
  const kopfX = 52;
  const kopfY = 26 + a.kopf;
  poly(
    ctx,
    [
      [26, 32],
      [4, 22 + a.schwanz * 12],
      [3, 27 + a.schwanz * 12],
      [26, 39],
    ],
    c.body
  );
  // Federn am Schwanz
  poly(ctx, [[10, 24 + a.schwanz * 12], [2, 18 + a.schwanz * 12], [8, 28 + a.schwanz * 12]], c.accent);
  bein(ctx, 30, 40, 17, 8, a.hinten * 0.6, schatten, dunkler(c.body, 0.4));
  ellipse(ctx, 34, 36, 13, 9, c.body);
  ellipse(ctx, 34, 40, 8, 4.5, c.belly);
  bein(ctx, 37, 40, 17, 8, a.vorn * 0.6, c.body, dunkler(c.body, 0.35));
  poly(
    ctx,
    [
      [42, 33],
      [49, 36],
      [49, 38],
      [42, 36],
    ],
    schatten
  );
  poly(
    ctx,
    [
      [40, 30],
      [kopfX - 6, kopfY - 3],
      [kopfX - 3, kopfY + 6],
      [41, 34],
    ],
    c.body
  );
  ellipse(ctx, kopfX, kopfY, 8.5, 7, c.body);
  rrect(ctx, kopfX + 1, kopfY - 1, 12, 7, 3, c.body);
  poly(ctx, [[kopfX - 5, kopfY - 6], [kopfX - 1, kopfY - 13], [kopfX + 3, kopfY - 5]], c.accent);
  auge(ctx, kopfX + 2, kopfY - 2, 3);
}

function sauropod(ctx, { c, schatten, a }) {
  const kopfX = 56;
  const kopfY = 14 + a.kopf;
  poly(
    ctx,
    [
      [24, 34],
      [2, 30 + a.schwanz * 14],
      [2, 34 + a.schwanz * 14],
      [24, 41],
    ],
    c.body
  );
  bein(ctx, 22, 40, 18, 9, a.hinten * 0.35, schatten, dunkler(c.body, 0.4));
  bein(ctx, 40, 40, 18, 9, a.vorn * 0.35, schatten, dunkler(c.body, 0.4));
  ellipse(ctx, 32, 36, 17, 12, c.body);
  ellipse(ctx, 32, 41, 11, 6, c.belly);
  bein(ctx, 26, 40, 18, 9, a.vorn * 0.35, c.body, dunkler(c.body, 0.35));
  bein(ctx, 44, 40, 18, 9, a.hinten * 0.35, c.body, dunkler(c.body, 0.35));
  // Langer Hals
  poly(
    ctx,
    [
      [40, 28],
      [kopfX - 6, kopfY + 2],
      [kopfX - 1, kopfY + 8],
      [45, 33],
    ],
    c.body
  );
  ellipse(ctx, kopfX, kopfY, 7, 5.5, c.body);
  rrect(ctx, kopfX + 2, kopfY - 1, 8, 5, 2.5, c.body);
  poly(ctx, [[kopfX - 3, kopfY - 5], [kopfX, kopfY - 10], [kopfX + 3, kopfY - 4]], c.accent);
  auge(ctx, kopfX + 2, kopfY - 1, 2.6);
}

function ceratops(ctx, { c, schatten, a }) {
  const kopfX = 50;
  const kopfY = 30 + a.kopf;
  schwanzDick(ctx, c.body, 34 + a.schwanz * 12, 10, 20);
  bein(ctx, 24, 42, 15, 9, a.hinten * 0.4, schatten, dunkler(c.body, 0.4));
  bein(ctx, 40, 42, 15, 9, a.vorn * 0.4, schatten, dunkler(c.body, 0.4));
  ellipse(ctx, 32, 38, 16, 11, c.body);
  ellipse(ctx, 32, 43, 10, 5, c.belly);
  bein(ctx, 28, 42, 15, 9, a.vorn * 0.4, c.body, dunkler(c.body, 0.35));
  bein(ctx, 44, 42, 15, 9, a.hinten * 0.4, c.body, dunkler(c.body, 0.35));
  // Nackenschild
  poly(
    ctx,
    [
      [42, 22],
      [52, 18],
      [58, 28],
      [52, 40],
      [42, 38],
    ],
    c.accent
  );
  ellipse(ctx, kopfX, kopfY, 10, 8, c.body);
  rrect(ctx, kopfX + 3, kopfY - 2, 12, 8, 4, c.body);
  // Hoerner
  poly(ctx, [[kopfX + 4, kopfY - 4], [kopfX + 10, kopfY - 14], [kopfX + 8, kopfY - 3]], '#f5efdc');
  poly(ctx, [[kopfX - 2, kopfY - 6], [kopfX + 2, kopfY - 15], [kopfX + 4, kopfY - 5]], '#f5efdc');
  poly(ctx, [[kopfX + 12, kopfY + 2], [kopfX + 18, kopfY - 2], [kopfX + 13, kopfY + 6]], '#f5efdc');
  auge(ctx, kopfX + 4, kopfY - 1, 3);
}

function anky(ctx, { c, schatten, a }) {
  const kopfX = 51;
  const kopfY = 38 + a.kopf;
  // Schwanzkeule
  poly(
    ctx,
    [
      [22, 36],
      [8, 32 + a.schwanz * 10],
      [8, 38 + a.schwanz * 10],
      [22, 42],
    ],
    c.body
  );
  ellipse(ctx, 6, 35 + a.schwanz * 10, 7, 6, c.accent);
  bein(ctx, 24, 44, 12, 9, a.hinten * 0.3, schatten, dunkler(c.body, 0.4));
  bein(ctx, 42, 44, 12, 9, a.vorn * 0.3, schatten, dunkler(c.body, 0.4));
  ellipse(ctx, 33, 40, 17, 10, c.body);
  // Panzer
  poly(
    ctx,
    [
      [17, 39],
      [22, 30],
      [33, 27],
      [44, 30],
      [49, 39],
    ],
    c.accent
  );
  for (let i = 0; i < 4; i += 1) {
    poly(
      ctx,
      [
        [21 + i * 8, 31],
        [24 + i * 8, 24],
        [27 + i * 8, 31],
      ],
      '#f5efdc'
    );
  }
  bein(ctx, 28, 44, 12, 9, a.vorn * 0.3, c.body, dunkler(c.body, 0.35));
  bein(ctx, 46, 44, 12, 9, a.hinten * 0.3, c.body, dunkler(c.body, 0.35));
  ellipse(ctx, kopfX, kopfY, 9, 7, c.body);
  rrect(ctx, kopfX + 3, kopfY - 2, 9, 7, 3, c.body);
  auge(ctx, kopfX + 3, kopfY - 2, 2.8);
}

function stego(ctx, { c, schatten, a }) {
  const kopfX = 54;
  const kopfY = 32 + a.kopf;
  poly(
    ctx,
    [
      [22, 34],
      [3, 26 + a.schwanz * 12],
      [3, 31 + a.schwanz * 12],
      [22, 41],
    ],
    c.body
  );
  // Schwanzstacheln
  poly(ctx, [[9, 28 + a.schwanz * 12], [3, 20 + a.schwanz * 12], [11, 31 + a.schwanz * 12]], '#f5efdc');
  poly(ctx, [[13, 29 + a.schwanz * 12], [9, 20 + a.schwanz * 12], [16, 32 + a.schwanz * 12]], '#f5efdc');
  bein(ctx, 24, 42, 15, 9, a.hinten * 0.35, schatten, dunkler(c.body, 0.4));
  bein(ctx, 42, 42, 13, 8, a.vorn * 0.35, schatten, dunkler(c.body, 0.4));
  ellipse(ctx, 33, 37, 16, 11, c.body);
  ellipse(ctx, 33, 42, 10, 5, c.belly);
  // Rueckenplatten
  for (let i = 0; i < 5; i += 1) {
    const x = 22 + i * 6;
    const h = 8 + Math.sin((i / 4) * Math.PI) * 6;
    poly(
      ctx,
      [
        [x - 4, 28],
        [x, 28 - h],
        [x + 4, 28],
      ],
      c.accent
    );
  }
  bein(ctx, 28, 42, 15, 9, a.vorn * 0.35, c.body, dunkler(c.body, 0.35));
  bein(ctx, 46, 42, 13, 8, a.hinten * 0.35, c.body, dunkler(c.body, 0.35));
  poly(
    ctx,
    [
      [42, 30],
      [kopfX - 5, kopfY - 2],
      [kopfX - 3, kopfY + 5],
      [43, 35],
    ],
    c.body
  );
  ellipse(ctx, kopfX, kopfY, 7, 5.5, c.body);
  rrect(ctx, kopfX + 2, kopfY - 1, 8, 5, 2.5, c.body);
  auge(ctx, kopfX + 2, kopfY - 2, 2.6);
}

function ptero(ctx, { c, schatten, a }) {
  const kopfX = 50;
  const kopfY = 24 + a.kopf;
  const fluegel = a.sprung ? -12 : a.kopf * 3;
  // Hinterer Fluegel
  poly(
    ctx,
    [
      [30, 32],
      [6, 20 + fluegel],
      [2, 30 + fluegel],
      [26, 40],
    ],
    schatten
  );
  bein(ctx, 30, 40, 11, 6, a.hinten * 0.5, schatten, dunkler(c.body, 0.4));
  bein(ctx, 36, 40, 11, 6, a.vorn * 0.5, c.body, dunkler(c.body, 0.35));
  ellipse(ctx, 33, 35, 12, 9, c.body);
  ellipse(ctx, 33, 39, 8, 4, c.belly);
  // Vorderer Fluegel
  poly(
    ctx,
    [
      [34, 30],
      [12, 14 + fluegel],
      [6, 24 + fluegel],
      [30, 38],
    ],
    c.accent
  );
  poly(
    ctx,
    [
      [38, 30],
      [kopfX - 5, kopfY + 2],
      [kopfX - 3, kopfY + 8],
      [38, 35],
    ],
    c.body
  );
  ellipse(ctx, kopfX, kopfY, 8, 6.5, c.body);
  // Langer Schnabel
  poly(
    ctx,
    [
      [kopfX + 4, kopfY - 2],
      [kopfX + 20, kopfY + 3],
      [kopfX + 4, kopfY + 5],
    ],
    heller(c.accent, 0.25)
  );
  // Kamm nach hinten
  poly(ctx, [[kopfX - 4, kopfY - 4], [kopfX - 16, kopfY - 10], [kopfX - 2, kopfY + 2]], c.accent);
  auge(ctx, kopfX + 2, kopfY - 2, 3);
}

function aquatic(ctx, { c, schatten, a }) {
  const kopfX = 56;
  const kopfY = 22 + a.kopf;
  // Schwanzflosse
  poly(
    ctx,
    [
      [22, 36],
      [4, 30 + a.schwanz * 14],
      [8, 40 + a.schwanz * 14],
      [22, 43],
    ],
    c.body
  );
  poly(ctx, [[10, 32 + a.schwanz * 14], [2, 24 + a.schwanz * 14], [8, 38 + a.schwanz * 14]], c.accent);
  ellipse(ctx, 34, 38, 17, 11, c.body);
  ellipse(ctx, 34, 43, 11, 5, c.belly);
  // Flossen
  poly(ctx, [[26, 44], [16, 54 + a.hinten * 4], [30, 47]], c.accent);
  poly(ctx, [[42, 44], [34, 55 + a.vorn * 4], [46, 46]], c.accent);
  // Hals
  poly(
    ctx,
    [
      [42, 30],
      [kopfX - 6, kopfY + 2],
      [kopfX - 2, kopfY + 9],
      [45, 36],
    ],
    c.body
  );
  ellipse(ctx, kopfX, kopfY, 8, 6, c.body);
  rrect(ctx, kopfX + 2, kopfY - 1, 9, 5, 2.5, c.body);
  auge(ctx, kopfX + 2, kopfY - 2, 2.8);
}

function hadro(ctx, { c, schatten, a }) {
  const kopfX = 51;
  const kopfY = 24 + a.kopf;
  schwanzDick(ctx, c.body, 30 + a.schwanz * 16, 10, 23);
  bein(ctx, 30, 40, 17, 9, a.hinten * 0.5, schatten, dunkler(c.body, 0.4));
  ellipse(ctx, 34, 36, 15, 11, c.body);
  ellipse(ctx, 34, 41, 10, 5.5, c.belly);
  bein(ctx, 38, 40, 17, 9, a.vorn * 0.5, c.body, dunkler(c.body, 0.35));
  poly(
    ctx,
    [
      [42, 33],
      [49, 36],
      [49, 38],
      [42, 36],
    ],
    schatten
  );
  poly(
    ctx,
    [
      [38, 28],
      [kopfX - 6, kopfY - 2],
      [kopfX - 3, kopfY + 7],
      [40, 34],
    ],
    c.body
  );
  ellipse(ctx, kopfX, kopfY, 9, 7, c.body);
  // Entenschnabel
  poly(
    ctx,
    [
      [kopfX + 3, kopfY - 2],
      [kopfX + 16, kopfY + 1],
      [kopfX + 15, kopfY + 6],
      [kopfX + 3, kopfY + 6],
    ],
    heller(c.body, 0.3)
  );
  // Roehrenkamm nach hinten
  poly(
    ctx,
    [
      [kopfX - 4, kopfY - 5],
      [kopfX - 20, kopfY - 16],
      [kopfX - 15, kopfY - 20],
      [kopfX - 2, kopfY - 2],
    ],
    c.accent
  );
  auge(ctx, kopfX + 2, kopfY - 2, 3);
}

function pachy(ctx, { c, schatten, a }) {
  const kopfX = 50;
  const kopfY = 26 + a.kopf;
  schwanzDick(ctx, c.body, 32 + a.schwanz * 14, 10, 22);
  bein(ctx, 30, 40, 16, 9, a.hinten * 0.5, schatten, dunkler(c.body, 0.4));
  ellipse(ctx, 34, 36, 15, 11, c.body);
  ellipse(ctx, 34, 41, 10, 5.5, c.belly);
  bein(ctx, 38, 40, 16, 9, a.vorn * 0.5, c.body, dunkler(c.body, 0.35));
  poly(
    ctx,
    [
      [38, 29],
      [kopfX - 6, kopfY],
      [kopfX - 3, kopfY + 8],
      [40, 34],
    ],
    c.body
  );
  ellipse(ctx, kopfX, kopfY, 9, 7.5, c.body);
  // Dicke Schaedelkuppel
  ellipse(ctx, kopfX + 1, kopfY - 6, 10, 6.5, c.accent);
  for (let i = 0; i < 5; i += 1) {
    poly(
      ctx,
      [
        [kopfX - 8 + i * 4, kopfY - 9],
        [kopfX - 6 + i * 4, kopfY - 13],
        [kopfX - 4 + i * 4, kopfY - 9],
      ],
      '#f5efdc'
    );
  }
  rrect(ctx, kopfX + 3, kopfY - 1, 11, 7, 3, c.body);
  auge(ctx, kopfX + 3, kopfY - 1, 2.9);
}

const BAUPLAENE = {
  theropod: (ctx, o) => theropod(ctx, o, { kopfR: 11 }),
  spino: (ctx, o) => theropod(ctx, o, { kopfR: 10, segel: true }),
  raptor,
  sauropod,
  ceratops,
  anky,
  stego,
  ptero,
  aquatic,
  hadro,
  pachy,
};

// Spezial-Auspraegungen ueber die Dino-ID
const SONDERFORM = {
  dilo: (ctx, o) => theropod(ctx, o, { kopfR: 10, kaemme: true }),
  carno: (ctx, o) => theropod(ctx, o, { kopfR: 11, hoerner: true }),
  allo: (ctx, o) => theropod(ctx, o, { kopfR: 11, hoerner: true }),
};

/**
 * Erzeugt ein Canvas mit allen 4 Frames nebeneinander.
 * @returns {HTMLCanvasElement}
 */
export function dinoSpriteSheet(dino, skalierung = 1) {
  const canvas = document.createElement('canvas');
  canvas.width = FRAME_W * FRAMES * skalierung;
  canvas.height = FRAME_H * skalierung;
  const ctx = canvas.getContext('2d');
  ctx.scale(skalierung, skalierung);
  for (let f = 0; f < FRAMES; f += 1) {
    ctx.save();
    ctx.translate(f * FRAME_W, 0);
    zeichnen(ctx, dino, f);
    ctx.restore();
  }
  return canvas;
}

function zeichnen(ctx, dino, frame) {
  const sonder = SONDERFORM[dino.id];
  if (!sonder) {
    zeichneDino(ctx, dino, frame);
    return;
  }
  const a = animation(frame);
  const c = dino.colors;
  ctx.save();
  ctx.translate(0, a.hebe);
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.strokeStyle = dunkler(c.body, 0.55);
  ctx.lineWidth = 2;
  sonder(ctx, { c, linie: dunkler(c.body, 0.55), schatten: dunkler(c.body, 0.22), a, dino });
  ctx.restore();
}

/**
 * Kleines Vorschaubild fuer die HTML-Auswahl.
 * @returns {HTMLCanvasElement}
 */
export function dinoVorschau(dino, groesse = 64) {
  const canvas = document.createElement('canvas');
  const s = groesse / FRAME_W;
  canvas.width = groesse;
  canvas.height = Math.round(FRAME_H * s);
  const ctx = canvas.getContext('2d');
  ctx.scale(s, s);
  zeichnen(ctx, dino, 0);
  return canvas;
}

export { BODEN };
