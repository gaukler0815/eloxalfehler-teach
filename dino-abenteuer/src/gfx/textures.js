/**
 * Erzeugt saemtliche Spiel-Texturen zur Laufzeit auf Canvas.
 * Dadurch startet das Spiel ohne eine einzige externe Bilddatei.
 */

import { BIOME } from '../data/levels.js';
import { DINOS } from '../data/dinos.js';
import { dinoSpriteSheet, dunkler, heller, FRAME_W, FRAME_H, FRAMES } from './dinoArt.js';

/** Kleiner deterministischer Zufall, damit Texturen bei jedem Start gleich aussehen. */
function rnd(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function malen(scene, key, w, h, fn) {
  if (scene.textures.exists(key)) scene.textures.remove(key);
  const tex = scene.textures.createCanvas(key, w, h);
  const ctx = tex.getContext();
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  fn(ctx, w, h);
  tex.refresh();
  return tex;
}

function kreis(ctx, x, y, r, farbe) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = farbe;
  ctx.fill();
}

function poly(ctx, punkte, farbe) {
  ctx.beginPath();
  punkte.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
  ctx.closePath();
  ctx.fillStyle = farbe;
  ctx.fill();
}

// ------------------------------------------------------------- Hintergruende

function hintergruende(scene) {
  Object.values(BIOME).forEach((b) => {
    const f = b.farben;

    malen(scene, `himmel_${b.id}`, 64, 512, (ctx, w, h) => {
      const g = ctx.createLinearGradient(0, 0, 0, h);
      g.addColorStop(0, f.himmel[0]);
      g.addColorStop(1, f.himmel[1]);
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, w, h);
    });

    // Ferne Silhouette (Berge / Baumkronen / Stalagmiten / Vulkane)
    malen(scene, `fern_${b.id}`, 512, 220, (ctx, w, h) => {
      const r = rnd(11 + b.id.length * 31);
      ctx.fillStyle = f.fern;
      ctx.beginPath();
      ctx.moveTo(0, h);
      for (let x = 0; x <= w; x += 32) {
        const hoehe = h - (40 + r() * 110);
        ctx.lineTo(x, hoehe);
        ctx.lineTo(x + 16, hoehe + 10 + r() * 30);
      }
      ctx.lineTo(w, h);
      ctx.closePath();
      ctx.fill();
      if (b.id === 'volcano') {
        poly(ctx, [[120, h], [200, 40], [280, h]], dunkler(f.fern, 0.25));
        poly(ctx, [[186, 52], [200, 34], [214, 52]], '#ff6b35');
      }
      if (b.id === 'cave') {
        for (let i = 0; i < 14; i += 1) {
          const x = r() * w;
          kreis(ctx, x, h - 20 - r() * 90, 3 + r() * 4, `${f.deko}55`);
        }
      }
    });

    // Mittlere Ebene
    malen(scene, `mittel_${b.id}`, 512, 200, (ctx, w, h) => {
      const r = rnd(77 + b.id.length * 13);
      ctx.fillStyle = f.mittel;
      ctx.beginPath();
      ctx.moveTo(0, h);
      for (let x = 0; x <= w; x += 24) {
        ctx.lineTo(x, h - (30 + r() * 80));
      }
      ctx.lineTo(w, h);
      ctx.closePath();
      ctx.fill();
      // Baumstaemme / Saeulen
      for (let i = 0; i < 8; i += 1) {
        const x = 20 + i * 62 + r() * 14;
        ctx.fillStyle = dunkler(f.mittel, 0.3);
        ctx.fillRect(x, h - 90, 9, 90);
        if (b.id === 'jungle' || b.id === 'swamp') {
          kreis(ctx, x + 4, h - 96, 22, f.nah);
          kreis(ctx, x - 12, h - 84, 15, f.nah);
          kreis(ctx, x + 20, h - 86, 16, f.nah);
        }
      }
    });

    // Nahe Deko-Ebene (Buesche, Farne, Kristalle, Lavarisse)
    malen(scene, `nah_${b.id}`, 512, 150, (ctx, w, h) => {
      const r = rnd(303 + b.id.length * 7);
      for (let i = 0; i < 16; i += 1) {
        const x = r() * w;
        const y = h - 6;
        if (b.id === 'cave') {
          poly(ctx, [[x - 8, y], [x, y - 30 - r() * 30], [x + 8, y]], `${f.deko}77`);
        } else if (b.id === 'volcano') {
          poly(ctx, [[x - 14, y], [x, y - 24 - r() * 26], [x + 14, y]], dunkler(f.nah, 0.1));
        } else {
          kreis(ctx, x, y - 8, 14 + r() * 12, f.nah);
          kreis(ctx, x + 14, y - 4, 11 + r() * 8, dunkler(f.nah, 0.12));
        }
      }
    });

    // Bodenkachel
    malen(scene, `boden_${b.id}`, 64, 64, (ctx, w, h) => {
      const r = rnd(5 + b.id.length * 97);
      ctx.fillStyle = f.boden;
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = f.bodenOben;
      ctx.fillRect(0, 0, w, 12);
      ctx.fillStyle = heller(f.bodenOben, 0.18);
      for (let x = 0; x < w; x += 8) ctx.fillRect(x, 10, 4, 4);
      for (let i = 0; i < 20; i += 1) {
        kreis(ctx, r() * w, 16 + r() * (h - 18), 1 + r() * 3, dunkler(f.boden, 0.18));
      }
    });

    // Plattformkachel
    malen(scene, `plattform_${b.id}`, 32, 32, (ctx, w, h) => {
      ctx.fillStyle = f.plattform;
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = f.plattformOben;
      ctx.fillRect(0, 0, w, 9);
      ctx.fillStyle = dunkler(f.plattform, 0.25);
      ctx.fillRect(0, h - 4, w, 4);
      ctx.fillStyle = heller(f.plattformOben, 0.2);
      ctx.fillRect(2, 2, 6, 3);
      ctx.fillRect(18, 3, 8, 3);
    });
  });
}

// ------------------------------------------------------------- Sammelobjekte

function sammelobjekte(scene) {
  malen(scene, 'ei', 26, 32, (ctx, w, h) => {
    ctx.strokeStyle = '#a9793a';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.ellipse(w / 2, h / 2 + 2, 11, 14, 0, 0, Math.PI * 2);
    ctx.fillStyle = '#fff4d6';
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = '#e0b060';
    [[8, 12, 3], [17, 18, 3.5], [12, 24, 2.5], [18, 9, 2]].forEach(([x, y, r]) => kreis(ctx, x, y, r, '#e0b060'));
    kreis(ctx, 10, 10, 3, '#ffffffaa');
  });

  const fruechte = [
    { key: 'frucht0', farbe: '#e8505b', blatt: '#3f9c53' },
    { key: 'frucht1', farbe: '#ffb43f', blatt: '#3f9c53' },
    { key: 'frucht2', farbe: '#8b5cf6', blatt: '#3f9c53' },
  ];
  fruechte.forEach(({ key, farbe, blatt }) => {
    malen(scene, key, 24, 26, (ctx, w, h) => {
      ctx.strokeStyle = dunkler(farbe, 0.4);
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.ellipse(w / 2, h / 2 + 3, 9, 9.5, 0, 0, Math.PI * 2);
      ctx.fillStyle = farbe;
      ctx.fill();
      ctx.stroke();
      kreis(ctx, 8, 11, 2.5, '#ffffff88');
      ctx.strokeStyle = '#6b4a2b';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(12, 6);
      ctx.lineTo(13, 1);
      ctx.stroke();
      poly(ctx, [[13, 4], [21, 1], [16, 7]], blatt);
    });
  });

  malen(scene, 'stern', 28, 28, (ctx, w, h) => {
    const cx = w / 2;
    const cy = h / 2;
    const punkte = [];
    for (let i = 0; i < 10; i += 1) {
      const r = i % 2 === 0 ? 13 : 5.5;
      const a = (Math.PI / 5) * i - Math.PI / 2;
      punkte.push([cx + Math.cos(a) * r, cy + Math.sin(a) * r]);
    }
    ctx.strokeStyle = '#b08900';
    ctx.lineWidth = 2;
    ctx.beginPath();
    punkte.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
    ctx.closePath();
    ctx.fillStyle = '#ffd23f';
    ctx.fill();
    ctx.stroke();
  });

  malen(scene, 'herz', 24, 22, (ctx) => {
    ctx.fillStyle = '#e8505b';
    ctx.beginPath();
    ctx.moveTo(12, 20);
    ctx.bezierCurveTo(-4, 10, 4, 0, 12, 7);
    ctx.bezierCurveTo(20, 0, 28, 10, 12, 20);
    ctx.fill();
    kreis(ctx, 8, 7, 2, '#ffffff88');
  });

  // Zielnest
  malen(scene, 'nest', 96, 56, (ctx, w, h) => {
    ctx.strokeStyle = '#6b4a2b';
    ctx.lineWidth = 3;
    for (let i = 0; i < 7; i += 1) {
      ctx.beginPath();
      ctx.ellipse(w / 2, h - 14, 40 - i * 2, 14 - i, (i * 0.3) % Math.PI, 0, Math.PI, false);
      ctx.stroke();
    }
    ctx.strokeStyle = '#a9793a';
    ctx.lineWidth = 2;
    [[36, 28], [58, 26], [47, 20]].forEach(([x, y]) => {
      ctx.beginPath();
      ctx.ellipse(x, y, 10, 13, 0, 0, Math.PI * 2);
      ctx.fillStyle = '#fff4d6';
      ctx.fill();
      ctx.stroke();
    });
  });

  // Checkpoint-Fahne: 2 Frames (grau / bunt)
  [['fahne', '#c9c2ae'], ['fahne_aktiv', '#ffd23f']].forEach(([key, farbe]) => {
    malen(scene, key, 40, 64, (ctx, w, h) => {
      ctx.fillStyle = '#6b4a2b';
      ctx.fillRect(6, 4, 5, h - 6);
      poly(ctx, [[11, 8], [36, 17], [11, 27]], farbe);
      ctx.fillStyle = '#00000022';
      ctx.fillRect(4, h - 6, 16, 5);
    });
  });
}

// --------------------------------------------------------------- Level-Teile

function levelteile(scene) {
  malen(scene, 'liane', 18, 64, (ctx, w, h) => {
    ctx.strokeStyle = '#3f7a3a';
    ctx.lineWidth = 6;
    ctx.beginPath();
    ctx.moveTo(w / 2, 0);
    for (let y = 0; y <= h; y += 8) {
      ctx.lineTo(w / 2 + Math.sin(y / 9) * 4, y);
    }
    ctx.stroke();
    ctx.fillStyle = '#5aa84f';
    for (let y = 6; y < h; y += 16) {
      poly(ctx, [[w / 2, y], [w / 2 + 8, y + 4], [w / 2, y + 8]], '#5aa84f');
      poly(ctx, [[w / 2, y + 6], [w / 2 - 8, y + 10], [w / 2, y + 14]], '#4a9640');
    }
  });

  malen(scene, 'kletterwand', 32, 32, (ctx, w, h) => {
    ctx.fillStyle = '#4a6b3a';
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = '#6f9b52';
    ctx.lineWidth = 3;
    for (let i = -h; i < w; i += 10) {
      ctx.beginPath();
      ctx.moveTo(i, h);
      ctx.lineTo(i + h, 0);
      ctx.stroke();
    }
  });

  malen(scene, 'wasser', 64, 64, (ctx, w, h) => {
    ctx.fillStyle = 'rgba(58,150,200,0.55)';
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = 'rgba(255,255,255,0.10)';
    for (let y = 6; y < h; y += 16) ctx.fillRect(0, y, w, 4);
  });

  malen(scene, 'wasser_oben', 64, 14, (ctx, w, h) => {
    ctx.fillStyle = 'rgba(150,220,245,0.85)';
    ctx.beginPath();
    ctx.moveTo(0, h);
    for (let x = 0; x <= w; x += 4) ctx.lineTo(x, 4 + Math.sin(x / 6) * 3.5);
    ctx.lineTo(w, h);
    ctx.closePath();
    ctx.fill();
  });

  malen(scene, 'treibholz', 108, 26, (ctx, w, h) => {
    ctx.strokeStyle = '#4a3220';
    ctx.lineWidth = 2;
    ctx.fillStyle = '#7a5433';
    ctx.beginPath();
    ctx.roundRect ? ctx.roundRect(1, 3, w - 2, h - 6, 10) : ctx.rect(1, 3, w - 2, h - 6);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = '#5f8b4c';
    for (let x = 8; x < w - 8; x += 22) {
      kreis(ctx, x, 6, 5, '#5f8b4c');
    }
    ctx.strokeStyle = '#5a3d24';
    for (let x = 14; x < w; x += 20) {
      ctx.beginPath();
      ctx.moveTo(x, 6);
      ctx.lineTo(x, h - 6);
      ctx.stroke();
    }
  });

  // Sprungfeder: 2 Frames (zusammengedrueckt / entspannt)
  [['feder', 20], ['feder_aus', 8]].forEach(([key, hoehe]) => {
    malen(scene, key, 48, 32, (ctx, w, h) => {
      ctx.strokeStyle = '#3a3a4a';
      ctx.lineWidth = 3;
      const basis = h - 4;
      ctx.beginPath();
      for (let i = 0; i <= 3; i += 1) {
        const y = basis - (hoehe / 3) * i;
        ctx.moveTo(8, y);
        ctx.lineTo(w - 8, y - hoehe / 6);
      }
      ctx.stroke();
      ctx.fillStyle = '#e8505b';
      ctx.fillRect(4, basis - hoehe - 8, w - 8, 8);
      ctx.fillStyle = '#8a8a9a';
      ctx.fillRect(2, basis, w - 4, 5);
    });
  });

  malen(scene, 'fels', 44, 44, (ctx, w, h) => {
    ctx.strokeStyle = '#4a4458';
    ctx.lineWidth = 2;
    poly(ctx, [[4, h - 2], [2, 14], [14, 3], [32, 2], [42, 16], [40, h - 2]], '#79728f');
    ctx.stroke();
    ctx.fillStyle = '#948dab';
    poly(ctx, [[10, 12], [22, 8], [26, 18], [12, 20]], '#948dab');
    ctx.fillStyle = '#5d5773';
    poly(ctx, [[24, 24], [36, 22], [34, 34], [24, 33]], '#5d5773');
  });

  malen(scene, 'broeckel', 32, 20, (ctx, w, h) => {
    ctx.fillStyle = '#8a7f6a';
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = '#a89b80';
    ctx.fillRect(0, 0, w, 6);
    ctx.strokeStyle = '#5f5748';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(6, 0);
    ctx.lineTo(11, h);
    ctx.moveTo(22, 0);
    ctx.lineTo(17, h);
    ctx.stroke();
  });

  malen(scene, 'hebebuehne', 96, 22, (ctx, w, h) => {
    ctx.fillStyle = '#5a4a3a';
    ctx.fillRect(0, 6, w, h - 6);
    ctx.fillStyle = '#c9a227';
    ctx.fillRect(0, 0, w, 7);
    ctx.fillStyle = '#8a6a2a';
    for (let x = 4; x < w; x += 16) ctx.fillRect(x, 9, 8, 5);
  });

  malen(scene, 'feuerball', 26, 26, (ctx, w, h) => {
    kreis(ctx, w / 2, h / 2, 12, '#ff6b35');
    kreis(ctx, w / 2, h / 2, 8, '#ffb43f');
    kreis(ctx, w / 2, h / 2 - 1, 4, '#fff3b0');
  });

  malen(scene, 'partikel', 12, 12, (ctx, w, h) => {
    kreis(ctx, w / 2, h / 2, 5, '#ffffff');
  });

  malen(scene, 'funke', 10, 10, (ctx, w, h) => {
    poly(ctx, [[5, 0], [7, 4], [10, 5], [7, 6], [5, 10], [3, 6], [0, 5], [3, 4]], '#ffd23f');
  });

  malen(scene, 'blase', 12, 12, (ctx, w, h) => {
    ctx.strokeStyle = 'rgba(255,255,255,0.85)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(6, 6, 4, 0, Math.PI * 2);
    ctx.stroke();
  });

  malen(scene, 'schild', 80, 80, (ctx, w, h) => {
    ctx.strokeStyle = 'rgba(120,220,255,0.9)';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.arc(w / 2, h / 2, 34, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fillStyle = 'rgba(120,220,255,0.18)';
    ctx.fill();
  });
}

// -------------------------------------------------------------------- Gegner

const GEGNER_TYPEN = {
  kaefer: { farbe: '#7b4b9c', akzent: '#ffd23f' },
  libelle: { farbe: '#3fa9a0', akzent: '#cdeaf7' },
  krabbler: { farbe: '#b06fa8', akzent: '#7ee0e8' },
  feuergeist: { farbe: '#ff8c42', akzent: '#fff3b0' },
};

function gegner(scene) {
  Object.entries(GEGNER_TYPEN).forEach(([typ, { farbe, akzent }]) => {
    const key = `gegner_${typ}`;
    if (scene.textures.exists(key)) scene.textures.remove(key);
    const w = 44;
    const h = 36;
    const tex = scene.textures.createCanvas(key, w * 2, h);
    const ctx = tex.getContext();
    ctx.lineJoin = 'round';
    for (let f = 0; f < 2; f += 1) {
      ctx.save();
      ctx.translate(f * w, 0);
      ctx.strokeStyle = dunkler(farbe, 0.45);
      ctx.lineWidth = 2;
      const wippe = f === 0 ? 0 : -2;

      if (typ === 'libelle') {
        const fl = f === 0 ? -6 : 4;
        poly(ctx, [[22, 16], [6, 8 + fl], [20, 20]], akzent);
        poly(ctx, [[24, 16], [40, 8 - fl], [26, 20]], akzent);
        ctx.beginPath();
        ctx.ellipse(22, 20 + wippe, 13, 6, 0, 0, Math.PI * 2);
        ctx.fillStyle = farbe;
        ctx.fill();
        ctx.stroke();
        kreis(ctx, 33, 18 + wippe, 5, dunkler(farbe, 0.15));
      } else if (typ === 'feuergeist') {
        ctx.beginPath();
        ctx.moveTo(10, 32);
        ctx.quadraticCurveTo(4, 16 + wippe, 22, 3 + wippe);
        ctx.quadraticCurveTo(40, 16 + wippe, 34, 32);
        ctx.closePath();
        ctx.fillStyle = farbe;
        ctx.fill();
        ctx.stroke();
        kreis(ctx, 22, 22, 7, akzent);
      } else {
        // Kaefer / Krabbler: runder Panzer mit Beinchen
        const beine = f === 0 ? 3 : -3;
        ctx.strokeStyle = dunkler(farbe, 0.45);
        ctx.lineWidth = 3;
        [10, 22, 34].forEach((x, i) => {
          ctx.beginPath();
          ctx.moveTo(x, 24);
          ctx.lineTo(x + (i % 2 === 0 ? beine : -beine), 33);
          ctx.stroke();
        });
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.ellipse(22, 18 + wippe, 16, 12, 0, 0, Math.PI * 2);
        ctx.fillStyle = farbe;
        ctx.fill();
        ctx.stroke();
        ctx.beginPath();
        ctx.ellipse(22, 14 + wippe, 12, 7, 0, 0, Math.PI * 2);
        ctx.fillStyle = akzent;
        ctx.fill();
      }
      // Freundliche Augen
      kreis(ctx, 26, 14 + wippe, 4, '#ffffff');
      kreis(ctx, 27, 14 + wippe, 2, '#1a1a26');
      kreis(ctx, 17, 14 + wippe, 4, '#ffffff');
      kreis(ctx, 18, 14 + wippe, 2, '#1a1a26');
      ctx.restore();
    }
    tex.refresh();
    tex.add(0, 0, 0, 0, w, h);
    tex.add(1, 0, w, 0, w, h);
  });
}

// --------------------------------------------------------------------- Dinos

function dinos(scene) {
  DINOS.forEach((d) => {
    const key = `dino_${d.id}`;
    if (scene.textures.exists(key)) scene.textures.remove(key);
    const canvas = dinoSpriteSheet(d, 1);
    const tex = scene.textures.addCanvas(key, canvas);
    for (let f = 0; f < FRAMES; f += 1) {
      tex.add(f, 0, f * FRAME_W, 0, FRAME_W, FRAME_H);
    }
  });
}

/** Alle Animationen anlegen (einmalig, global im Anim-Manager). */
export function animationenAnlegen(scene) {
  DINOS.forEach((d) => {
    const key = `dino_${d.id}`;
    if (!scene.anims.exists(`${d.id}_lauf`)) {
      scene.anims.create({
        key: `${d.id}_lauf`,
        frames: [
          { key, frame: 1 },
          { key, frame: 0 },
          { key, frame: 2 },
          { key, frame: 0 },
        ],
        frameRate: 10,
        repeat: -1,
      });
    }
    if (!scene.anims.exists(`${d.id}_stehen`)) {
      scene.anims.create({ key: `${d.id}_stehen`, frames: [{ key, frame: 0 }], frameRate: 1 });
    }
    if (!scene.anims.exists(`${d.id}_sprung`)) {
      scene.anims.create({ key: `${d.id}_sprung`, frames: [{ key, frame: 3 }], frameRate: 1 });
    }
  });

  Object.keys(GEGNER_TYPEN).forEach((typ) => {
    if (!scene.anims.exists(`gegner_${typ}_lauf`)) {
      scene.anims.create({
        key: `gegner_${typ}_lauf`,
        frames: [
          { key: `gegner_${typ}`, frame: 0 },
          { key: `gegner_${typ}`, frame: 1 },
        ],
        frameRate: 6,
        repeat: -1,
      });
    }
  });
}

/** Wird einmal in der BootScene aufgerufen. */
export function texturenErzeugen(scene) {
  hintergruende(scene);
  sammelobjekte(scene);
  levelteile(scene);
  gegner(scene);
  dinos(scene);
}

export { GEGNER_TYPEN };
