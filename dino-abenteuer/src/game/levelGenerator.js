/**
 * Baut aus einer Level-Konfiguration die konkrete Geometrie.
 *
 * Wichtig fuer ein 7-jaehriges Kind: Alles muss erreichbar sein - auch
 * mit dem langsamsten und schwaechsten Dino (Ankor: speed 165, jump 440).
 * Bei Gravitation 1000 schafft er ~97 px Sprunghoehe und ~145 px Weite.
 * Deshalb: Luecken max. 130 px, Stufenhoehe max. 80 px.
 */

export const GRAVITATION = 1000;
export const BODEN_Y = 640; // Oberkante des Bodens
export const WELT_H = 760;

const STUFE = 80; // vertikaler Abstand zweier Plattform-Ebenen
const EBENEN = [BODEN_Y - 95, BODEN_Y - 175, BODEN_Y - 255, BODEN_Y - 335];

/** mulberry32 - kleiner, schneller Seed-Zufall. */
function zufall(seed) {
  let a = seed >>> 0;
  return function next() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function zwischen(r, min, max) {
  return min + r() * (max - min);
}

function ganz(r, min, max) {
  return Math.floor(zwischen(r, min, max + 1));
}

function waehle(r, liste) {
  return liste[Math.min(liste.length - 1, Math.floor(r() * liste.length))];
}

/**
 * @param {object} cfg Eintrag aus LEVELS
 * @returns {object} Geometrie-Beschreibung des Levels
 */
export function levelBauen(cfg) {
  const r = zufall(cfg.seed);
  const L = {
    cfg,
    breite: cfg.breite,
    hoehe: WELT_H,
    start: { x: 110, y: BODEN_Y - 60 },
    boden: [],
    plattformen: [],
    broeckel: [],
    federn: [],
    felsen: [],
    lianen: [],
    kletterwaende: [],
    wasser: [],
    treibhoelzer: [],
    hebebuehnen: [],
    feuerbaelle: [],
    gegner: [],
    eier: [],
    fruechte: [],
    checkpoints: [],
    ziel: { x: cfg.breite - 130, y: BODEN_Y - 40 },
  };

  const istSumpf = cfg.biom === 'swamp';
  const gegnerTyp = { jungle: 'kaefer', swamp: 'libelle', cave: 'krabbler', volcano: 'feuergeist' }[
    cfg.biom
  ];

  // ---------------------------------------------------------------- Boden
  // Abwechselnd feste Boden-Abschnitte und Luecken (Wasser bzw. Abgrund).
  const luecken = [];
  let x = 0;
  const startFest = 420; // Der Anfang ist immer sicher.
  while (x < cfg.breite) {
    const rest = cfg.breite - x;
    let laenge = x === 0 ? startFest : ganz(r, 300, 620);
    if (rest < 700) laenge = rest; // Ziel-Abschnitt am Stueck
    L.boden.push({ x, y: BODEN_Y, w: laenge });
    x += laenge;
    if (cfg.breite - x > 520) {
      const luecke = istSumpf ? ganz(r, 150, 260) : ganz(r, 90, 130);
      luecken.push({ x, w: luecke });
      x += luecke;
    }
  }

  // Sumpf: Luecken mit Wasser fuellen (Schwimm-Dinos im Vorteil).
  if (istSumpf) {
    luecken.slice(0, Math.max(1, cfg.wasser)).forEach((l) => {
      L.wasser.push({ x: l.x, y: BODEN_Y - 10, w: l.w, h: WELT_H - BODEN_Y + 10 });
      // Boden unter Wasser, damit niemand ins Leere faellt
      L.boden.push({ x: l.x, y: WELT_H - 30, w: l.w, unterwasser: true });
      // Treibholz als bewegliche Bruecke
      const anzahl = Math.max(1, Math.round(l.w / 170));
      for (let i = 0; i < anzahl && L.treibhoelzer.length < cfg.treibhoelzer; i += 1) {
        L.treibhoelzer.push({
          x: l.x + 40 + i * 150,
          y: BODEN_Y - 30,
          von: l.x + 20,
          bis: l.x + l.w - 20,
          tempo: zwischen(r, 30, 55),
          richtung: i % 2 === 0 ? 1 : -1,
        });
      }
    });
  }

  // -------------------------------------------------- Plattform-"Treppen"
  // Cluster aus 2-4 Plattformen, die je eine Ebene ansteigen.
  const clusterAnzahl = Math.max(3, Math.round(cfg.plattformen / 3));
  const bereich = cfg.breite - 700;
  for (let ci = 0; ci < clusterAnzahl; ci += 1) {
    const basisX = 380 + (bereich / clusterAnzahl) * ci + zwischen(r, -40, 40);
    const stufen = ganz(r, 2, 4);
    let px = basisX;
    for (let s = 0; s < stufen; s += 1) {
      const ebene = Math.min(EBENEN.length - 1, s);
      const breite = ganz(r, 110, 210);
      const platte = { x: px, y: EBENEN[ebene], w: breite, ebene };
      if (platte.x + platte.w < cfg.breite - 220) L.plattformen.push(platte);
      px += breite + ganz(r, 60, 120);
    }
  }

  const alleFlaechen = () => [
    ...L.plattformen.map((p) => ({ x: p.x, y: p.y, w: p.w })),
    ...L.boden.filter((b) => !b.unterwasser).map((b) => ({ x: b.x, y: b.y, w: b.w })),
  ];

  // ------------------------------------------------------- Biom-Bausteine

  // Broeckelnde Plattformen (Hoehle / Vulkan)
  for (let i = 0; i < cfg.broeckelplattformen; i += 1) {
    const lu = luecken[i % Math.max(1, luecken.length)];
    const px = lu ? lu.x + lu.w / 2 - 32 : zwischen(r, 600, cfg.breite - 600);
    L.broeckel.push({ x: px, y: BODEN_Y - ganz(r, 70, 150), w: 96 });
  }

  // Sprungfedern (Hoehle)
  for (let i = 0; i < cfg.sprungfedern; i += 1) {
    const flaeche = waehle(r, alleFlaechen());
    L.federn.push({ x: flaeche.x + flaeche.w / 2, y: flaeche.y - 14, kraft: 900 });
  }

  // Felsbloecke - dahinter liegt immer ein Bonus-Ei
  for (let i = 0; i < cfg.felsen; i += 1) {
    const b = L.boden.filter((s) => !s.unterwasser && s.w > 360)[
      i % Math.max(1, L.boden.filter((s) => !s.unterwasser && s.w > 360).length)
    ];
    if (!b) break;
    const fx = b.x + 120 + ((i * 173) % Math.max(1, b.w - 220));
    L.felsen.push({ x: fx, y: BODEN_Y - 22 });
    L.felsen.push({ x: fx, y: BODEN_Y - 66 });
    L.eier.push({ x: fx + 54, y: BODEN_Y - 40, bonus: true });
  }

  // Lianen (Dschungel / Sumpf)
  for (let i = 0; i < cfg.lianen; i += 1) {
    const px = 500 + (bereich / Math.max(1, cfg.lianen)) * i + zwischen(r, -60, 60);
    const hoehe = ganz(r, 180, 300);
    L.lianen.push({ x: px, y: BODEN_Y - hoehe, h: hoehe - 20 });
    // Belohnung am oberen Ende
    L.fruechte.push({ x: px, y: BODEN_Y - hoehe - 10, sorte: ganz(r, 0, 2) });
  }

  // Kletterwaende (Dschungel)
  for (let i = 0; i < cfg.kletterwaende; i += 1) {
    const px = 700 + (bereich / Math.max(1, cfg.kletterwaende)) * i;
    L.kletterwaende.push({ x: px, y: BODEN_Y - 300, w: 34, h: 300 });
    L.plattformen.push({ x: px - 90, y: BODEN_Y - 320, w: 150, ebene: 3 });
  }

  // Hebebuehnen (Vulkan / tiefe Hoehle)
  for (let i = 0; i < cfg.hebebuehnen; i += 1) {
    const px = 700 + (bereich / Math.max(1, cfg.hebebuehnen)) * i + zwischen(r, -50, 50);
    const oben = BODEN_Y - ganz(r, 230, 330);
    L.hebebuehnen.push({
      x: px,
      y: BODEN_Y - 60,
      von: oben,
      bis: BODEN_Y - 60,
      tempo: zwischen(r, 45, 70),
      achse: 'y',
    });
    L.fruechte.push({ x: px, y: oben - 40, sorte: ganz(r, 0, 2) });
  }

  // Feuerbaelle (Vulkan)
  for (let i = 0; i < cfg.feuerbaelle; i += 1) {
    const px = 620 + (bereich / Math.max(1, cfg.feuerbaelle)) * i + zwischen(r, -70, 70);
    L.feuerbaelle.push({
      x: px,
      y: BODEN_Y - 10,
      intervall: ganz(r, 1900, 3200),
      kraft: ganz(r, 620, 820),
      verzoegerung: ganz(r, 0, 1500),
    });
  }

  // ----------------------------------------------------------- Sammelzeug
  const flaechen = alleFlaechen().filter((f) => f.x > 240 && f.x < cfg.breite - 240);

  for (let i = 0; i < cfg.eier; i += 1) {
    const f = flaechen[(i * 3 + 1) % Math.max(1, flaechen.length)];
    if (!f) break;
    L.eier.push({ x: f.x + f.w / 2 + zwischen(r, -f.w / 3, f.w / 3), y: f.y - 34 });
  }

  // Fruechte in Bogen ueber den Luecken - laden zum Springen ein
  luecken.forEach((l, i) => {
    if (L.fruechte.length >= cfg.fruechte) return;
    const mitte = l.x + l.w / 2;
    for (let k = -1; k <= 1; k += 1) {
      if (L.fruechte.length >= cfg.fruechte) break;
      L.fruechte.push({
        x: mitte + k * 40,
        y: BODEN_Y - 90 + Math.abs(k) * 26,
        sorte: (i + k + 3) % 3,
      });
    }
  });
  while (L.fruechte.length < cfg.fruechte) {
    const f = flaechen[(L.fruechte.length * 5) % Math.max(1, flaechen.length)];
    if (!f) break;
    L.fruechte.push({ x: f.x + f.w / 2, y: f.y - 40, sorte: L.fruechte.length % 3 });
  }

  // ---------------------------------------------------------------- Gegner
  for (let i = 0; i < cfg.gegner; i += 1) {
    const kandidaten = flaechen.filter((f) => f.w > 150);
    const f = kandidaten[(i * 2 + 1) % Math.max(1, kandidaten.length)];
    if (!f) break;
    const fliegt = gegnerTyp === 'libelle' || (gegnerTyp === 'feuergeist' && i % 2 === 0);
    L.gegner.push({
      typ: gegnerTyp,
      x: f.x + f.w / 2,
      y: f.y - (fliegt ? 90 : 26),
      von: f.x + 24,
      bis: f.x + f.w - 24,
      tempo: 40 + i * 4 + (cfg.nr > 15 ? 20 : 0),
      fliegt,
    });
  }

  // ----------------------------------------------------------- Checkpoints
  for (let i = 1; i <= cfg.checkpoints; i += 1) {
    const ziel = (cfg.breite / (cfg.checkpoints + 1)) * i;
    const b = L.boden
      .filter((s) => !s.unterwasser)
      .reduce((best, s) =>
        Math.abs(s.x + s.w / 2 - ziel) < Math.abs(best.x + best.w / 2 - ziel) ? s : best
      );
    L.checkpoints.push({ x: Math.min(Math.max(ziel, b.x + 60), b.x + b.w - 60), y: BODEN_Y - 32 });
  }

  // ------------------------------------------------------------------ Ziel
  L.luecken = luecken;
  L.sterneMoeglich = 3;
  L.gesamtSammelbar = L.eier.length + L.fruechte.length;
  return L;
}
