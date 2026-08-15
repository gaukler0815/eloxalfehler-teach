/**
 * Level-Definitionen: 30 Level in 4 Urzeit-Biomen.
 *
 * Jede Definition beschreibt nur die "Zutaten" eines Levels.
 * Die konkrete Geometrie baut src/game/levelGenerator.js daraus
 * deterministisch (per Seed) zusammen - gleicher Seed = gleiches Level.
 */

export const BIOME = {
  jungle: {
    id: 'jungle',
    name: 'Urzeit-Dschungel',
    level: [1, 8],
    farben: {
      himmel: ['#8fd6ff', '#d8f6c4'],
      fern: '#8fbf7a',
      mittel: '#5f9e57',
      nah: '#3d7a41',
      boden: '#6b4a2b',
      bodenOben: '#4f9a3e',
      plattform: '#7a5433',
      plattformOben: '#5cb04a',
      deko: '#2f5d33',
    },
    features: ['lianen', 'kletterwand'],
  },
  swamp: {
    id: 'swamp',
    name: 'Sumpf & Mangroven',
    level: [9, 15],
    farben: {
      himmel: ['#bfe3d6', '#e7f2c8'],
      fern: '#8aa98d',
      mittel: '#5f8b6c',
      nah: '#3d6650',
      boden: '#4a3b2a',
      bodenOben: '#6f8f4a',
      plattform: '#5b452e',
      plattformOben: '#7d9a4d',
      deko: '#2c4a38',
    },
    features: ['wasser', 'treibholz', 'lianen'],
  },
  cave: {
    id: 'cave',
    name: 'Kristallhoehlen & Felsen',
    level: [16, 22],
    farben: {
      himmel: ['#2b2a4a', '#4b3f6b'],
      fern: '#3b3760',
      mittel: '#4a4472',
      nah: '#332f52',
      boden: '#514c66',
      bodenOben: '#6f6a8c',
      plattform: '#5a5473',
      plattformOben: '#8f88b5',
      deko: '#7ee0e8',
    },
    features: ['sprungfeder', 'broeckel', 'felsen'],
  },
  volcano: {
    id: 'volcano',
    name: 'Vulkanland',
    level: [23, 30],
    farben: {
      himmel: ['#3a1c2a', '#c25a2b'],
      fern: '#5c2b30',
      mittel: '#77332f',
      nah: '#4a1f22',
      boden: '#3a2320',
      bodenOben: '#6b3a2c',
      plattform: '#4a2a24',
      plattformOben: '#8a4630',
      deko: '#ff8c42',
    },
    features: ['feuerbaelle', 'hebebuehne', 'rauch', 'broeckel'],
  },
};

export function biomeFuerLevel(nr) {
  if (nr <= 8) return BIOME.jungle;
  if (nr <= 15) return BIOME.swamp;
  if (nr <= 22) return BIOME.cave;
  return BIOME.volcano;
}

// [Name, Breite in px, Gegner, Eier, Fruechte, Zielzeit in Sekunden]
const SPEC = [
  ['Erste Schritte', 2600, 2, 6, 5, 75],
  ['Farnwald', 2900, 3, 7, 6, 80],
  ['Lianen-Schaukel', 3100, 3, 8, 6, 85],
  ['Der hohle Baum', 3300, 4, 8, 7, 90],
  ['Wasserfall-Pfad', 3500, 4, 9, 7, 95],
  ['Nest im Geaest', 3600, 5, 9, 8, 95],
  ['Kaefer-Schlucht', 3800, 5, 10, 8, 100],
  ['Am Rand des Dschungels', 4000, 6, 10, 9, 105],

  ['Die erste Pfuetze', 3400, 4, 8, 7, 95],
  ['Mangrovenwurzeln', 3700, 5, 9, 7, 100],
  ['Treibholz-Fahrt', 3900, 5, 9, 8, 105],
  ['Tiefes Wasser', 4100, 6, 10, 8, 110],
  ['Nebelmoor', 4300, 6, 10, 9, 115],
  ['Libellen-Teich', 4400, 7, 11, 9, 115],
  ['Der Sumpfkoenig', 4600, 7, 12, 10, 120],

  ['Dunkler Eingang', 3800, 5, 9, 8, 105],
  ['Kristallgang', 4000, 6, 10, 8, 110],
  ['Sprungfeder-Halle', 4200, 6, 10, 9, 115],
  ['Broeckelbruecke', 4400, 7, 11, 9, 120],
  ['Tropfsteinsee', 4600, 7, 11, 10, 125],
  ['Der grosse Schacht', 4800, 8, 12, 10, 130],
  ['Das Kristallherz', 5000, 8, 12, 11, 135],

  ['Ascheweg', 4200, 6, 10, 9, 115],
  ['Feuerbaelle', 4400, 7, 10, 9, 120],
  ['Hebebuehnen', 4600, 7, 11, 10, 125],
  ['Am Lavasee', 4800, 8, 11, 10, 130],
  ['Rauchschlucht', 5000, 8, 12, 11, 135],
  ['Der gluehende Grat', 5200, 9, 12, 11, 140],
  ['Im Vulkankrater', 5400, 9, 13, 12, 145],
  ['Rexis grosses Finale', 5800, 10, 14, 12, 155],
];

/**
 * Baut aus SPEC + Biom die 30 Level-Konfigurationen.
 * Die Feature-Mengen wachsen mit der Levelnummer innerhalb des Bioms.
 */
export const LEVELS = SPEC.map(([name, breite, gegner, eier, fruechte, zeit], i) => {
  const nr = i + 1;
  const biom = biomeFuerLevel(nr);
  const [von, bis] = biom.level;
  // 0 = erstes Level des Bioms, 1 = letztes Level des Bioms
  const t = (nr - von) / Math.max(1, bis - von);
  const stufe = (basis, extra) => Math.round(basis + extra * t);

  const cfg = {
    nr,
    name,
    biom: biom.id,
    biomName: biom.name,
    seed: 1000 + nr * 977,
    breite,
    hoehe: 720,
    gegner,
    eier,
    fruechte,
    zielzeit: zeit,
    // Standard-Bausteine, in jedem Biom vorhanden
    plattformen: stufe(9 + Math.floor(nr / 3), 4),
    checkpoints: nr <= 4 ? 1 : 2,
    // Biom-Features (0 = kommt hier nicht vor)
    lianen: 0,
    kletterwaende: 0,
    wasser: 0,
    treibhoelzer: 0,
    sprungfedern: 0,
    broeckelplattformen: 0,
    felsen: 0,
    feuerbaelle: 0,
    hebebuehnen: 0,
    rauch: false,
  };

  switch (biom.id) {
    case 'jungle':
      cfg.lianen = stufe(2, 3);
      cfg.kletterwaende = nr >= 3 ? stufe(1, 2) : 0;
      cfg.felsen = nr >= 5 ? 2 : 0;
      break;
    case 'swamp':
      cfg.wasser = stufe(1, 2);
      cfg.treibhoelzer = stufe(2, 3);
      cfg.lianen = stufe(1, 2);
      cfg.felsen = 2;
      break;
    case 'cave':
      cfg.sprungfedern = stufe(2, 3);
      cfg.broeckelplattformen = stufe(3, 4);
      cfg.felsen = stufe(3, 3);
      cfg.hebebuehnen = nr >= 19 ? stufe(1, 2) : 0;
      break;
    case 'volcano':
      cfg.feuerbaelle = stufe(2, 4);
      cfg.hebebuehnen = stufe(2, 3);
      cfg.broeckelplattformen = stufe(2, 3);
      cfg.felsen = stufe(2, 3);
      cfg.rauch = true;
      break;
    default:
      break;
  }
  return cfg;
});

export function getLevel(nr) {
  return LEVELS[Math.max(0, Math.min(LEVELS.length - 1, nr - 1))];
}

export const ANZAHL_LEVEL = LEVELS.length;
