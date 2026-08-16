/*
 * levels.js — the campaign: 12 levels from tutorial to boss fight, grouped in
 * three worlds along the process chain. Difficulty rises by introducing one
 * idea per level (design bible rule): materials, one new rebel at a time,
 * bank shots, top-only entries, conductivity puzzles, chain reactions and
 * finally Baron Korrosius. Later levels use wider worlds (`width`), which the
 * camera handles by zooming out while aiming.
 *
 * This embedded catalogue is the runtime source (fetch() of local JSON is
 * blocked under file://). The JSON files in levels/ are generated from it —
 * see tests/levels.test.js which also validates every level.
 *
 * Geometry notes: ground top is y=980. A block of height h rests at
 * y = 980 - h/2; a stack adds half-heights. Enemies (radius r) rest at
 * top-of-support minus r.
 */
(function (global) {
  'use strict';
  const ER = (global.ER = global.ER || {});

  const EMBEDDED = [
    // ---------------------------------------------------------- Welt 1: Wareneingang
    {
      id: 'l01', world: 1, name: 'Wareneingang',
      subtitle: 'Ziehen, zielen, loslassen. Räum die Palette ab.',
      width: 1920,
      slingshot: { x: 340, y: 720 },
      projectiles: ['ali', 'ali', 'ali'],
      blocks: [
        { id: 'b1', material: 'kartonage', x: 1360, y: 935, w: 150, h: 90 },
        { id: 'b2', material: 'kartonage', x: 1360, y: 845, w: 150, h: 90 },
        { id: 'b3', material: 'kartonage', x: 1520, y: 935, w: 150, h: 90 }
      ],
      enemies: [
        { id: 'e1', type: 'stauber', x: 1360, y: 776 },
        { id: 'e2', type: 'stauber', x: 1520, y: 866 }
      ]
    },
    {
      id: 'l02', world: 1, name: 'Palettenturm',
      subtitle: 'Bolle antippen im Flug: Sturzflug und Durchschlag.',
      width: 1920,
      slingshot: { x: 340, y: 720 },
      projectiles: ['bolle', 'ali', 'ali'],
      blocks: [
        { id: 'box', material: 'kartonage', x: 1300, y: 935, w: 130, h: 90 },
        { id: 'c1', material: 'aluminium', x: 1480, y: 895, w: 44, h: 170 },
        { id: 'c2', material: 'aluminium', x: 1680, y: 895, w: 44, h: 170 },
        { id: 'beam', material: 'aluminium', x: 1580, y: 790, w: 260, h: 40 },
        { id: 'roof', material: 'fehlcharge', x: 1580, y: 752, w: 200, h: 36 }
      ],
      enemies: [
        { id: 'e1', type: 'stauber', x: 1300, y: 866 },
        { id: 'e2', type: 'lenny', x: 1580, y: 704 },
        { id: 'e3', type: 'stauber', x: 1790, y: 956 }
      ]
    },
    {
      id: 'l03', world: 1, name: 'Sprödes Dach',
      subtitle: 'Fehlcharge-Eloxal zersplittert. Das Säurefass hilft nach.',
      width: 1920,
      slingshot: { x: 340, y: 720 },
      projectiles: ['bolle', 'ali', 'ali'],
      blocks: [
        { id: 'base', material: 'stahl', x: 1520, y: 960, w: 320, h: 40 },
        { id: 'l', material: 'fehlcharge', x: 1400, y: 890, w: 40, h: 100 },
        { id: 'r', material: 'fehlcharge', x: 1640, y: 890, w: 40, h: 100 },
        { id: 'roof', material: 'fehlcharge', x: 1520, y: 820, w: 300, h: 40 },
        { id: 'barrel', material: 'saeurefass', x: 1520, y: 905, w: 70, h: 70 }
      ],
      enemies: [
        { id: 'e1', type: 'kalki', x: 1455, y: 910 },
        { id: 'e2', type: 'lenny', x: 1590, y: 910 },
        { id: 'e3', type: 'stauber', x: 1520, y: 776 }
      ]
    },
    {
      id: 'l04', world: 1, name: 'Hinterm Schild',
      subtitle: 'Der Stahl blockt flache Schüsse. Hoch drüber — Rippi teilt sich auf.',
      width: 1920,
      slingshot: { x: 340, y: 720 },
      projectiles: ['rippi', 'ali', 'ali'],
      blocks: [
        { id: 'shield', material: 'stahl', x: 1180, y: 860, w: 44, h: 240 },
        { id: 'ramp', material: 'kunststoff', x: 1500, y: 470, w: 320, h: 30, angle: -32, static: true },
        { id: 'box', material: 'kartonage', x: 1560, y: 935, w: 120, h: 90 }
      ],
      enemies: [
        { id: 'e1', type: 'stauber', x: 1300, y: 956 },
        { id: 'e2', type: 'stauber', x: 1420, y: 956 },
        { id: 'e3', type: 'fetti', x: 1560, y: 862 },
        { id: 'e4', type: 'stauber', x: 1700, y: 956 }
      ]
    },
    // ---------------------------------------------------------- Welt 2: Beckenreihe
    {
      id: 'l05', world: 2, name: 'Kipplast',
      subtitle: 'Titania hakt sich an Kanten fest und reißt sie herunter.',
      width: 2400,
      slingshot: { x: 320, y: 720 },
      projectiles: ['titania', 'bolle', 'ali'],
      blocks: [
        { id: 'barrel', material: 'saeurefass', x: 1600, y: 945, w: 70, h: 70 },
        { id: 't1a', material: 'aluminium', x: 1750, y: 850, w: 50, h: 260 },
        { id: 't1b', material: 'aluminium', x: 1900, y: 850, w: 50, h: 260 },
        { id: 'cap', material: 'aluminium', x: 1825, y: 700, w: 220, h: 40 },
        { id: 's1', material: 'aluminium', x: 2150, y: 850, w: 50, h: 260 },
        { id: 'cap2', material: 'aluminium', x: 2150, y: 700, w: 180, h: 40 }
      ],
      enemies: [
        { id: 'e1', type: 'lenny', x: 1825, y: 650 },
        { id: 'e2', type: 'stauber', x: 2150, y: 656 },
        { id: 'e3', type: 'lenny', x: 2020, y: 950 },
        { id: 'e4', type: 'stauber', x: 2290, y: 956 }
      ]
    },
    {
      id: 'l06', world: 2, name: 'Unterm Regal',
      subtitle: 'Bubbles fliegt tief hinein — Tap: sie steigt und zündet an der Decke.',
      width: 2400,
      slingshot: { x: 320, y: 720 },
      projectiles: ['bubbles', 'ali', 'ali'],
      blocks: [
        { id: 'colL', material: 'stahl', x: 1900, y: 880, w: 44, h: 200 },
        { id: 'colR', material: 'stahl', x: 2240, y: 880, w: 44, h: 200 },
        { id: 'shelf', material: 'stahl', x: 2070, y: 763, w: 400, h: 34 },
        { id: 'topbox', material: 'fehlcharge', x: 2000, y: 711, w: 120, h: 70 }
      ],
      enemies: [
        { id: 'e1', type: 'fetti', x: 2000, y: 952 },
        { id: 'e2', type: 'stauber', x: 2090, y: 956 },
        { id: 'e3', type: 'stauber', x: 2170, y: 956 },
        { id: 'e4', type: 'kalki', x: 2180, y: 716 }
      ]
    },
    {
      id: 'l07', world: 2, name: 'Säurebad',
      subtitle: 'Stahl ist mit Wucht allein nicht zu knacken. Erst Säuri, dann Bolle.',
      width: 2400,
      slingshot: { x: 320, y: 720 },
      projectiles: ['saeuri', 'bolle', 'ali'],
      blocks: [
        { id: 'g1', material: 'stahl', x: 1850, y: 880, w: 44, h: 200 },
        { id: 'g2', material: 'stahl', x: 2050, y: 880, w: 44, h: 200 },
        { id: 'g3', material: 'stahl', x: 2250, y: 880, w: 44, h: 200 },
        { id: 'floor1', material: 'stahl', x: 2050, y: 762, w: 460, h: 36 },
        { id: 'u1', material: 'stahl', x: 1950, y: 654, w: 44, h: 180 },
        { id: 'u2', material: 'stahl', x: 2150, y: 654, w: 44, h: 180 },
        { id: 'roof', material: 'stahl', x: 2050, y: 546, w: 300, h: 36 },
        { id: 'barrel', material: 'saeurefass', x: 2350, y: 945, w: 70, h: 70 }
      ],
      enemies: [
        { id: 'e1', type: 'lenny', x: 1950, y: 950 },
        { id: 'e2', type: 'kalki', x: 2150, y: 950 },
        { id: 'e3', type: 'lenny', x: 2050, y: 714 },
        { id: 'e4', type: 'stauber', x: 2050, y: 504 }
      ]
    },
    {
      id: 'l08', world: 2, name: 'Kontaktstelle',
      subtitle: 'Bürsti bläst die Isolierung weg, das Blech fällt auf den Stahl. Dann die Schiene: ein Lichtbogen, drei Gegner.',
      width: 2400,
      slingshot: { x: 320, y: 720 },
      projectiles: ['buersti', 'ali', 'ali'],
      blocks: [
        { id: 'rail', material: 'rail', x: 2150, y: 958, w: 150, h: 26 },
        { id: 'steel', material: 'stahl', x: 2150, y: 860, w: 120, h: 170 },
        { id: 'spacer', material: 'kartonage', x: 2150, y: 753, w: 220, h: 44 },
        { id: 'plate', material: 'aluminium', x: 2150, y: 716, w: 160, h: 30 }
      ],
      enemies: [
        { id: 'e1', type: 'lenny', x: 2100, y: 671 },
        { id: 'e2', type: 'kalki', x: 2200, y: 671 },
        { id: 'e3', type: 'stauber', x: 2060, y: 956 }
      ]
    },
    // ---------------------------------------------------------- Welt 3: Endspurt
    {
      id: 'l09', world: 3, name: 'Kettenreaktion',
      subtitle: 'Fehlcharge splittert, Fässer zünden Fässer. Ein Treffer reicht — der richtige.',
      width: 2800,
      slingshot: { x: 320, y: 720 },
      projectiles: ['ali', 'rippi', 'bolle'],
      blocks: [
        { id: 'gA', material: 'fehlcharge', x: 1900, y: 910, w: 40, h: 140 },
        { id: 'gB', material: 'fehlcharge', x: 2050, y: 910, w: 40, h: 140 },
        { id: 'gC', material: 'fehlcharge', x: 2200, y: 910, w: 40, h: 140 },
        { id: 'gD', material: 'fehlcharge', x: 2350, y: 910, w: 40, h: 140 },
        { id: 'deck', material: 'fehlcharge', x: 2125, y: 825, w: 560, h: 30 },
        { id: 'f1', material: 'saeurefass', x: 1975, y: 775, w: 70, h: 70 },
        { id: 'f2', material: 'saeurefass', x: 2275, y: 775, w: 70, h: 70 },
        { id: 'f3', material: 'saeurefass', x: 2540, y: 945, w: 70, h: 70 }
      ],
      enemies: [
        { id: 'e1', type: 'stauber', x: 1975, y: 956 },
        { id: 'e2', type: 'stauber', x: 2125, y: 956 },
        { id: 'e3', type: 'fetti', x: 2275, y: 952 },
        { id: 'e4', type: 'lenny', x: 2450, y: 950 },
        { id: 'e5', type: 'stauber', x: 2650, y: 956 }
      ]
    },
    {
      id: 'l10', world: 3, name: 'Doppelfestung',
      subtitle: 'Zwei Festungen, vier Geschosse. Teile gut ein.',
      width: 2800,
      slingshot: { x: 320, y: 720 },
      projectiles: ['titania', 'rippi', 'bolle', 'ali'],
      blocks: [
        { id: 'a1', material: 'aluminium', x: 1650, y: 880, w: 44, h: 200 },
        { id: 'a2', material: 'aluminium', x: 1800, y: 880, w: 44, h: 200 },
        { id: 'aRoof', material: 'fehlcharge', x: 1725, y: 762, w: 220, h: 36 },
        { id: 'b1', material: 'stahl', x: 2300, y: 870, w: 44, h: 220 },
        { id: 'b2', material: 'stahl', x: 2500, y: 870, w: 44, h: 220 },
        { id: 'bBeam', material: 'stahl', x: 2400, y: 742, w: 280, h: 36 },
        { id: 'bUp1', material: 'aluminium', x: 2400, y: 644, w: 44, h: 160 },
        { id: 'bRoof', material: 'aluminium', x: 2400, y: 549, w: 200, h: 30 },
        { id: 'barrel', material: 'saeurefass', x: 2400, y: 945, w: 70, h: 70 }
      ],
      enemies: [
        { id: 'e1', type: 'stauber', x: 1725, y: 956 },
        { id: 'e2', type: 'lenny', x: 1725, y: 714 },
        { id: 'e3', type: 'stauber', x: 2200, y: 956 },
        { id: 'e4', type: 'kalki', x: 2450, y: 950 },
        { id: 'e5', type: 'stauber', x: 2400, y: 700 },
        { id: 'e6', type: 'fetti', x: 2640, y: 952 }
      ]
    },
    {
      id: 'l11', world: 3, name: 'Laserschnitt',
      subtitle: 'Lasar friert den Flug ein — zieh die Linie, die den Bunker öffnet. Nur einmal.',
      width: 2800,
      slingshot: { x: 320, y: 720 },
      projectiles: ['lasar', 'bolle', 'ali'],
      blocks: [
        { id: 'k1', material: 'stahl', x: 2200, y: 860, w: 60, h: 240 },
        { id: 'k2', material: 'stahl', x: 2450, y: 860, w: 60, h: 240 },
        { id: 'kRoof', material: 'stahl', x: 2325, y: 720, w: 380, h: 40 },
        { id: 'inner', material: 'fehlcharge', x: 2325, y: 940, w: 120, h: 80 },
        { id: 'guardbox', material: 'kartonage', x: 2000, y: 935, w: 120, h: 90 }
      ],
      enemies: [
        { id: 'e1', type: 'lenny', x: 2000, y: 860 },
        { id: 'e2', type: 'kalki', x: 2270, y: 948 },
        { id: 'e3', type: 'kalki', x: 2390, y: 948 },
        { id: 'e4', type: 'lenny', x: 2325, y: 670 },
        { id: 'e5', type: 'stauber', x: 2620, y: 956 }
      ]
    },
    {
      id: 'l12', world: 3, name: 'Das letzte Becken',
      subtitle: 'Baron Korrosius braucht drei Treffer — sein Kunststoff-Thron isoliert. Die Wachen auf dem Stahl nicht.',
      width: 3000,
      slingshot: { x: 320, y: 720 },
      projectiles: ['ali', 'bolle', 'titania', 'buersti', 'saeuri'],
      blocks: [
        { id: 'fass1', material: 'saeurefass', x: 2200, y: 945, w: 70, h: 70 },
        { id: 'base', material: 'stahl', x: 2550, y: 960, w: 400, h: 40 },
        { id: 'fass2', material: 'saeurefass', x: 2550, y: 905, w: 70, h: 70 },
        { id: 'c1', material: 'stahl', x: 2400, y: 830, w: 50, h: 220 },
        { id: 'c2', material: 'stahl', x: 2700, y: 830, w: 50, h: 220 },
        { id: 'platform', material: 'stahl', x: 2550, y: 700, w: 420, h: 40 },
        { id: 'thron', material: 'kunststoff', x: 2550, y: 650, w: 160, h: 60 },
        { id: 'post', material: 'stahl', x: 2870, y: 855, w: 60, h: 180 },
        { id: 'rail', material: 'rail', x: 2870, y: 958, w: 160, h: 26 }
      ],
      enemies: [
        { id: 'boss', type: 'korrosius', x: 2550, y: 566 },
        { id: 'e1', type: 'kalki', x: 2440, y: 650 },
        { id: 'e2', type: 'lenny', x: 2660, y: 650 },
        { id: 'e3', type: 'stauber', x: 2300, y: 956 },
        { id: 'e4', type: 'fetti', x: 2480, y: 912 },
        { id: 'e5', type: 'lenny', x: 2870, y: 735 },
        { id: 'e6', type: 'stauber', x: 2100, y: 956 }
      ]
    }
  ];

  const byId = {};
  EMBEDDED.forEach((l) => (byId[l.id] = l));
  const order = EMBEDDED.map((l) => l.id);

  // Deep clone so a play session never mutates the catalogue.
  function clone(obj) {
    return JSON.parse(JSON.stringify(obj));
  }

  function get(id) {
    return byId[id] ? clone(byId[id]) : null;
  }

  function nextId(id) {
    const i = order.indexOf(id);
    return i >= 0 && i < order.length - 1 ? order[i + 1] : null;
  }

  // Optional: when served over http(s), refresh a level from its JSON file.
  async function fetchJson(id) {
    try {
      const res = await fetch('levels/' + id + '.json', { cache: 'no-store' });
      if (res.ok) return await res.json();
    } catch (e) {
      /* file:// or offline — fall back to embedded */
    }
    return get(id);
  }

  ER.levels = { order, get, nextId, fetchJson, all: () => EMBEDDED.map(clone) };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = ER.levels; // Node: tests + JSON generation
  }
})(typeof window !== 'undefined' ? window : globalThis);
