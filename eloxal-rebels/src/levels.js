/*
 * levels.js — level catalogue + loader.
 *
 * Level content lives in levels/*.json (the canonical source, also what the
 * editor exports). Because the game must run by double-clicking index.html,
 * where fetch() of local JSON is blocked by the browser, the same data is
 * embedded here. When served over http(s) the loader can additionally fetch
 * fresh JSON. Keep the embedded copies in sync with the JSON files.
 */
(function (global) {
  'use strict';
  const ER = (global.ER = global.ER || {});

  // Embedded mirror of levels/*.json, in play order.
  const EMBEDDED = [
    {
      id: 'w1-l01', world: 1, name: 'Wareneingang',
      subtitle: 'Ziehen, zielen, loslassen. Räum die Palette ab.',
      slingshot: { x: 340, y: 720 },
      projectiles: ['ali', 'ali', 'ali'],
      blocks: [
        { id: 'b1', material: 'kartonage', x: 1360, y: 940, w: 150, h: 90 },
        { id: 'b2', material: 'kartonage', x: 1360, y: 850, w: 150, h: 90 },
        { id: 'b3', material: 'kartonage', x: 1520, y: 940, w: 150, h: 90 }
      ],
      enemies: [
        { id: 'e1', type: 'stauber', x: 1360, y: 785 },
        { id: 'e2', type: 'stauber', x: 1520, y: 875 }
      ]
    },
    {
      id: 'w1-l02', world: 1, name: 'Palettenturm',
      subtitle: 'Bolle antippen im Flug: Sturzflug und Durchschlag.',
      slingshot: { x: 340, y: 720 },
      projectiles: ['bolle', 'ali', 'ali'],
      blocks: [
        { id: 'c1', material: 'aluminium', x: 1480, y: 895, w: 44, h: 170 },
        { id: 'c2', material: 'aluminium', x: 1680, y: 895, w: 44, h: 170 },
        { id: 'beam', material: 'aluminium', x: 1580, y: 795, w: 260, h: 40 },
        { id: 'roof', material: 'fehlcharge', x: 1580, y: 745, w: 200, h: 36 }
      ],
      enemies: [
        { id: 'e1', type: 'lenny', x: 1580, y: 700 },
        { id: 'e2', type: 'stauber', x: 1680, y: 780 }
      ]
    },
    {
      id: 'w1-l03', world: 1, name: 'Sprödes Dach',
      subtitle: 'Fehlcharge-Eloxal zersplittert. Ein Säurefass hilft nach.',
      slingshot: { x: 340, y: 720 },
      projectiles: ['bolle', 'ali', 'ali'],
      blocks: [
        { id: 'base', material: 'stahl', x: 1520, y: 950, w: 320, h: 40 },
        { id: 'l', material: 'fehlcharge', x: 1400, y: 880, w: 40, h: 100 },
        { id: 'r', material: 'fehlcharge', x: 1640, y: 880, w: 40, h: 100 },
        { id: 'roof', material: 'fehlcharge', x: 1520, y: 815, w: 300, h: 40 },
        { id: 'barrel', material: 'saeurefass', x: 1520, y: 892, w: 70, h: 70 }
      ],
      enemies: [
        { id: 'e1', type: 'kalki', x: 1440, y: 895 },
        { id: 'e2', type: 'lenny', x: 1600, y: 895 }
      ]
    },
    {
      id: 'w5-l01', world: 5, name: 'Kontaktstelle',
      subtitle: 'Bürsti räumt die isolierende Lage weg, das Blech fällt auf den Stahl. Dann die Schiene treffen: ein Lichtbogen, drei Gegner.',
      slingshot: { x: 340, y: 720 },
      projectiles: ['buersti', 'ali', 'ali'],
      blocks: [
        { id: 'rail', material: 'rail', x: 1650, y: 958, w: 150, h: 26 },
        { id: 'steel', material: 'stahl', x: 1650, y: 855, w: 120, h: 170 },
        { id: 'spacer', material: 'kartonage', x: 1650, y: 748, w: 220, h: 44 },
        { id: 'plate', material: 'aluminium', x: 1650, y: 705, w: 160, h: 30 }
      ],
      enemies: [
        { id: 'e1', type: 'lenny', x: 1600, y: 655 },
        { id: 'e2', type: 'kalki', x: 1700, y: 655 },
        { id: 'e3', type: 'stauber', x: 1560, y: 918 }
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

  // Optional: when running over http(s), refresh a level from its JSON file.
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
})(typeof window !== 'undefined' ? window : globalThis);
