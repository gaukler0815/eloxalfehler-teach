/**
 * Minispiel "Dino-Malbuch".
 *
 * Vier Motive aus einfachen Farbflaechen. Farbe antippen, dann eine
 * Flaeche antippen - fertig. Die Bilder werden in localStorage gemerkt.
 */

import { el, karteZeigen, overlayLeeren, toastZeigen } from './Dialog.js';
import Spielstand from '../state/storage.js';

const SVG_NS = 'http://www.w3.org/2000/svg';

export const FARBEN = [
  '#e8505b', '#ff8c42', '#ffd23f', '#8ac926',
  '#1b7f4b', '#34c47c', '#2f7fd1', '#7ee0e8',
  '#8b5cf6', '#d47b9a', '#8a5a2b', '#3a3a4a',
];

/** Ein Motiv besteht aus ausmalbaren Flaechen (`feld`) und festen Details. */
const MOTIVE = [
  {
    id: 'trex',
    name: 'Baby-T-Rex',
    breite: 400,
    hoehe: 300,
    felder: [
      { typ: 'poly', punkte: [[70, 190], [10, 150], [12, 168], [72, 214]] }, // Schwanz
      { typ: 'ellipse', cx: 150, cy: 190, rx: 78, ry: 58 }, // Koerper
      { typ: 'ellipse', cx: 152, cy: 214, rx: 50, ry: 30 }, // Bauch
      { typ: 'poly', punkte: [[122, 246], [104, 292], [140, 292], [150, 246]] }, // Bein hinten
      { typ: 'poly', punkte: [[178, 246], [166, 292], [204, 292], [206, 246]] }, // Bein vorn
      { typ: 'poly', punkte: [[196, 160], [258, 96], [280, 128], [222, 184]] }, // Hals
      { typ: 'ellipse', cx: 292, cy: 112, rx: 56, ry: 44 }, // Kopf
      { typ: 'poly', punkte: [[300, 128], [376, 132], [372, 158], [300, 154]] }, // Schnauze
      { typ: 'poly', punkte: [[216, 172], [258, 186], [256, 200], [214, 188]] }, // Arm
    ],
    details: [
      { typ: 'circle', cx: 300, cy: 96, r: 11, fill: '#ffffff' },
      { typ: 'circle', cx: 304, cy: 97, r: 5, fill: '#1a1a26' },
      { typ: 'poly', punkte: [[306, 154], [312, 166], [318, 154], [326, 166], [332, 154]], fill: '#ffffff' },
    ],
  },
  {
    id: 'triceratops',
    name: 'Triceratops',
    breite: 400,
    hoehe: 300,
    felder: [
      { typ: 'poly', punkte: [[64, 186], [12, 168], [14, 184], [66, 210]] },
      { typ: 'ellipse', cx: 150, cy: 196, rx: 84, ry: 56 },
      { typ: 'ellipse', cx: 152, cy: 220, rx: 54, ry: 28 },
      { typ: 'poly', punkte: [[104, 246], [92, 292], [128, 292], [134, 246]] },
      { typ: 'poly', punkte: [[188, 246], [180, 292], [216, 292], [218, 246]] },
      { typ: 'poly', punkte: [[230, 108], [300, 84], [332, 152], [292, 216], [232, 200]] }, // Nackenschild
      { typ: 'ellipse', cx: 296, cy: 168, rx: 56, ry: 42 }, // Kopf
      { typ: 'poly', punkte: [[318, 178], [382, 186], [378, 208], [316, 200]] }, // Schnabel
      { typ: 'poly', punkte: [[288, 132], [306, 74], [320, 136]] }, // Horn oben
      { typ: 'poly', punkte: [[326, 152], [376, 128], [336, 172]] }, // Horn vorn
    ],
    details: [
      { typ: 'circle', cx: 306, cy: 158, r: 10, fill: '#ffffff' },
      { typ: 'circle', cx: 310, cy: 159, r: 4.5, fill: '#1a1a26' },
    ],
  },
  {
    id: 'stegosaurus',
    name: 'Stegosaurus',
    breite: 400,
    hoehe: 300,
    felder: [
      { typ: 'poly', punkte: [[62, 196], [8, 172], [10, 190], [64, 218]] },
      { typ: 'ellipse', cx: 158, cy: 200, rx: 90, ry: 54 },
      { typ: 'poly', punkte: [[92, 148], [112, 96], [134, 148]] }, // Platte 1
      { typ: 'poly', punkte: [[128, 142], [152, 78], [176, 142]] }, // Platte 2
      { typ: 'poly', punkte: [[170, 142], [196, 74], [220, 142]] }, // Platte 3
      { typ: 'poly', punkte: [[212, 146], [236, 88], [258, 146]] }, // Platte 4
      { typ: 'poly', punkte: [[116, 248], [104, 292], [138, 292], [144, 248]] },
      { typ: 'poly', punkte: [[196, 248], [190, 292], [224, 292], [226, 248]] },
      { typ: 'poly', punkte: [[240, 176], [300, 158], [306, 190], [246, 204]] }, // Hals
      { typ: 'ellipse', cx: 316, cy: 176, rx: 40, ry: 28 }, // Kopf
    ],
    details: [
      { typ: 'circle', cx: 322, cy: 168, r: 9, fill: '#ffffff' },
      { typ: 'circle', cx: 325, cy: 169, r: 4, fill: '#1a1a26' },
      { typ: 'poly', punkte: [[52, 190], [24, 150], [44, 196]], fill: '#f5efdc' },
      { typ: 'poly', punkte: [[70, 196], [48, 152], [62, 200]], fill: '#f5efdc' },
    ],
  },
  {
    id: 'nest',
    name: 'Nest mit Ei',
    breite: 400,
    hoehe: 300,
    felder: [
      { typ: 'circle', cx: 330, cy: 62, r: 40 }, // Sonne
      { typ: 'ellipse', cx: 92, cy: 74, rx: 46, ry: 26 }, // Wolke
      { typ: 'poly', punkte: [[0, 250], [400, 250], [400, 300], [0, 300]] }, // Boden
      { typ: 'ellipse', cx: 200, cy: 244, rx: 118, ry: 42 }, // Nest
      { typ: 'ellipse', cx: 200, cy: 186, rx: 56, ry: 70 }, // Grosses Ei
      { typ: 'ellipse', cx: 122, cy: 224, rx: 30, ry: 38 }, // Kleines Ei links
      { typ: 'ellipse', cx: 278, cy: 224, rx: 30, ry: 38 }, // Kleines Ei rechts
      { typ: 'poly', punkte: [[30, 250], [58, 186], [86, 250]] }, // Farn / Busch
      { typ: 'poly', punkte: [[320, 250], [350, 180], [380, 250]] },
    ],
    details: [],
  },
];

export function malbuchOeffnen(onFertig) {
  let motivIndex = 0;
  let farbe = FARBEN[0];

  const anzeigen = () => {
    const motiv = MOTIVE[motivIndex];
    const gespeichert = Spielstand.malbuchLaden(motiv.id) || {};

    const svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${motiv.breite} ${motiv.hoehe}`);
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');

    const aktuelleFarben = { ...gespeichert };

    const formErzeugen = (f, fuellung, klasse) => {
      let node;
      if (f.typ === 'ellipse') {
        node = document.createElementNS(SVG_NS, 'ellipse');
        node.setAttribute('cx', f.cx);
        node.setAttribute('cy', f.cy);
        node.setAttribute('rx', f.rx);
        node.setAttribute('ry', f.ry);
      } else if (f.typ === 'circle') {
        node = document.createElementNS(SVG_NS, 'circle');
        node.setAttribute('cx', f.cx);
        node.setAttribute('cy', f.cy);
        node.setAttribute('r', f.r);
      } else {
        node = document.createElementNS(SVG_NS, 'polygon');
        node.setAttribute('points', f.punkte.map((p) => p.join(',')).join(' '));
      }
      node.setAttribute('fill', fuellung);
      node.setAttribute('stroke', '#23303d');
      node.setAttribute('stroke-width', '3');
      node.setAttribute('stroke-linejoin', 'round');
      if (klasse) node.setAttribute('class', klasse);
      return node;
    };

    motiv.felder.forEach((f, i) => {
      const node = formErzeugen(f, aktuelleFarben[i] || '#ffffff', 'feld');
      const malen = (ev) => {
        ev.preventDefault();
        aktuelleFarben[i] = farbe;
        node.setAttribute('fill', farbe);
        Spielstand.malbuchSpeichern(motiv.id, aktuelleFarben);
      };
      node.addEventListener('click', malen);
      node.addEventListener('touchstart', malen, { passive: false });
      svg.appendChild(node);
    });

    motiv.details.forEach((d) => {
      const node = formErzeugen(d, d.fill || '#1a1a26', null);
      node.setAttribute('stroke-width', '1.5');
      node.setAttribute('pointer-events', 'none');
      svg.appendChild(node);
    });

    const palette = el(
      'div',
      { class: 'farbpalette' },
      FARBEN.map((f) =>
        el('button', {
          type: 'button',
          class: f === farbe ? 'aktiv' : '',
          style: { background: f },
          title: f,
          onClick: (ev) => {
            farbe = f;
            ev.currentTarget.parentElement
              .querySelectorAll('button')
              .forEach((b) => b.classList.remove('aktiv'));
            ev.currentTarget.classList.add('aktiv');
          },
        })
      )
    );

    karteZeigen({
      titel: `🎨 Dino-Malbuch: ${motiv.name}`,
      untertitel: 'Wähle eine Farbe und tippe dann auf eine Fläche.',
      breit: true,
      inhalt: [el('div', { class: 'malbuch-flaeche' }, [svg]), palette],
      knoepfe: [
        {
          text: '➜ Nächstes Bild',
          klasse: 'blau',
          onClick: () => {
            motivIndex = (motivIndex + 1) % MOTIVE.length;
            anzeigen();
          },
        },
        {
          text: '🧽 Bild leeren',
          klasse: 'gelb',
          onClick: () => {
            Spielstand.malbuchSpeichern(motiv.id, {});
            anzeigen();
          },
        },
        {
          text: '✔ Fertig',
          klasse: 'grau',
          onClick: () => {
            overlayLeeren();
            toastZeigen('Schönes Bild! 🎨');
            onFertig?.();
          },
        },
      ],
    });
  };

  anzeigen();
}

export const ANZAHL_MOTIVE = MOTIVE.length;
