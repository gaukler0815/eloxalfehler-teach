/**
 * Baut ein Artefakt als 3D-wirkendes Schmuckstück - ganz ohne Bilddatei.
 *
 * Die Tiefe entsteht aus mehreren übereinander liegenden Ebenen:
 * Rahmen, Körper mit Metallverlauf, vertiefter Kern, wanderndes Glanzlicht
 * und ein Schlagschatten. Das Symbol steht per translateZ wirklich vor dem
 * Körper, dadurch wirkt die leichte Drehung räumlich.
 */

import { el } from './dom.js';
import { artefaktFuer } from '../data/artefakte.js';

/** Form und Farben je Welt. */
const STIL = {
  dinos: { form: 'form-hexagon', hell: '#6fe0a0', dunkel: '#0f5537' },
  tiere: { form: 'form-kreis', hell: '#ffc07a', dunkel: '#9d4a12' },
  natur: { form: 'form-kristall', hell: '#b6ec7c', dunkel: '#2f6b2a' },
  weltraum: { form: 'form-orbit', hell: '#8fd0ff', dunkel: '#123f77' },
};

/**
 * @param {object} level  Eintrag aus LEVEL
 * @param {object} opt
 * @param {number} [opt.sterne]   3 = goldener Rahmen
 * @param {boolean} [opt.gesperrt]
 * @param {number} [opt.groesse]  Kantenlänge in px
 */
export function artefaktBild(level, { sterne = 3, gesperrt = false, groesse = 180 } = {}) {
  const stil = STIL[level.welt.id] || STIL.dinos;
  const daten = artefaktFuer(level);

  const klassen = ['artefakt', stil.form];
  if (sterne >= 3 && !gesperrt) klassen.push('dreisterne');
  if (gesperrt) klassen.push('gesperrt');

  return el(
    'div',
    {
      class: klassen.join(' '),
      style: {
        '--a-groesse': `${groesse}px`,
        '--a-hell': stil.hell,
        '--a-dunkel': stil.dunkel,
      },
      title: gesperrt ? 'Noch nicht freigespielt' : daten.name,
    },
    [
      el('div', { class: 'rahmen' }),
      el('div', { class: 'koerper' }),
      el('div', { class: 'kern' }),
      el('div', { class: 'glanz' }),
      level.welt.id === 'weltraum' ? el('div', { class: 'ring' }) : null,
      el('div', { class: 'symbol', text: gesperrt ? '🔒' : daten.symbol }),
      el('div', { class: 'schatten' }),
    ]
  );
}

/** Artefakt mit Namen darunter, wie es nach einem Level gezeigt wird. */
export function artefaktBuehne(level, opt = {}) {
  const daten = artefaktFuer(level);
  return el('div', { class: 'artefakt-buehne' }, [
    opt.neu ? el('div', { class: 'artefakt-neu', text: '✨ Neues Artefakt ✨' }) : null,
    artefaktBild(level, opt),
    el('div', { class: 'artefakt-name', text: opt.gesperrt ? '???' : daten.name }),
  ]);
}

export { artefaktFuer };
