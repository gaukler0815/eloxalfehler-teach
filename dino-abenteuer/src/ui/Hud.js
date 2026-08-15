/**
 * HTML-HUD ueber dem Spiel-Canvas: Level, Punkte, Sammelstand, Herzen, Zeit.
 */

import { el } from './Dialog.js';

export default class Hud {
  constructor({ onPause }) {
    this.wurzel = document.getElementById('hud');
    this.felder = {};

    const chip = (name, inhalt) => {
      const c = el('span', { class: 'hud-chip' }, inhalt);
      this.felder[name] = c;
      return c;
    };

    this.leiste = el('div', { class: 'hud-bar' }, [
      chip('level', 'Level'),
      chip('punkte', '0'),
      chip('eier', '🥚 0/0'),
      chip('fruechte', '🍎 0/0'),
      chip('herzen', '❤❤❤'),
      el('span', { class: 'hud-spacer' }),
      chip('zeit', '0:00'),
      el('button', { class: 'hud-btn', type: 'button', title: 'Pause', onClick: onPause }, '⏸'),
    ]);

    this.wurzel.replaceChildren(this.leiste);
  }

  aktualisieren(d) {
    const f = this.felder;
    f.level.textContent = `Level ${d.level}`;
    f.punkte.textContent = `⭐ ${d.punkte}`;
    f.eier.textContent = `🥚 ${d.eier}/${d.eierGesamt}`;
    f.fruechte.textContent = `🍎 ${d.fruechte}/${d.fruechteGesamt}`;
    f.herzen.textContent = '❤'.repeat(Math.max(0, d.herzen)) + '🤍'.repeat(Math.max(0, 3 - d.herzen));
    const s = Math.floor(d.zeit);
    f.zeit.textContent = `⏱ ${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
  }

  sichtbar(an) {
    this.wurzel.classList.toggle('hidden', !an);
  }

  zerstoeren() {
    this.wurzel.replaceChildren();
  }
}
