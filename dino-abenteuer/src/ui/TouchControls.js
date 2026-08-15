/**
 * Touch-Steuerung.
 *
 * Der Joystick ist "schwebend": Er ist unsichtbar, bis ein Finger die linke
 * Bildhaelfte beruehrt, und erscheint dann genau unter dem Finger. So liegt
 * kein Ring dauerhaft im Bild und nichts verdeckt den Dino.
 * Sprung- und Kraft-Knopf bleiben sichtbar, damit man sie findet.
 * Alles laeuft parallel zur Tastatur - beides darf gleichzeitig benutzt werden.
 */

import { el } from './Dialog.js';
import Spielstand from '../state/storage.js';

const SCHWELLE = 0.34; // ab wann eine Richtung als gedrueckt gilt
const RADIUS = 62; // Weg des Knopfes bis zum vollen Ausschlag (px)

export function istTouchGeraet() {
  return (
    'ontouchstart' in window ||
    (navigator.maxTouchPoints || 0) > 0 ||
    window.matchMedia('(pointer: coarse)').matches
  );
}

export default class TouchControls {
  constructor() {
    this.wurzel = document.getElementById('touch');
    this.zustandDaten = {
      links: false,
      rechts: false,
      hoch: false,
      runter: false,
      sprung: false,
      spezial: false,
    };
    this.stickZeiger = null;
    this.mitte = { x: 0, y: 0 };
    this.aufbauen();
    this.anzeigeAktualisieren();
  }

  aufbauen() {
    this.knopf = el('div', { class: 'knob' });
    this.stick = el('div', { class: 'pad-stick' }, [this.knopf]);
    // Unsichtbare Flaeche: hier darf der Daumen den Joystick aufrufen.
    this.zone = el('div', { class: 'pad-zone' }, [this.stick]);

    this.sprungKnopf = el('button', { class: 'pad-btn', type: 'button' }, [
      el('span', { class: 'ico', text: '⤒' }),
      el('span', { text: 'Springen' }),
    ]);
    this.spezialKnopf = el('button', { class: 'pad-btn special', type: 'button' }, [
      el('span', { class: 'ico', text: '✨' }),
      el('span', { text: 'Kraft' }),
    ]);

    this.knoepfe = el('div', { class: 'pad-buttons' }, [this.spezialKnopf, this.sprungKnopf]);

    this.wurzel.replaceChildren(this.zone, this.knoepfe);
    this.ereignisseBinden();
  }

  ereignisseBinden() {
    const start = (ev) => {
      if (this.stickZeiger !== null) return;
      ev.preventDefault();
      const z = ev.changedTouches ? ev.changedTouches[0] : ev;
      this.stickZeiger = z.identifier !== undefined ? z.identifier : 'maus';
      this.stickZeigen(z);
    };
    const bewegung = (ev) => {
      if (this.stickZeiger === null) return;
      const z = this.zeigerFinden(ev);
      if (!z) return;
      ev.preventDefault();
      this.stickBewegen(z);
    };
    const ende = (ev) => {
      if (this.stickZeiger === null) return;
      if (ev.changedTouches && !this.zeigerFinden(ev, true)) return;
      this.stickVerbergen();
    };

    this.zone.addEventListener('touchstart', start, { passive: false });
    this.zone.addEventListener('mousedown', start);
    window.addEventListener('touchmove', bewegung, { passive: false });
    window.addEventListener('mousemove', bewegung);
    window.addEventListener('touchend', ende);
    window.addEventListener('touchcancel', ende);
    window.addEventListener('mouseup', ende);

    this.knopfBinden(this.sprungKnopf, 'sprung');
    this.knopfBinden(this.spezialKnopf, 'spezial');
  }

  zeigerFinden(ev, nurEnde = false) {
    if (!ev.changedTouches) return this.stickZeiger === 'maus' ? ev : null;
    const liste = nurEnde ? ev.changedTouches : ev.touches;
    for (let i = 0; i < liste.length; i += 1) {
      if (liste[i].identifier === this.stickZeiger) return liste[i];
    }
    return null;
  }

  /** Joystick unter dem Finger aufblenden. */
  stickZeigen(zeiger) {
    const r = this.zone.getBoundingClientRect();
    // Nah am Rand etwas hereinruecken, damit der Ring komplett sichtbar bleibt
    const x = Math.min(Math.max(zeiger.clientX - r.left, 80), r.width - 20);
    const y = Math.min(Math.max(zeiger.clientY - r.top, 80), r.height - 80);
    this.mitte = { x: r.left + x, y: r.top + y };
    this.stick.style.left = `${x}px`;
    this.stick.style.top = `${y}px`;
    this.stick.classList.add('sichtbar');
    this.knopf.style.transform = '';
  }

  stickVerbergen() {
    this.stickZeiger = null;
    this.stick.classList.remove('sichtbar');
    this.knopf.style.transform = '';
    Object.assign(this.zustandDaten, {
      links: false,
      rechts: false,
      hoch: false,
      runter: false,
    });
  }

  stickBewegen(zeiger) {
    let dx = (zeiger.clientX - this.mitte.x) / RADIUS;
    let dy = (zeiger.clientY - this.mitte.y) / RADIUS;
    const laenge = Math.hypot(dx, dy);
    if (laenge > 1) {
      dx /= laenge;
      dy /= laenge;
    }
    this.knopf.style.transform = `translate(${dx * 34}px, ${dy * 34}px)`;
    this.zustandDaten.links = dx < -SCHWELLE;
    this.zustandDaten.rechts = dx > SCHWELLE;
    this.zustandDaten.hoch = dy < -SCHWELLE;
    this.zustandDaten.runter = dy > SCHWELLE;
  }

  knopfBinden(knopf, name) {
    const an = (ev) => {
      ev.preventDefault();
      this.zustandDaten[name] = true;
      knopf.classList.add('pressed');
    };
    const aus = () => {
      this.zustandDaten[name] = false;
      knopf.classList.remove('pressed');
    };
    knopf.addEventListener('touchstart', an, { passive: false });
    knopf.addEventListener('mousedown', an);
    knopf.addEventListener('touchend', aus);
    knopf.addEventListener('touchcancel', aus);
    knopf.addEventListener('mouseup', aus);
    knopf.addEventListener('mouseleave', aus);
  }

  /** Ladebalken der Spezialfaehigkeit anzeigen. */
  spezialLadestand(anteil, icon) {
    if (icon && this.spezialKnopf.firstChild) {
      this.spezialKnopf.firstChild.textContent = icon;
    }
    this.spezialKnopf.classList.toggle('cooldown', anteil < 1);
  }

  zustand() {
    return this.zustandDaten;
  }

  anzeigeAktualisieren() {
    const modus = Spielstand.einstellung('touch') || 'auto';
    const zeigen = modus === 'an' || (modus === 'auto' && istTouchGeraet());
    this.wurzel.classList.toggle('hidden', !zeigen);
  }

  sichtbar(an) {
    this.wurzel.style.display = an ? '' : 'none';
    if (an) this.anzeigeAktualisieren();
  }

  zuruecksetzen() {
    Object.keys(this.zustandDaten).forEach((k) => {
      this.zustandDaten[k] = false;
    });
    this.stickVerbergen();
    this.sprungKnopf.classList.remove('pressed');
    this.spezialKnopf.classList.remove('pressed');
  }
}
