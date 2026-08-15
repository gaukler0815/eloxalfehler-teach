/**
 * Touch-Steuerung: virtueller Joystick links, Sprung- und Spezial-Knopf rechts.
 * Laeuft parallel zur Tastatur - beides darf gleichzeitig benutzt werden.
 */

import { el } from './Dialog.js';
import Spielstand from '../state/storage.js';

const SCHWELLE = 0.34; // ab wann eine Richtung als gedrueckt gilt

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
    this.aufbauen();
    this.anzeigeAktualisieren();
  }

  aufbauen() {
    this.knopf = el('div', { class: 'knob' });
    this.stick = el('div', { class: 'pad-stick' }, [this.knopf]);

    this.sprungKnopf = el('button', { class: 'pad-btn', type: 'button' }, [
      el('span', { class: 'ico', text: '⤒' }),
      el('span', { text: 'Springen' }),
    ]);
    this.spezialKnopf = el('button', { class: 'pad-btn special', type: 'button' }, [
      el('span', { class: 'ico', text: '✨' }),
      el('span', { text: 'Kraft' }),
    ]);

    this.knoepfe = el('div', { class: 'pad-buttons' }, [this.spezialKnopf, this.sprungKnopf]);

    this.wurzel.replaceChildren(this.stick, this.knoepfe);
    this.ereignisseBinden();
  }

  ereignisseBinden() {
    const stickStart = (ev) => {
      ev.preventDefault();
      const z = ev.changedTouches ? ev.changedTouches[0] : ev;
      this.stickZeiger = z.identifier !== undefined ? z.identifier : 'maus';
      this.stick.classList.add('active');
      this.stickBewegen(z);
    };
    const stickBewegung = (ev) => {
      if (this.stickZeiger === null) return;
      const z = this.zeigerFinden(ev);
      if (!z) return;
      ev.preventDefault();
      this.stickBewegen(z);
    };
    const stickEnde = (ev) => {
      if (this.stickZeiger === null) return;
      if (ev.changedTouches && !this.zeigerFinden(ev, true)) return;
      this.stickZeiger = null;
      this.stick.classList.remove('active');
      this.knopf.style.transform = '';
      Object.assign(this.zustandDaten, {
        links: false,
        rechts: false,
        hoch: false,
        runter: false,
      });
    };

    this.stick.addEventListener('touchstart', stickStart, { passive: false });
    this.stick.addEventListener('mousedown', stickStart);
    window.addEventListener('touchmove', stickBewegung, { passive: false });
    window.addEventListener('mousemove', stickBewegung);
    window.addEventListener('touchend', stickEnde);
    window.addEventListener('touchcancel', stickEnde);
    window.addEventListener('mouseup', stickEnde);

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

  stickBewegen(zeiger) {
    const r = this.stick.getBoundingClientRect();
    const mx = r.left + r.width / 2;
    const my = r.top + r.height / 2;
    let dx = (zeiger.clientX - mx) / (r.width / 2);
    let dy = (zeiger.clientY - my) / (r.height / 2);
    const laenge = Math.hypot(dx, dy);
    if (laenge > 1) {
      dx /= laenge;
      dy /= laenge;
    }
    this.knopf.style.transform = `translate(${dx * r.width * 0.3}px, ${dy * r.height * 0.3}px)`;
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
    this.knopf.style.transform = '';
    this.stick.classList.remove('active');
    this.sprungKnopf.classList.remove('pressed');
    this.spezialKnopf.classList.remove('pressed');
  }
}
