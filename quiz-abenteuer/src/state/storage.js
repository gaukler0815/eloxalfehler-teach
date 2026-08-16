/**
 * Spielstand in localStorage.
 * Faellt der Speicher aus (privater Modus), laeuft die App trotzdem -
 * der Fortschritt gilt dann nur fuer die aktuelle Sitzung.
 */

import { ANZAHL_LEVEL, WELTEN, levelDerWelt } from '../data/welten.js';

const KEY = 'linnea-quiz-abenteuer-v1';

const STANDARD = {
  version: 1,
  spielerName: 'Linnea',
  freigeschaltet: 1,
  /** pro Levelnummer: { richtig, sterne, punkte, versuche } */
  ergebnisse: {},
  punkte: 0,
  einstellungen: { sound: true },
};

let notfallSpeicher = null;

function lesen() {
  try {
    const roh = window.localStorage.getItem(KEY);
    return roh ? JSON.parse(roh) : null;
  } catch (e) {
    return notfallSpeicher;
  }
}

function schreiben(daten) {
  notfallSpeicher = daten;
  try {
    window.localStorage.setItem(KEY, JSON.stringify(daten));
  } catch (e) {
    /* nicht speicherbar - egal, die Sitzung laeuft weiter */
  }
}

class Speicher {
  constructor() {
    const g = lesen() || {};
    this.daten = {
      ...STANDARD,
      ...g,
      einstellungen: { ...STANDARD.einstellungen, ...(g.einstellungen || {}) },
      ergebnisse: { ...(g.ergebnisse || {}) },
    };
    this.sichern();
  }

  sichern() {
    schreiben(this.daten);
  }

  get() {
    return this.daten;
  }

  // ---- Level ------------------------------------------------------------

  frei(nr) {
    return nr <= this.daten.freigeschaltet;
  }

  hoechstes() {
    return Math.min(ANZAHL_LEVEL, this.daten.freigeschaltet);
  }

  ergebnis(nr) {
    return this.daten.ergebnisse[nr] || null;
  }

  geschafft(nr) {
    return !!this.daten.ergebnisse[nr]?.bestanden;
  }

  /**
   * Ergebnis eines Versuchs speichern.
   * Behalten wird immer das beste Resultat, die Punkte zaehlen nur einmal.
   */
  ergebnisSpeichern(nr, { richtig, gesamt, punkte, sterne, bestanden }) {
    const alt = this.daten.ergebnisse[nr];
    const neu = {
      richtig: Math.max(alt?.richtig || 0, richtig),
      gesamt,
      sterne: Math.max(alt?.sterne || 0, sterne),
      punkte: Math.max(alt?.punkte || 0, punkte),
      versuche: (alt?.versuche || 0) + 1,
      bestanden: !!(alt?.bestanden || bestanden),
    };
    this.daten.punkte += Math.max(0, neu.punkte - (alt?.punkte || 0));
    this.daten.ergebnisse[nr] = neu;
    if (bestanden && nr + 1 > this.daten.freigeschaltet) {
      this.daten.freigeschaltet = Math.min(ANZAHL_LEVEL, nr + 1);
    }
    this.sichern();
    return neu;
  }

  sterneGesamt() {
    return Object.values(this.daten.ergebnisse).reduce((s, e) => s + (e.sterne || 0), 0);
  }

  anzahlGeschafft() {
    return Object.values(this.daten.ergebnisse).filter((e) => e.bestanden).length;
  }

  weltFortschritt(weltId) {
    const level = levelDerWelt(weltId);
    const fertig = level.filter((l) => this.geschafft(l.nr)).length;
    return { fertig, gesamt: level.length };
  }

  // ---- Abzeichen --------------------------------------------------------

  abzeichen() {
    const geschafft = this.anzahlGeschafft();
    const sterne = this.sterneGesamt();
    const liste = WELTEN.map((w) => {
      const p = this.weltFortschritt(w.id);
      return {
        em: w.icon,
        name: `${w.name} komplett`,
        offen: p.fertig < p.gesamt,
        info: `${p.fertig}/${p.gesamt}`,
      };
    });
    [
      [10, '🥉', '10 Level geschafft'],
      [25, '🥈', '25 Level geschafft'],
      [50, '🥇', '50 Level geschafft'],
      [100, '👑', 'Alle 100 Level'],
    ].forEach(([n, em, name]) => {
      liste.push({ em, name, offen: geschafft < n, info: `${geschafft}/${n}` });
    });
    liste.push({
      em: '⭐',
      name: '100 Sterne',
      offen: sterne < 100,
      info: `${sterne}/100`,
    });
    return liste;
  }

  // ---- Einstellungen ----------------------------------------------------

  einstellung(name, wert) {
    if (wert === undefined) return this.daten.einstellungen[name];
    this.daten.einstellungen[name] = wert;
    this.sichern();
    return wert;
  }

  alleLoeschen() {
    this.daten = { ...STANDARD, ergebnisse: {}, einstellungen: { ...STANDARD.einstellungen } };
    this.sichern();
  }
}

export const Spielstand = new Speicher();
export default Spielstand;
