/**
 * Spielstand in localStorage.
 *
 * Gespeichert werden: freigeschaltete Level und Dinos, Highscores pro Level,
 * bestandene Lerneinheiten, Einstellungen, Malbuch-Bilder und Fossilien.
 * Faellt localStorage aus (Privatmodus), wird auf ein reines
 * Speicherobjekt zurueckgefallen - das Spiel laeuft dann trotzdem.
 */

import { DINOS } from '../data/dinos.js';
import { ANZAHL_LEVEL } from '../data/levels.js';

const KEY = 'linnea-dino-abenteuer-v1';

const STANDARD = {
  version: 1,
  spielerName: 'Linnea',
  freigeschalteteLevel: 1,
  dinos: ['rexi'],
  gewaehlterDino: 'rexi',
  /** pro Level: { punkte, sterne, zeit, eier, fruechte } */
  highscores: {},
  /** pro Level: true, wenn die Lerneinheit bestanden wurde */
  lerneinheiten: {},
  /** Statistik der Lerneinheiten: { versuche, richtig, gesamt } */
  lernstatistik: { versuche: 0, richtig: 0, gesamt: 0 },
  fossilien: {},
  malbuch: {},
  einstellungen: { sound: true, musik: true, touch: 'auto' },
  gesamtpunkte: 0,
};

let speicherFallback = null;

function lesenRoh() {
  try {
    const raw = window.localStorage.getItem(KEY);
    return raw ? JSON.parse(raw) : null;
  } catch (e) {
    return speicherFallback;
  }
}

function schreibenRoh(daten) {
  speicherFallback = daten;
  try {
    window.localStorage.setItem(KEY, JSON.stringify(daten));
  } catch (e) {
    /* Speichern nicht moeglich - Spielstand bleibt nur fuer diese Sitzung. */
  }
}

function verschmelzen(basis, gespeichert) {
  const out = { ...basis, ...(gespeichert || {}) };
  out.einstellungen = { ...basis.einstellungen, ...(gespeichert?.einstellungen || {}) };
  out.lernstatistik = { ...basis.lernstatistik, ...(gespeichert?.lernstatistik || {}) };
  out.highscores = { ...(gespeichert?.highscores || {}) };
  out.lerneinheiten = { ...(gespeichert?.lerneinheiten || {}) };
  out.fossilien = { ...(gespeichert?.fossilien || {}) };
  out.malbuch = { ...(gespeichert?.malbuch || {}) };
  if (!Array.isArray(out.dinos) || out.dinos.length === 0) out.dinos = ['rexi'];
  return out;
}

class SpielstandSpeicher {
  constructor() {
    this.daten = verschmelzen(STANDARD, lesenRoh());
    this.sichern();
  }

  sichern() {
    schreibenRoh(this.daten);
  }

  get() {
    return this.daten;
  }

  // ---- Level ------------------------------------------------------------

  levelFrei(nr) {
    return nr <= this.daten.freigeschalteteLevel;
  }

  hoechstesLevel() {
    return this.daten.freigeschalteteLevel;
  }

  /** Wird nach bestandener Lerneinheit aufgerufen. */
  levelFreischalten(nr) {
    if (nr > this.daten.freigeschalteteLevel) {
      this.daten.freigeschalteteLevel = Math.min(ANZAHL_LEVEL, nr);
      this.sichern();
    }
  }

  ergebnisSpeichern(nr, ergebnis) {
    const alt = this.daten.highscores[nr];
    const neu = {
      punkte: Math.max(alt?.punkte || 0, ergebnis.punkte),
      sterne: Math.max(alt?.sterne || 0, ergebnis.sterne),
      zeit: alt?.zeit ? Math.min(alt.zeit, ergebnis.zeit) : ergebnis.zeit,
      eier: Math.max(alt?.eier || 0, ergebnis.eier),
      fruechte: Math.max(alt?.fruechte || 0, ergebnis.fruechte),
    };
    const zuwachs = neu.punkte - (alt?.punkte || 0);
    this.daten.highscores[nr] = neu;
    this.daten.gesamtpunkte = Math.max(0, (this.daten.gesamtpunkte || 0) + Math.max(0, zuwachs));
    this.sichern();
    return neu;
  }

  ergebnis(nr) {
    return this.daten.highscores[nr] || null;
  }

  sterneGesamt() {
    return Object.values(this.daten.highscores).reduce((s, e) => s + (e.sterne || 0), 0);
  }

  // ---- Dinos ------------------------------------------------------------

  dinoFrei(id) {
    return this.daten.dinos.includes(id);
  }

  freieDinos() {
    return DINOS.filter((d) => this.dinoFrei(d.id));
  }

  /**
   * Schaltet den Dino frei, der zu einem abgeschlossenen Level gehoert.
   * Level 1 -> Dino Nr. 2, Level 2 -> Dino Nr. 3, ... Level 30 -> Geheim-Dino.
   * @returns {object|null} der neu freigeschaltete Dino oder null
   */
  dinoFuerLevelFreischalten(levelNr) {
    const dino = DINOS[levelNr];
    if (!dino || this.dinoFrei(dino.id)) return null;
    this.daten.dinos.push(dino.id);
    this.sichern();
    return dino;
  }

  dinoWaehlen(id) {
    if (!this.dinoFrei(id)) return;
    this.daten.gewaehlterDino = id;
    this.sichern();
  }

  gewaehlterDino() {
    return this.dinoFrei(this.daten.gewaehlterDino) ? this.daten.gewaehlterDino : 'rexi';
  }

  // ---- Lerneinheiten ----------------------------------------------------

  lerneinheitBestanden(levelNr) {
    return !!this.daten.lerneinheiten[levelNr];
  }

  lerneinheitSpeichern(levelNr, { richtig, gesamt, bestanden }) {
    const s = this.daten.lernstatistik;
    s.versuche += 1;
    s.richtig += richtig;
    s.gesamt += gesamt;
    if (bestanden) this.daten.lerneinheiten[levelNr] = true;
    this.sichern();
  }

  // ---- Minispiele -------------------------------------------------------

  fossilSpeichern(id, daten) {
    this.daten.fossilien[id] = { ...(this.daten.fossilien[id] || {}), ...daten };
    this.sichern();
  }

  malbuchSpeichern(id, farben) {
    this.daten.malbuch[id] = farben;
    this.sichern();
  }

  malbuchLaden(id) {
    return this.daten.malbuch[id] || null;
  }

  // ---- Einstellungen ----------------------------------------------------

  einstellung(name, wert) {
    if (wert === undefined) return this.daten.einstellungen[name];
    this.daten.einstellungen[name] = wert;
    this.sichern();
    return wert;
  }

  alleLoeschen() {
    this.daten = verschmelzen(STANDARD, null);
    this.sichern();
  }
}

export const Spielstand = new SpielstandSpeicher();
export default Spielstand;
