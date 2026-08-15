/**
 * Kleine Klang-Erzeugung mit der Web Audio API.
 * Keine Sound-Dateien noetig - alle Toene werden berechnet.
 */

import Spielstand from '../state/storage.js';

let ctx = null;

function audio() {
  if (!ctx) {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return null;
    ctx = new AC();
  }
  if (ctx.state === 'suspended') ctx.resume().catch(() => {});
  return ctx;
}

/** Muss einmal nach einer Nutzer-Geste aufgerufen werden (Browser-Regel). */
export function audioFreischalten() {
  audio();
}

function ton(frequenz, dauer, typ = 'square', lautstaerke = 0.16, gleitZu = null) {
  if (!Spielstand.einstellung('sound')) return;
  const a = audio();
  if (!a) return;
  const osz = a.createOscillator();
  const gain = a.createGain();
  osz.type = typ;
  osz.frequency.setValueAtTime(frequenz, a.currentTime);
  if (gleitZu) osz.frequency.exponentialRampToValueAtTime(gleitZu, a.currentTime + dauer);
  gain.gain.setValueAtTime(lautstaerke, a.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.0001, a.currentTime + dauer);
  osz.connect(gain).connect(a.destination);
  osz.start();
  osz.stop(a.currentTime + dauer + 0.02);
}

function melodie(noten) {
  noten.forEach(([f, start, dauer], i) => {
    setTimeout(() => ton(f, dauer, 'triangle', 0.18), start);
  });
}

const KLAENGE = {
  sprung: () => ton(420, 0.12, 'square', 0.12, 760),
  sammeln: () => ton(880, 0.09, 'triangle', 0.14, 1320),
  ei: () => melodie([[660, 0, 0.09], [990, 70, 0.12]]),
  platt: () => ton(300, 0.12, 'sawtooth', 0.12, 120),
  aua: () => ton(220, 0.25, 'sawtooth', 0.16, 90),
  spezial: () => ton(520, 0.18, 'triangle', 0.15, 1040),
  feder: () => ton(300, 0.2, 'sine', 0.18, 900),
  checkpoint: () => melodie([[523, 0, 0.1], [659, 90, 0.1], [784, 180, 0.16]]),
  ziel: () => melodie([[523, 0, 0.12], [659, 120, 0.12], [784, 240, 0.12], [1046, 360, 0.3]]),
  richtig: () => melodie([[784, 0, 0.1], [1046, 90, 0.16]]),
  falsch: () => ton(200, 0.22, 'square', 0.12, 140),
  klick: () => ton(600, 0.05, 'square', 0.08),
  wasser: () => ton(240, 0.16, 'sine', 0.1, 480),
};

export function tonSpielen(name) {
  KLAENGE[name]?.();
}

export default tonSpielen;
