/**
 * Prueft den kompletten Fragenbestand.
 * Aufruf: npm test
 *
 * Bei 500 Fragen faellt ein Tippfehler sonst erst dem Kind auf.
 */

import { WELTEN, LEVEL, ANZAHL_LEVEL, FRAGEN_PRO_LEVEL } from '../src/data/welten.js';

let fehler = 0;
const pruefe = (bedingung, text) => {
  if (!bedingung) {
    console.error(`  ✗ ${text}`);
    fehler += 1;
  }
};

console.log('Prüfe Welten ...');
pruefe(WELTEN.length === 4, 'Es müssen 4 Welten sein');
WELTEN.forEach((w) => {
  pruefe(w.saetze.length === 25, `${w.name}: 25 Level-Sätze erwartet, sind ${w.saetze.length}`);
  pruefe(w.themen.length === 25, `${w.name}: 25 Themen erwartet, sind ${w.themen.length}`);
  pruefe(!!w.icon && !!w.farbe, `${w.name}: Icon und Farbe fehlen`);
  w.saetze.forEach((satz, i) => {
    pruefe(
      satz.length === FRAGEN_PRO_LEVEL,
      `${w.name} Satz ${i + 1}: ${FRAGEN_PRO_LEVEL} Fragen erwartet, sind ${satz.length}`
    );
  });
});

console.log('Prüfe Level ...');
pruefe(LEVEL.length === ANZAHL_LEVEL, `${ANZAHL_LEVEL} Level erwartet, sind ${LEVEL.length}`);
LEVEL.forEach((l) => {
  pruefe(!!l.thema, `Level ${l.nr}: Thema fehlt`);
  pruefe(l.fragen?.length === FRAGEN_PRO_LEVEL, `Level ${l.nr}: falsche Fragenzahl`);
});
// Jede Welt kommt gleich oft vor und die Level wechseln sich ab
WELTEN.forEach((w) => {
  const anzahl = LEVEL.filter((l) => l.welt.id === w.id).length;
  pruefe(anzahl === 25, `${w.name}: 25 Level erwartet, sind ${anzahl}`);
});
// Jeder Fragensatz wird genau einmal verwendet
const benutzt = new Set();
LEVEL.forEach((l) => {
  const schluessel = `${l.welt.id}#${l.index}`;
  pruefe(!benutzt.has(schluessel), `Fragensatz ${schluessel} wird doppelt verwendet`);
  benutzt.add(schluessel);
});

console.log('Prüfe Fragen ...');
const alleFragen = [];
LEVEL.forEach((l) => {
  l.fragen.forEach((fr, i) => {
    const wo = `Level ${l.nr} (${l.welt.name}, ${l.thema}) Frage ${i + 1}`;
    pruefe(typeof fr.frage === 'string' && fr.frage.length > 5, `${wo}: Fragetext fehlt`);
    pruefe(fr.frage.trim().endsWith('?'), `${wo}: Frage endet nicht mit Fragezeichen`);
    pruefe(Array.isArray(fr.antworten) && fr.antworten.length === 4, `${wo}: es braucht genau 4 Antworten`);
    pruefe(
      new Set(fr.antworten.map((a) => a.toLowerCase().trim())).size === 4,
      `${wo}: doppelte Antwortmöglichkeit`
    );
    fr.antworten.forEach((a, k) => {
      pruefe(typeof a === 'string' && a.trim().length > 0, `${wo}: Antwort ${k + 1} ist leer`);
      pruefe(a.length <= 70, `${wo}: Antwort ${k + 1} ist sehr lang (${a.length} Zeichen)`);
    });
    pruefe(typeof fr.info === 'string' && fr.info.length > 5, `${wo}: Erklärung fehlt`);
    pruefe(fr.frage.length <= 110, `${wo}: Frage ist sehr lang (${fr.frage.length} Zeichen)`);
    alleFragen.push({ text: fr.frage.toLowerCase().trim(), wo });
  });
});

pruefe(
  alleFragen.length === ANZAHL_LEVEL * FRAGEN_PRO_LEVEL,
  `${ANZAHL_LEVEL * FRAGEN_PRO_LEVEL} Fragen erwartet, sind ${alleFragen.length}`
);

// Doppelte Fragen finden (innerhalb einer Welt besonders aergerlich)
const gesehen = new Map();
alleFragen.forEach(({ text, wo }) => {
  if (gesehen.has(text)) {
    console.warn(`  ! Gleiche Frage zweimal: "${text}"\n      ${gesehen.get(text)}\n      ${wo}`);
  } else {
    gesehen.set(text, wo);
  }
});
const doppelt = alleFragen.length - gesehen.size;

console.log('');
console.log(`Welten: ${WELTEN.length} | Level: ${LEVEL.length} | Fragen: ${alleFragen.length}`);
console.log(`Verschiedene Fragen: ${gesehen.size} (${doppelt} Wiederholungen im Finale)`);

if (fehler > 0) {
  console.error(`\n${fehler} Fehler gefunden.`);
  process.exit(1);
}
console.log('\nAlles in Ordnung.');
