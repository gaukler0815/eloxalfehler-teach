/**
 * 15 Mathe-Einheiten (gerade Level 2, 4, 6 ... 30).
 *
 * Zahlenraum bis 20: Plus, Minus, Lueckenaufgaben, Verdoppeln, Halbieren.
 * Jede Einheit erzeugt ihre Aufgaben neu - bei einem zweiten Versuch
 * bekommt Linnea also frische Zahlen und kann nichts auswendig lernen.
 * Bestanden ab 80 % richtig geloester Aufgaben.
 */

function zi(r, min, max) {
  return min + Math.floor(r() * (max - min + 1));
}

/**
 * Baut eine Aufgabe.
 * @param {string} anzeige z. B. "7 + 5 = ?" - die Luecke ist "?" oder "__"
 * @param {number} loesung
 * @param {string} art Kurzname fuer die Statistik
 */
function aufgabe(anzeige, loesung, art) {
  return { anzeige, loesung, art, id: `${art}:${anzeige}` };
}

const plus = (r, max) => {
  const a = zi(r, 1, max - 1);
  const b = zi(r, 1, max - a);
  return aufgabe(`${a} + ${b} = ?`, a + b, 'plus');
};

const minus = (r, max) => {
  const a = zi(r, 2, max);
  const b = zi(r, 1, a - 1);
  return aufgabe(`${a} − ${b} = ?`, a - b, 'minus');
};

/** Plus im Zahlenraum 20 ohne Zehneruebergang (Einer bleiben unter 10). */
const plusOhneUebergang = (r) => {
  const zehner = zi(r, 1, 1) * 10;
  const a = zehner + zi(r, 0, 8);
  const b = zi(r, 1, 19 - a);
  return aufgabe(`${a} + ${b} = ?`, a + b, 'plus20');
};

const minusOhneUebergang = (r) => {
  const a = 10 + zi(r, 1, 9);
  const b = zi(r, 1, a - 10);
  return aufgabe(`${a} − ${b} = ?`, a - b, 'minus20');
};

const plusUebergang = (r) => {
  const a = zi(r, 5, 9);
  const b = zi(r, 11 - a, Math.min(9, 20 - a));
  return aufgabe(`${a} + ${b} = ?`, a + b, 'uebergang+');
};

const minusUebergang = (r) => {
  const a = zi(r, 11, 18);
  const b = zi(r, a - 9, 9);
  return aufgabe(`${a} − ${b} = ?`, a - b, 'uebergang−');
};

const verdoppeln = (r, max) => {
  const a = zi(r, 1, max);
  return aufgabe(`Das Doppelte von ${a} ist ?`, a * 2, 'doppelt');
};

const halbieren = (r, max) => {
  const a = zi(r, 1, max) * 2;
  return aufgabe(`Die Hälfte von ${a} ist ?`, a / 2, 'halb');
};

const lueckePlusHinten = (r, max) => {
  const a = zi(r, 1, max - 1);
  const summe = zi(r, a + 1, max);
  return aufgabe(`${a} + __ = ${summe}`, summe - a, 'lücke+');
};

const lueckePlusVorne = (r, max) => {
  const b = zi(r, 1, max - 1);
  const summe = zi(r, b + 1, max);
  return aufgabe(`__ + ${b} = ${summe}`, summe - b, 'lücke+');
};

const lueckeMinusHinten = (r, max) => {
  const a = zi(r, 3, max);
  const rest = zi(r, 0, a - 1);
  return aufgabe(`${a} − __ = ${rest}`, a - rest, 'lücke−');
};

const lueckeMinusVorne = (r, max) => {
  const b = zi(r, 1, Math.floor(max / 2));
  const rest = zi(r, 1, max - b);
  return aufgabe(`__ − ${b} = ${rest}`, rest + b, 'lücke−');
};

const dreiZahlen = (r) => {
  if (r() < 0.5) {
    const a = zi(r, 1, 8);
    const b = zi(r, 1, 8);
    const c = zi(r, 1, Math.max(1, 20 - a - b));
    return aufgabe(`${a} + ${b} + ${c} = ?`, a + b + c, 'drei+');
  }
  const a = zi(r, 5, 12);
  const b = zi(r, 1, 20 - a);
  const c = zi(r, 1, a + b - 1);
  return aufgabe(`${a} + ${b} − ${c} = ?`, a + b - c, 'drei±');
};

/** Sachaufgabe im Dino-Gewand - dieselbe Rechnung, nur bunter erzaehlt. */
const sachaufgabe = (r) => {
  const vorlagen = [
    (a, b) => [`Rexi sammelt ${a} Eier. Dann findet er noch ${b}. Wie viele hat er?`, a + b],
    (a, b) => [`Im Nest liegen ${a + b} Eier. ${b} schlüpfen. Wie viele liegen noch da?`, a],
    (a, b) => [`Trixi frisst ${a} Farne und ${b} Blätter. Wie viele Pflanzen sind das?`, a + b],
    (a, b) => [`${a + b} Käfer sitzen auf dem Stein. ${b} fliegen weg. Wie viele bleiben?`, a],
    (a, b) => [`Pterri fliegt ${a} Runden. Dann noch ${b} Runden. Wie viele Runden sind das?`, a + b],
  ];
  const a = zi(r, 2, 9);
  const b = zi(r, 1, Math.min(9, 20 - a));
  const [text, loesung] = vorlagen[zi(r, 0, vorlagen.length - 1)](a, b);
  // Das angehaengte "?" markiert die Luecke fuer die Anzeige.
  return aufgabe(`${text} ?`, loesung, 'sach');
};

export const MATHE_EINHEITEN = [
  {
    titel: 'Plus bis 10',
    hinweis: 'Zähle einfach weiter - deine Finger dürfen helfen!',
    anzahl: 6,
    erzeugen: (r) => plus(r, 10),
  },
  {
    titel: 'Minus bis 10',
    hinweis: 'Zähle rückwärts von der großen Zahl.',
    anzahl: 6,
    erzeugen: (r) => minus(r, 10),
  },
  {
    titel: 'Plus und Minus bis 10',
    hinweis: 'Achte gut auf das Rechenzeichen.',
    anzahl: 7,
    erzeugen: (r) => (r() < 0.5 ? plus(r, 10) : minus(r, 10)),
  },
  {
    titel: 'Verdoppeln bis 10',
    hinweis: 'Das Doppelte heißt: die Zahl plus sich selbst.',
    anzahl: 6,
    erzeugen: (r) => verdoppeln(r, 10),
  },
  {
    titel: 'Halbieren bis 20',
    hinweis: 'Teile die Zahl gerecht auf zwei Dinos auf.',
    anzahl: 6,
    erzeugen: (r) => halbieren(r, 10),
  },
  {
    titel: 'Plus bis 20',
    hinweis: 'Erst bis zur 10, dann weiter.',
    anzahl: 7,
    erzeugen: (r) => plusOhneUebergang(r),
  },
  {
    titel: 'Lückenaufgaben mit Plus',
    hinweis: 'Was fehlt bis zur Zahl am Ende?',
    anzahl: 7,
    erzeugen: (r) => (r() < 0.6 ? lueckePlusHinten(r, 10) : lueckePlusVorne(r, 10)),
  },
  {
    titel: 'Minus bis 20',
    hinweis: 'Die Zehn bleibt stehen, nur die Einer werden weniger.',
    anzahl: 7,
    erzeugen: (r) => minusOhneUebergang(r),
  },
  {
    titel: 'Lückenaufgaben mit Minus',
    hinweis: 'Überlege: Wie viel wurde weggenommen?',
    anzahl: 7,
    erzeugen: (r) => (r() < 0.6 ? lueckeMinusHinten(r, 15) : lueckeMinusVorne(r, 15)),
  },
  {
    titel: 'Plus mit Zehnerübergang',
    hinweis: 'Mache erst die 10 voll, dann rechne weiter.',
    anzahl: 8,
    erzeugen: (r) => plusUebergang(r),
  },
  {
    titel: 'Minus mit Zehnerübergang',
    hinweis: 'Gehe erst zurück bis zur 10.',
    anzahl: 8,
    erzeugen: (r) => minusUebergang(r),
  },
  {
    titel: 'Verdoppeln und Halbieren',
    hinweis: 'Doppelt ist zweimal so viel, halb ist die Hälfte.',
    anzahl: 8,
    erzeugen: (r) => (r() < 0.5 ? verdoppeln(r, 10) : halbieren(r, 10)),
  },
  {
    titel: 'Lücken überall',
    hinweis: 'Die Lücke kann vorne oder hinten stehen.',
    anzahl: 8,
    erzeugen: (r) => {
      const w = r();
      if (w < 0.3) return lueckePlusHinten(r, 20);
      if (w < 0.55) return lueckePlusVorne(r, 20);
      if (w < 0.8) return lueckeMinusHinten(r, 20);
      return lueckeMinusVorne(r, 20);
    },
  },
  {
    titel: 'Drei Zahlen',
    hinweis: 'Rechne von links nach rechts, Schritt für Schritt.',
    anzahl: 8,
    erzeugen: (r) => dreiZahlen(r),
  },
  {
    titel: 'Der große Dino-Mix',
    hinweis: 'Alles, was du schon kannst - du schaffst das!',
    anzahl: 10,
    erzeugen: (r) => {
      const w = r();
      if (w < 0.2) return plusUebergang(r);
      if (w < 0.4) return minusUebergang(r);
      if (w < 0.55) return lueckePlusHinten(r, 20);
      if (w < 0.7) return lueckeMinusHinten(r, 20);
      if (w < 0.8) return verdoppeln(r, 10);
      if (w < 0.88) return halbieren(r, 10);
      return sachaufgabe(r);
    },
  },
];

/** Welche Mathe-Einheit gehoert zu welchem (geraden) Level? */
export function matheEinheitFuerLevel(levelNr) {
  const index = (Math.floor(levelNr / 2) - 1 + MATHE_EINHEITEN.length) % MATHE_EINHEITEN.length;
  return { einheit: MATHE_EINHEITEN[index], nummer: index + 1 };
}

/**
 * Erzeugt den Aufgabensatz einer Einheit (ohne Dubletten).
 * @param {object} einheit
 * @param {() => number} r Zufallsfunktion
 */
export function aufgabensatz(einheit, r = Math.random) {
  const satz = [];
  const gesehen = new Set();
  let versuche = 0;
  while (satz.length < einheit.anzahl && versuche < 300) {
    versuche += 1;
    const a = einheit.erzeugen(r);
    if (gesehen.has(a.id)) continue;
    gesehen.add(a.id);
    satz.push(a);
  }
  return satz;
}

export const ANZAHL_MATHE_EINHEITEN = MATHE_EINHEITEN.length;
