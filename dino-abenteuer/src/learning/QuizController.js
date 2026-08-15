/**
 * Steuert die Lerneinheiten nach jedem Level.
 *
 * Ungerade Level -> Lesetext mit 5 Verstaendnisfragen (4 von 5 = bestanden)
 * Gerade Level   -> Mathe-Set mit 5-10 Aufgaben (80 % = bestanden)
 *
 * Wiederholen ist beliebig oft moeglich. Erst wenn die Einheit bestanden
 * ist, wird das naechste Level freigeschaltet.
 */

import { lesetextFuerLevel } from './readingData.js';
import { matheEinheitFuerLevel, aufgabensatz } from './mathData.js';
import { karteZeigen, el, overlayLeeren } from '../ui/Dialog.js';
import Spielstand from '../state/storage.js';

export const BESTEHENSQUOTE = 0.8;

const NAME = () => Spielstand.get().spielerName || 'Linnea';

const LOB = [
  'Klasse gemacht',
  'Super',
  'Stark',
  'Wow',
  'Toll gelesen',
  'Prima',
];

function lob(i) {
  return LOB[i % LOB.length];
}

/** Ist das Level eine Lese- oder eine Mathe-Einheit? */
export function einheitsArt(levelNr) {
  return levelNr % 2 === 1 ? 'lesen' : 'mathe';
}

export function einheitsTitel(levelNr) {
  if (einheitsArt(levelNr) === 'lesen') return lesetextFuerLevel(levelNr).titel;
  return matheEinheitFuerLevel(levelNr).einheit.titel;
}

/**
 * Startet die Lerneinheit fuer ein abgeschlossenes Level.
 * @param {number} levelNr
 * @param {(bestanden:boolean) => void} onFertig
 */
export function lerneinheitStarten(levelNr, onFertig) {
  if (einheitsArt(levelNr) === 'lesen') leseEinheit(levelNr, onFertig);
  else matheEinheit(levelNr, onFertig);
}

// ============================================================ Lese-Einheit

function leseEinheit(levelNr, onFertig) {
  const text = lesetextFuerLevel(levelNr);
  const nummer = Math.floor((levelNr - 1) / 2) + 1;
  let grosseSchrift = false;

  const intro = () =>
    karteZeigen({
      titel: `Geschafft, ${NAME()}!`,
      untertitel: `Du hast Level ${levelNr} gemeistert. Jetzt wartet Lese-Einheit ${nummer} von 15 auf dich.`,
      inhalt: [
        el('div', { class: 'hinweis' }, [
          el('strong', { text: text.titel }),
          el('div', {
            text:
              'Lies den Text in Ruhe durch. Danach kommen 5 Fragen. ' +
              'Vier richtige Antworten reichen schon zum Bestehen.',
          }),
        ]),
      ],
      knoepfe: [{ text: '📖 Lesen starten', onClick: lesen }],
    });

  const lesen = () => {
    const textBox = el(
      'div',
      { class: `lesetext${grosseSchrift ? ' gross' : ''}` },
      text.absaetze.map((p) => el('p', { text: p }))
    );
    karteZeigen({
      titel: text.titel,
      inhalt: [textBox],
      knoepfe: [
        { text: '✅ Fertig gelesen - los zu den Fragen', onClick: () => fragen(0, []) },
        {
          text: grosseSchrift ? '🔍 Kleinere Schrift' : '🔍 Größere Schrift',
          klasse: 'gelb',
          onClick: () => {
            grosseSchrift = !grosseSchrift;
            lesen();
          },
        },
      ],
    });
  };

  const fragen = (index, ergebnisse) => {
    if (index >= text.fragen.length) {
      auswerten(ergebnisse);
      return;
    }
    const f = text.fragen[index];
    const balken = el(
      'div',
      { class: 'fortschritt' },
      text.fragen.map((_, i) =>
        el('span', {
          class: i < ergebnisse.length ? (ergebnisse[i] ? 'ok' : 'nok') : '',
        })
      )
    );

    const antwortKnoepfe = [];
    let beantwortet = false;

    const rueckmeldung = el('div', { style: { minHeight: '4px' } });

    const antwortReihe = el(
      'div',
      { class: 'antworten' },
      f.optionen.map((o, i) => {
        const b = el(
          'button',
          {
            class: 'antwort',
            type: 'button',
            onClick: () => {
              if (beantwortet) return;
              beantwortet = true;
              const richtig = i === f.richtig;
              b.classList.add(richtig ? 'richtig' : 'falsch');
              if (!richtig) antwortKnoepfe[f.richtig].classList.add('richtig');
              rueckmeldung.replaceChildren(
                el('div', {
                  class: `hinweis ${richtig ? 'gut' : 'schade'}`,
                  text: richtig
                    ? `${lob(index)}, ${NAME()}! Das ist richtig.`
                    : `Fast! Richtig ist: „${f.optionen[f.richtig]}". ${f.tipp || ''}`,
                }),
                el(
                  'div',
                  { class: 'knopf-reihe' },
                  [
                    el(
                      'button',
                      {
                        class: 'knopf',
                        type: 'button',
                        onClick: () => fragen(index + 1, [...ergebnisse, richtig]),
                      },
                      index + 1 < text.fragen.length ? 'Weiter ➜' : 'Ergebnis ansehen'
                    ),
                  ]
                )
              );
            },
          },
          `${String.fromCharCode(65 + i)})  ${o}`
        );
        antwortKnoepfe.push(b);
        return b;
      })
    );

    karteZeigen({
      titel: `Frage ${index + 1} von ${text.fragen.length}`,
      inhalt: [
        balken,
        el('div', { class: 'frage', text: f.frage }),
        antwortReihe,
        rueckmeldung,
      ],
      knoepfe: [],
    });
  };

  const auswerten = (ergebnisse) => {
    const richtig = ergebnisse.filter(Boolean).length;
    const gesamt = ergebnisse.length;
    abschluss({
      levelNr,
      richtig,
      gesamt,
      onFertig,
      nochmal: lesen,
      artText: 'Lese-Einheit',
    });
  };

  intro();
}

// =========================================================== Mathe-Einheit

function matheEinheit(levelNr, onFertig) {
  const { einheit, nummer } = matheEinheitFuerLevel(levelNr);

  const starten = () => {
    const satz = aufgabensatz(einheit, Math.random);
    aufgabe(satz, 0, []);
  };

  const intro = () =>
    karteZeigen({
      titel: `Stark, ${NAME()}!`,
      untertitel: `Level ${levelNr} ist geschafft. Jetzt kommt Mathe-Einheit ${nummer} von 15.`,
      inhalt: [
        el('div', { class: 'hinweis' }, [
          el('strong', { text: einheit.titel }),
          el('div', { text: einheit.hinweis }),
          el('div', {
            text: `${einheit.anzahl} Aufgaben - ab 80 % richtig hast du bestanden.`,
          }),
        ]),
      ],
      knoepfe: [{ text: '🔢 Los geht es', onClick: starten }],
    });

  const aufgabe = (satz, index, ergebnisse) => {
    if (index >= satz.length) {
      abschluss({
        levelNr,
        richtig: ergebnisse.filter(Boolean).length,
        gesamt: ergebnisse.length,
        onFertig,
        nochmal: starten,
        artText: 'Mathe-Einheit',
      });
      return;
    }

    const a = satz[index];
    let eingabe = '';
    let fertig = false;

    const balken = el(
      'div',
      { class: 'fortschritt' },
      satz.map((_, i) =>
        el('span', { class: i < ergebnisse.length ? (ergebnisse[i] ? 'ok' : 'nok') : '' })
      )
    );

    const anzeige = el('div', { class: `rechnung${a.anzeige.length > 26 ? ' lang' : ''}` });
    const rueckmeldung = el('div', { style: { minHeight: '4px' } });

    const anzeigeAktualisieren = () => {
      const platzhalter = eingabe === '' ? '?' : eingabe;
      const roh = a.anzeige.replace('__', '@@').replace(/\?$/, '@@');
      const teile = roh.split('@@');
      anzeige.replaceChildren(
        document.createTextNode(teile[0] || ''),
        el('span', { class: 'luecke', text: ` ${platzhalter} ` }),
        document.createTextNode(teile[1] || '')
      );
    };

    const pruefen = () => {
      if (fertig || eingabe === '') return;
      fertig = true;
      const richtig = Number(eingabe) === a.loesung;
      rueckmeldung.replaceChildren(
        el('div', {
          class: `hinweis ${richtig ? 'gut' : 'schade'}`,
          text: richtig
            ? `${lob(index)}, ${NAME()}! ${a.loesung} ist richtig.`
            : `Nicht ganz. Die richtige Antwort ist ${a.loesung}. ${einheit.hinweis}`,
        }),
        el('div', { class: 'knopf-reihe' }, [
          el(
            'button',
            {
              class: 'knopf',
              type: 'button',
              onClick: () => {
                tastaturAus();
                aufgabe(satz, index + 1, [...ergebnisse, richtig]);
              },
            },
            index + 1 < satz.length ? 'Weiter ➜' : 'Ergebnis ansehen'
          ),
        ])
      );
    };

    const ziffer = (z) => {
      if (fertig || eingabe.length >= 2) return;
      eingabe += z;
      anzeigeAktualisieren();
    };

    const feld = el('div', { class: 'zahlenfeld' }, [
      ...['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'].map((z) =>
        el('button', { type: 'button', onClick: () => ziffer(z) }, z)
      ),
      el(
        'button',
        {
          class: 'loeschen',
          type: 'button',
          onClick: () => {
            if (fertig) return;
            eingabe = eingabe.slice(0, -1);
            anzeigeAktualisieren();
          },
        },
        '⌫ Weg'
      ),
      el('button', { class: 'aktion', type: 'button', onClick: pruefen }, '✓ Prüfen'),
    ]);

    const beiTaste = (ev) => {
      if (ev.key >= '0' && ev.key <= '9') ziffer(ev.key);
      else if (ev.key === 'Backspace') {
        eingabe = eingabe.slice(0, -1);
        anzeigeAktualisieren();
      } else if (ev.key === 'Enter') pruefen();
    };
    const tastaturAus = () => window.removeEventListener('keydown', beiTaste);
    window.addEventListener('keydown', beiTaste);

    anzeigeAktualisieren();
    karteZeigen({
      titel: `Aufgabe ${index + 1} von ${satz.length}`,
      inhalt: [balken, anzeige, feld, rueckmeldung],
      knoepfe: [],
    });
  };

  intro();
}

// ================================================================ Abschluss

function abschluss({ levelNr, richtig, gesamt, onFertig, nochmal, artText }) {
  const quote = gesamt > 0 ? richtig / gesamt : 0;
  const bestanden = quote >= BESTEHENSQUOTE;
  Spielstand.lerneinheitSpeichern(levelNr, { richtig, gesamt, bestanden });

  const prozent = Math.round(quote * 100);
  const inhalt = [
    el('div', { class: 'sterne-gross', text: bestanden ? '⭐⭐⭐' : '💪' }),
    el('div', { class: 'werte' }, [
      el('span', { text: `${richtig} von ${gesamt} richtig` }),
      el('span', { text: `${prozent} %` }),
      el('span', { text: `Ziel: 80 %` }),
    ]),
    el('div', {
      class: `hinweis ${bestanden ? 'gut' : 'schade'}`,
      text: bestanden
        ? `Bestanden! Du hast dir Level ${levelNr + 1} und einen neuen Dino verdient, ${NAME()}.`
        : `Das war knapp, ${NAME()}. Probier es einfach noch einmal - du bekommst neue Chancen, so oft du willst.`,
    }),
  ];

  const knoepfe = bestanden
    ? [{ text: '🎉 Belohnung ansehen', onClick: () => { overlayLeeren(); onFertig(true); } }]
    : [
        { text: '🔁 Nochmal versuchen', onClick: () => nochmal() },
        {
          text: 'Später weitermachen',
          klasse: 'grau',
          onClick: () => {
            overlayLeeren();
            onFertig(false);
          },
        },
      ];

  karteZeigen({
    titel: bestanden ? `${artText} bestanden!` : 'Fast geschafft',
    inhalt,
    knoepfe,
  });
}
