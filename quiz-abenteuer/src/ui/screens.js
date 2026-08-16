/**
 * Alle Bildschirme der Quiz-App.
 *
 * Ablauf: Start -> Level -> 5 Fragen -> Ergebnis -> naechstes Level.
 * Ab 4 von 5 richtigen Antworten (80 %) ist ein Level bestanden und das
 * naechste wird freigeschaltet. Wiederholen geht beliebig oft.
 */

import { el, knopf, karte, zeigen, mischen, sterneText, weltFarbe } from './dom.js';
import Spielstand from '../state/storage.js';
import { tonSpielen } from '../audio/sfx.js';
import {
  WELTEN,
  LEVEL,
  ANZAHL_LEVEL,
  FRAGEN_PRO_LEVEL,
  BESTEHENSQUOTE,
  getLevel,
  levelDerWelt,
} from '../data/welten.js';

const PUNKTE_PRO_FRAGE = 10;
const BONUS_ALLES_RICHTIG = 20;
const NOETIG = Math.ceil(FRAGEN_PRO_LEVEL * BESTEHENSQUOTE); // 4 von 5

const NAME = () => Spielstand.get().spielerName || 'Linnea';

const LOB = ['Super', 'Klasse', 'Stark', 'Wow', 'Prima', 'Toll'];
const lob = (i) => LOB[i % LOB.length];

// ============================================================== Startseite

export function start() {
  weltFarbe('#1b7f4b');
  const s = Spielstand.get();
  const naechstes = Spielstand.hoechstes();
  const lvl = getLevel(naechstes);
  const geschafft = Spielstand.anzahlGeschafft();

  zeigen(
    karte([
      el('h1', { text: `🧠 ${NAME()}s Quiz-Abenteuer` }),
      el('p', {
        class: 'lead',
        text: `100 Level mit je 5 Fragen über Dinos, Tiere, Natur und den Weltraum. Vier von fünf Antworten müssen stimmen, dann geht es weiter.`,
      }),
      el('div', { class: 'werte' }, [
        el('span', { text: `🗺 ${geschafft} von ${ANZAHL_LEVEL} Leveln` }),
        el('span', { text: `⭐ ${Spielstand.sterneGesamt()} Sterne` }),
        el('span', { text: `🏆 ${s.punkte} Punkte` }),
      ]),
      el('div', { class: 'knopf-reihe' }, [
        knopf(`▶ Level ${naechstes}: ${lvl.welt.icon} ${lvl.thema}`, () => quizStarten(naechstes)),
      ]),
      el('div', { class: 'knopf-reihe' }, [
        knopf('🗺 Welten & Level', welten, 'blau'),
        knopf('🏅 Abzeichen', abzeichen, 'lila'),
        knopf('⚙ Einstellungen', einstellungen, 'grau'),
      ]),
    ])
  );
}

// ================================================================= Welten

export function welten() {
  weltFarbe('#2f7fd1');
  zeigen(
    karte(
      [
        el('h1', { text: '🗺 Die vier Welten' }),
        el('p', {
          class: 'lead',
          text: 'Die Level wechseln sich ab: erst ein Dino-Level, dann Tiere, dann Natur, dann Weltraum.',
        }),
        el(
          'div',
          { class: 'welten' },
          WELTEN.map((w) => {
            const p = Spielstand.weltFortschritt(w.id);
            return el(
              'button',
              { class: 'weltkachel', type: 'button', onClick: () => levelListe(w.id) },
              [
                el('span', { class: 'icon', text: w.icon }),
                el('span', { style: { flex: '1' } }, [
                  el('div', { class: 'titel', text: w.name }),
                  el('div', { class: 'unter', text: `${p.fertig} von ${p.gesamt} Leveln` }),
                  el('div', { class: 'balken' }, [
                    el('i', { style: { width: `${(p.fertig / p.gesamt) * 100}%` } }),
                  ]),
                ]),
              ]
            );
          })
        ),
        el('div', { class: 'knopf-reihe' }, [knopf('↩ Zurück', start, 'grau')]),
      ],
      'breit'
    )
  );
}

export function levelListe(weltId) {
  const welt = WELTEN.find((w) => w.id === weltId) || WELTEN[0];
  weltFarbe(welt.farbe);
  const naechstes = Spielstand.hoechstes();

  zeigen(
    karte(
      [
        el('h1', { text: `${welt.icon} ${welt.name}` }),
        el('p', { class: 'lead', text: 'Tippe auf ein Level. Graue Level musst du erst freispielen.' }),
        el(
          'div',
          { class: 'level-gitter' },
          levelDerWelt(weltId).map((l) => {
            const frei = Spielstand.frei(l.nr);
            const erg = Spielstand.ergebnis(l.nr);
            const klassen = [
              'levelkachel',
              !frei && 'gesperrt',
              erg?.bestanden && 'geschafft',
              l.nr === naechstes && 'naechstes',
            ]
              .filter(Boolean)
              .join(' ');
            return el(
              'button',
              {
                class: klassen,
                type: 'button',
                title: `${l.thema}${frei ? '' : ' (noch gesperrt)'}`,
                onClick: () => frei && quizStarten(l.nr),
              },
              [
                el('span', { class: 'nr', text: frei ? String(l.nr) : '🔒' }),
                el('span', { class: 'st', text: erg ? sterneText(erg.sterne) : '☆☆☆' }),
              ]
            );
          })
        ),
        el('div', { class: 'knopf-reihe' }, [
          knopf('↩ Welten', welten, 'blau'),
          knopf('🏠 Start', start, 'grau'),
        ]),
      ],
      'breit'
    )
  );
}

// =================================================================== Quiz

export function quizStarten(nr) {
  const level = getLevel(nr);
  // Fragen und Antworten mischen - so lernt man den Inhalt, nicht die Position.
  const fragen = mischen(level.fragen).map((fr) => ({
    frage: fr.frage,
    info: fr.info,
    optionen: mischen(fr.antworten.map((t, i) => ({ text: t, richtig: i === 0 }))),
  }));
  frageZeigen(level, fragen, 0, []);
}

function kopfzeile(level, index, ergebnisse) {
  return el('div', { class: 'kopf' }, [
    el('span', { class: 'chip welt', text: `${level.welt.icon} ${level.welt.name}` }),
    el('span', { class: 'chip', text: `Level ${level.nr}` }),
    el('span', { class: 'chip', text: `Frage ${index + 1} von ${FRAGEN_PRO_LEVEL}` }),
    el('span', { class: 'schieber' }),
    el('span', { class: 'chip', text: `✔ ${ergebnisse.filter(Boolean).length}` }),
  ]);
}

function fortschritt(index, ergebnisse) {
  return el(
    'div',
    { class: 'fortschritt' },
    Array.from({ length: FRAGEN_PRO_LEVEL }, (_, i) =>
      el('span', {
        class: i < ergebnisse.length ? (ergebnisse[i] ? 'ok' : 'nok') : i === index ? 'jetzt' : '',
      })
    )
  );
}

function frageZeigen(level, fragen, index, ergebnisse) {
  if (index >= fragen.length) {
    ergebnis(level, ergebnisse);
    return;
  }
  weltFarbe(level.welt.farbe);
  const f = fragen[index];
  let beantwortet = false;

  const rueckmeldung = el('div');
  const knoepfe = [];

  const antworten = el(
    'div',
    { class: 'antworten' },
    f.optionen.map((o, i) => {
      const b = el(
        'button',
        {
          class: 'antwort',
          type: 'button',
          onClick: () => antworten_klick(i),
        },
        [
          el('span', { class: 'buchstabe', text: 'ABCD'[i] }),
          el('span', { text: o.text }),
        ]
      );
      knoepfe.push(b);
      return b;
    })
  );

  function antworten_klick(i) {
    if (beantwortet) return;
    beantwortet = true;
    const richtig = f.optionen[i].richtig;
    knoepfe.forEach((b, k) => {
      b.disabled = true;
      if (f.optionen[k].richtig) b.classList.add('richtig');
      else if (k === i) b.classList.add('falsch');
    });
    tonSpielen(richtig ? 'richtig' : 'falsch');

    const richtigeAntwort = f.optionen.find((o) => o.richtig).text;
    rueckmeldung.replaceChildren(
      el('div', { class: `hinweis ${richtig ? 'gut' : 'schade'}` }, [
        el('strong', {
          text: richtig ? `${lob(index)}, ${NAME()}! Richtig.` : `Fast! Richtig ist: ${richtigeAntwort}`,
        }),
        el('span', { text: f.info }),
      ]),
      el('div', { class: 'knopf-reihe' }, [
        knopf(
          index + 1 < fragen.length ? 'Weiter ➜' : 'Ergebnis ansehen 🎉',
          () => {
            tastaturAus();
            frageZeigen(level, fragen, index + 1, [...ergebnisse, richtig]);
          }
        ),
      ])
    );
    rueckmeldung.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  const beiTaste = (ev) => {
    const k = ev.key.toUpperCase();
    const pos = 'ABCD'.indexOf(k) >= 0 ? 'ABCD'.indexOf(k) : '1234'.indexOf(k);
    if (pos >= 0 && !beantwortet) antworten_klick(pos);
    else if (beantwortet && (ev.key === 'Enter' || ev.key === ' ')) {
      rueckmeldung.querySelector('button.knopf')?.click();
    }
  };
  const tastaturAus = () => window.removeEventListener('keydown', beiTaste);
  window.addEventListener('keydown', beiTaste);

  zeigen(
    karte([
      kopfzeile(level, index, ergebnisse),
      fortschritt(index, ergebnisse),
      el('div', { class: 'frage', text: f.frage }),
      antworten,
      rueckmeldung,
    ])
  );
}

// =============================================================== Ergebnis

function ergebnis(level, ergebnisse) {
  const richtig = ergebnisse.filter(Boolean).length;
  const bestanden = richtig >= NOETIG;
  const sterne = richtig === FRAGEN_PRO_LEVEL ? 3 : bestanden ? 2 : 0;
  const punkte = richtig * PUNKTE_PRO_FRAGE + (richtig === FRAGEN_PRO_LEVEL ? BONUS_ALLES_RICHTIG : 0);

  const vorher = Spielstand.ergebnis(level.nr);
  const warSchonBestanden = !!vorher?.bestanden;
  Spielstand.ergebnisSpeichern(level.nr, {
    richtig,
    gesamt: FRAGEN_PRO_LEVEL,
    punkte,
    sterne,
    bestanden,
  });

  tonSpielen(bestanden ? 'ziel' : 'falsch');

  const naechstes = level.nr + 1;
  const fertig = naechstes > ANZAHL_LEVEL;
  const naechsterLevel = fertig ? null : getLevel(naechstes);

  const inhalt = [
    el('h1', { text: bestanden ? '🎉 Level geschafft!' : '💪 Fast geschafft' }),
    el('p', { class: 'lead', text: `${level.welt.icon} ${level.welt.name} — ${level.thema}` }),
    el('div', { class: 'sterne-gross', text: bestanden ? sterneText(sterne) : '☆☆☆' }),
    el('div', { class: 'werte' }, [
      el('span', { text: `${richtig} von ${FRAGEN_PRO_LEVEL} richtig` }),
      el('span', { text: `${Math.round((richtig / FRAGEN_PRO_LEVEL) * 100)} %` }),
      el('span', { text: `+${punkte} Punkte` }),
    ]),
    el('div', { class: `hinweis ${bestanden ? 'gut' : 'schade'}` }, [
      el('strong', {
        text: bestanden
          ? richtig === FRAGEN_PRO_LEVEL
            ? `Alles richtig, ${NAME()}! Volle drei Sterne.`
            : `Bestanden, ${NAME()}!`
          : `Du brauchst ${NOETIG} von ${FRAGEN_PRO_LEVEL} richtigen Antworten.`,
      }),
      el('span', {
        text: bestanden
          ? fertig
            ? 'Du hast alle 100 Level geschafft. Wahnsinn!'
            : warSchonBestanden
              ? 'Dieses Level hattest du schon geschafft - jetzt mit neuem Bestwert.'
              : `Level ${naechstes} ist jetzt frei: ${naechsterLevel.welt.icon} ${naechsterLevel.thema}`
          : 'Probier es einfach noch einmal. Die Fragen kommen in neuer Reihenfolge.',
      }),
    ]),
  ];

  const knoepfe = [];
  if (bestanden && !fertig) {
    knoepfe.push(
      knopf(`▶ Level ${naechstes}: ${naechsterLevel.welt.icon} ${naechsterLevel.thema}`, () =>
        quizStarten(naechstes)
      )
    );
  }
  knoepfe.push(knopf('🔁 Nochmal spielen', () => quizStarten(level.nr), bestanden ? 'gelb' : ''));
  knoepfe.push(knopf('🗺 Level-Übersicht', () => levelListe(level.welt.id), 'blau'));
  knoepfe.push(knopf('🏠 Start', start, 'grau'));

  inhalt.push(el('div', { class: 'knopf-reihe' }, knoepfe.slice(0, 2)));
  inhalt.push(el('div', { class: 'knopf-reihe' }, knoepfe.slice(2)));

  zeigen(karte(inhalt));
}

// =============================================================== Abzeichen

export function abzeichen() {
  weltFarbe('#8b5cf6');
  const liste = Spielstand.abzeichen();
  const offen = liste.filter((a) => a.offen).length;

  zeigen(
    karte(
      [
        el('h1', { text: '🏅 Deine Abzeichen' }),
        el('p', {
          class: 'lead',
          text: `${liste.length - offen} von ${liste.length} gesammelt. Für jedes Abzeichen musst du Level schaffen.`,
        }),
        el(
          'div',
          { class: 'abzeichen' },
          liste.map((a) =>
            el('div', { class: a.offen ? 'offen' : '' }, [
              el('span', { class: 'em', text: a.em }),
              el('div', { text: a.name }),
              el('div', { style: { opacity: '0.7' }, text: a.info }),
            ])
          )
        ),
        el('div', { class: 'knopf-reihe' }, [knopf('↩ Zurück', start, 'grau')]),
      ],
      'breit'
    )
  );
}

// ============================================================ Einstellungen

export function einstellungen() {
  weltFarbe('#7a8896');
  const s = Spielstand.get();
  const geschafft = Spielstand.anzahlGeschafft();
  const versuche = Object.values(s.ergebnisse).reduce((n, e) => n + (e.versuche || 0), 0);

  zeigen(
    karte([
      el('h1', { text: '⚙ Fortschritt & Einstellungen' }),
      el('div', { class: 'werte' }, [
        el('span', { text: `🗺 ${geschafft} von ${ANZAHL_LEVEL} Leveln` }),
        el('span', { text: `⭐ ${Spielstand.sterneGesamt()} Sterne` }),
        el('span', { text: `🏆 ${s.punkte} Punkte` }),
        el('span', { text: `🔁 ${versuche} Versuche` }),
      ]),
      el('div', { class: 'hinweis' }, [
        el('strong', { text: 'So funktioniert es' }),
        el('span', {
          text: `Jedes Level hat ${FRAGEN_PRO_LEVEL} Fragen. Ab ${NOETIG} richtigen Antworten ist es bestanden. Alle ${FRAGEN_PRO_LEVEL} richtig gibt drei Sterne und ${BONUS_ALLES_RICHTIG} Bonuspunkte.`,
        }),
      ]),
      el('div', { class: 'knopf-reihe' }, [
        knopf(
          Spielstand.einstellung('sound') ? '🔊 Töne sind AN' : '🔇 Töne sind AUS',
          () => {
            Spielstand.einstellung('sound', !Spielstand.einstellung('sound'));
            einstellungen();
          },
          Spielstand.einstellung('sound') ? '' : 'grau'
        ),
      ]),
      el('div', { class: 'knopf-reihe' }, [
        knopf('↩ Zurück', start, 'blau'),
        knopf('🗑 Fortschritt löschen', () => loeschenFragen(), 'rot'),
      ]),
    ])
  );
}

function loeschenFragen() {
  zeigen(
    karte([
      el('h1', { text: 'Wirklich alles löschen?' }),
      el('p', {
        class: 'lead',
        text: 'Alle geschafften Level, Sterne und Punkte gehen verloren. Das kann man nicht rückgängig machen.',
      }),
      el('div', { class: 'knopf-reihe' }, [
        knopf('Ja, löschen', () => {
          Spielstand.alleLoeschen();
          start();
        }, 'rot'),
        knopf('Nein, behalten', einstellungen, 'grau'),
      ]),
    ])
  );
}
