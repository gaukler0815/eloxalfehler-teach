import Phaser from 'phaser';
import { kulisseAufbauen } from '../gfx/kulisse.js';
import { karteZeigen, el, overlayLeeren, frageZeigen } from '../ui/Dialog.js';
import { malbuchOeffnen } from '../ui/ColoringBook.js';
import Spielstand from '../state/storage.js';
import { DINOS, getDino, REGULAERE_DINOS } from '../data/dinos.js';
import { ANZAHL_LEVEL } from '../data/levels.js';
import { audioFreischalten, tonSpielen } from '../audio/sfx.js';
import { einheitsArt, einheitsTitel } from '../learning/QuizController.js';

/** Hauptmenue mit lebendiger Dschungel-Kulisse und HTML-Karte. */
export default class MenuScene extends Phaser.Scene {
  constructor() {
    super('Menu');
  }

  create() {
    this.kulisse = kulisseAufbauen(this, 'jungle');
    document.getElementById('hud').replaceChildren();
    document.getElementById('touch').classList.add('hidden');

    // Ein paar Dinos spazieren im Hintergrund
    this.laeufer = [];
    const frei = Spielstand.freieDinos();
    for (let i = 0; i < Math.min(4, frei.length); i += 1) {
      const d = frei[(frei.length - 1 - i) % frei.length];
      const s = this.add
        .sprite(120 + i * 260, this.scale.height - 120 - (i % 2) * 40, `dino_${d.id}`, 0)
        .setScale(1.3 + (i % 2) * 0.2)
        .setDepth(-50 + i);
      s.play(`${d.id}_lauf`);
      this.tweens.add({
        targets: s,
        x: this.scale.width + 120,
        duration: 9000 + i * 2500,
        repeat: -1,
        onRepeat: () => s.setX(-120),
      });
      this.laeufer.push(s);
    }

    this.hauptmenue();
    this.input.once('pointerdown', audioFreischalten);
    this.events.on('wake', () => this.hauptmenue());
  }

  update() {
    this.kulisse.aktualisieren();
  }

  // ------------------------------------------------------------ Hauptmenue

  hauptmenue() {
    const s = Spielstand.get();
    const naechstes = Math.min(ANZAHL_LEVEL, s.freigeschalteteLevel);
    const dino = getDino(Spielstand.gewaehlterDino());

    karteZeigen({
      titel: '🦖 Linneas Dino-Abenteuer',
      untertitel: `Hallo ${s.spielerName}! Dein Dino heißt gerade ${dino.name} (${dino.art}).`,
      inhalt: [
        el('div', { class: 'werte' }, [
          el('span', { text: `🗺 Level ${naechstes} von ${ANZAHL_LEVEL}` }),
          el('span', { text: `⭐ ${Spielstand.sterneGesamt()} Sterne` }),
          el('span', { text: `🏆 ${s.gesamtpunkte} Punkte` }),
          el('span', { text: `🦕 ${s.dinos.length} Dinos` }),
        ]),
      ],
      knoepfe: [
        {
          text: `▶ Level ${naechstes} spielen`,
          onClick: () => {
            tonSpielen('klick');
            overlayLeeren();
            this.scene.start('DinoSelect', { levelNr: naechstes });
          },
        },
        {
          text: '🗺 Level auswählen',
          klasse: 'blau',
          onClick: () => {
            tonSpielen('klick');
            overlayLeeren();
            this.scene.start('LevelSelect');
          },
        },
        {
          text: '🦕 Meine Dinos',
          klasse: 'lila',
          onClick: () => this.dinoSammlung(),
        },
        {
          text: '🦴 Fossilien ausgraben',
          klasse: 'gelb',
          onClick: () => {
            tonSpielen('klick');
            overlayLeeren();
            this.scene.start('Minigame', { zurueck: 'Menu' });
          },
        },
        {
          text: '🎨 Dino-Malbuch',
          klasse: 'gelb',
          onClick: () => malbuchOeffnen(() => this.hauptmenue()),
        },
        {
          text: '⚙ Fortschritt & Einstellungen',
          klasse: 'grau',
          onClick: () => this.einstellungen(),
        },
      ],
    });
  }

  // ------------------------------------------------------- Dino-Sammlung

  dinoSammlung() {
    const gitter = el(
      'div',
      { class: 'gitter' },
      DINOS.map((d, i) => {
        const frei = Spielstand.dinoFrei(d.id);
        const kachel = el('div', { class: `kachel${frei ? '' : ' gesperrt'}` }, [
          el('div', { class: 'nr', text: frei ? '🦕' : '🔒' }),
          el('div', { class: 'name', text: frei ? `${d.name}\n${d.art}` : `Level ${i} schaffen` }),
        ]);
        return kachel;
      })
    );

    karteZeigen({
      titel: `🦕 Meine Dinos (${Spielstand.get().dinos.length} von ${REGULAERE_DINOS + 1})`,
      untertitel:
        'Nach jedem geschafften Level mit bestandener Lernaufgabe kommt ein neuer Dino dazu.',
      breit: true,
      inhalt: [gitter],
      knoepfe: [{ text: '↩ Zurück', klasse: 'grau', onClick: () => this.hauptmenue() }],
    });
  }

  // -------------------------------------------------------- Einstellungen

  einstellungen() {
    const s = Spielstand.get();
    const bestanden = Object.keys(s.lerneinheiten).length;
    const stat = s.lernstatistik;
    const quote = stat.gesamt > 0 ? Math.round((stat.richtig / stat.gesamt) * 100) : 0;

    const naechsteEinheit = Math.min(ANZAHL_LEVEL, s.freigeschalteteLevel);

    const umschalter = (name, text) =>
      el('div', { class: 'balken' }, [
        el('span', { text, style: { minWidth: '170px' } }),
        el(
          'button',
          {
            class: `knopf ${Spielstand.einstellung(name) ? '' : 'grau'}`,
            type: 'button',
            style: { minHeight: '46px', fontSize: '17px', flex: '0 0 auto' },
            onClick: (ev) => {
              const neu = !Spielstand.einstellung(name);
              Spielstand.einstellung(name, neu);
              ev.currentTarget.textContent = neu ? 'AN' : 'AUS';
              ev.currentTarget.classList.toggle('grau', !neu);
            },
          },
          Spielstand.einstellung(name) ? 'AN' : 'AUS'
        ),
      ]);

    const touchModus = Spielstand.einstellung('touch') || 'auto';

    karteZeigen({
      titel: '⚙ Fortschritt & Einstellungen',
      inhalt: [
        el('div', { class: 'werte' }, [
          el('span', { text: `🗺 ${s.freigeschalteteLevel - 1} Level geschafft` }),
          el('span', { text: `📚 ${bestanden} Lerneinheiten bestanden` }),
          el('span', { text: `✔ ${stat.richtig} von ${stat.gesamt} Aufgaben richtig (${quote} %)` }),
        ]),
        el('div', {
          class: 'hinweis',
          text: `Als Nächstes: Level ${naechsteEinheit} und danach eine ${
            einheitsArt(naechsteEinheit) === 'lesen' ? 'Lese' : 'Mathe'
          }-Einheit („${einheitsTitel(naechsteEinheit)}").`,
        }),
        umschalter('sound', 'Töne'),
        umschalter('musik', 'Jubel-Melodien'),
        el('div', { class: 'balken' }, [
          el('span', { text: 'Touch-Steuerung', style: { minWidth: '170px' } }),
          ...['auto', 'an', 'aus'].map((m) =>
            el(
              'button',
              {
                class: `knopf ${touchModus === m ? '' : 'grau'}`,
                type: 'button',
                style: { minHeight: '46px', fontSize: '17px' },
                onClick: () => {
                  Spielstand.einstellung('touch', m);
                  this.einstellungen();
                },
              },
              m.toUpperCase()
            )
          ),
        ]),
      ],
      knoepfe: [
        { text: '↩ Zurück', onClick: () => this.hauptmenue() },
        {
          text: '🗑 Spielstand löschen',
          klasse: 'rot',
          onClick: () =>
            frageZeigen(
              'Wirklich alles löschen?',
              'Alle Level, Dinos und Bilder gehen verloren. Das kann man nicht rückgängig machen.',
              () => {
                Spielstand.alleLoeschen();
                this.scene.restart();
              },
              () => this.einstellungen(),
              'Ja, löschen',
              'Nein, behalten'
            ),
        },
      ],
    });
  }
}
