import Phaser from 'phaser';
import { kulisseAufbauen } from '../gfx/kulisse.js';
import { karteZeigen, el, overlayLeeren } from '../ui/Dialog.js';
import Spielstand from '../state/storage.js';
import { DINOS, ABILITIES, getDino } from '../data/dinos.js';
import { getLevel } from '../data/levels.js';
import { dinoVorschau } from '../gfx/dinoArt.js';
import { einheitsArt, einheitsTitel } from '../learning/QuizController.js';
import { tonSpielen } from '../audio/sfx.js';

/** Dino-Auswahl vor jedem Level. */
export default class DinoSelectScene extends Phaser.Scene {
  constructor() {
    super('DinoSelect');
  }

  init(daten) {
    this.levelNr = daten?.levelNr || Spielstand.hoechstesLevel();
  }

  create() {
    this.level = getLevel(this.levelNr);
    this.kulisse = kulisseAufbauen(this, this.level.biom);
    document.getElementById('touch').classList.add('hidden');
    document.getElementById('hud').replaceChildren();
    this.gewaehlt = Spielstand.gewaehlterDino();
    this.anzeigen();
  }

  update() {
    this.kulisse.aktualisieren();
  }

  anzeigen() {
    const dino = getDino(this.gewaehlt);
    const f = ABILITIES[dino.ability];

    const balken = (name, wert, max) =>
      el('div', { class: 'balken' }, [
        el('span', { text: name, style: { minWidth: '110px' } }),
        el('span', { class: 'spur' }, [
          el('span', {
            class: 'fuell',
            style: { width: `${Math.round((wert / max) * 100)}%` },
          }),
        ]),
      ]);

    const vorschau = dinoVorschau(dino, 150);

    const info = el('div', { class: 'dino-info' }, [
      el('div', {}, [vorschau]),
      el('div', { style: { flex: '1 1 240px' } }, [
        el('strong', { text: `${dino.name} — ${dino.art}` }),
        el('div', { text: dino.fakt, style: { margin: '4px 0 8px' } }),
        balken('Tempo', dino.speed, 280),
        balken('Sprungkraft', dino.jump, 600),
        balken('Schwimmen', dino.swim, 2),
        el('div', {
          class: 'hinweis',
          style: { marginTop: '10px' },
          text: `${f.icon} ${f.name}: ${f.hint}`,
        }),
      ]),
    ]);

    const gitter = el(
      'div',
      { class: 'gitter' },
      DINOS.filter((d) => Spielstand.dinoFrei(d.id)).map((d) =>
        el(
          'button',
          {
            class: `kachel${d.id === this.gewaehlt ? ' gewaehlt' : ''}`,
            type: 'button',
            onClick: () => {
              this.gewaehlt = d.id;
              Spielstand.dinoWaehlen(d.id);
              tonSpielen('klick');
              this.anzeigen();
            },
          },
          [dinoVorschau(d, 88), el('div', { class: 'name', text: d.name })]
        )
      )
    );

    karteZeigen({
      titel: `Level ${this.levelNr}: ${this.level.name}`,
      untertitel: `${this.level.biomName} — such dir einen Dino aus, ${
        Spielstand.get().spielerName
      }!`,
      breit: true,
      inhalt: [
        info,
        el('div', { class: 'biom-titel', text: 'Deine freigeschalteten Dinos' }),
        gitter,
        el('div', {
          class: 'hinweis',
          text: `Nach dem Level wartet ${
            einheitsArt(this.levelNr) === 'lesen' ? 'eine Lese-Einheit' : 'eine Mathe-Einheit'
          }: „${einheitsTitel(this.levelNr)}".`,
        }),
      ],
      knoepfe: [
        {
          text: '▶ Los geht es!',
          onClick: () => {
            overlayLeeren();
            this.scene.start('Game', { levelNr: this.levelNr, dinoId: this.gewaehlt });
          },
        },
        {
          text: '🗺 Anderes Level',
          klasse: 'blau',
          onClick: () => {
            overlayLeeren();
            this.scene.start('LevelSelect');
          },
        },
        {
          text: '↩ Menü',
          klasse: 'grau',
          onClick: () => {
            overlayLeeren();
            this.scene.start('Menu');
          },
        },
      ],
    });
  }
}
