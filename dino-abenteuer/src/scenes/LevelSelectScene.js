import Phaser from 'phaser';
import { kulisseAufbauen } from '../gfx/kulisse.js';
import { karteZeigen, el, overlayLeeren, sterneText } from '../ui/Dialog.js';
import Spielstand from '../state/storage.js';
import { LEVELS, BIOME, ANZAHL_LEVEL } from '../data/levels.js';
import { einheitsArt, einheitsTitel } from '../learning/QuizController.js';
import { tonSpielen } from '../audio/sfx.js';

/** Levelauswahl: 30 Kacheln, nach Biom gruppiert. */
export default class LevelSelectScene extends Phaser.Scene {
  constructor() {
    super('LevelSelect');
  }

  create() {
    const naechstes = Math.min(ANZAHL_LEVEL, Spielstand.hoechstesLevel());
    const biom = LEVELS[naechstes - 1].biom;
    this.kulisse = kulisseAufbauen(this, biom);
    document.getElementById('touch').classList.add('hidden');
    document.getElementById('hud').replaceChildren();
    this.anzeigen();
  }

  update() {
    this.kulisse.aktualisieren();
  }

  anzeigen() {
    const inhalt = [];

    Object.values(BIOME).forEach((b) => {
      const [von, bis] = b.level;
      inhalt.push(el('div', { class: 'biom-titel', text: `${b.name} — Level ${von} bis ${bis}` }));
      inhalt.push(
        el(
          'div',
          { class: 'gitter' },
          LEVELS.slice(von - 1, bis).map((lvl) => this.kachel(lvl))
        )
      );
    });

    karteZeigen({
      titel: '🗺 Level auswählen',
      untertitel: `Du hast ${Spielstand.sterneGesamt()} von ${ANZAHL_LEVEL * 3} Sternen gesammelt.`,
      breit: true,
      inhalt,
      knoepfe: [
        {
          text: '↩ Zurück zum Menü',
          klasse: 'grau',
          onClick: () => {
            overlayLeeren();
            this.scene.start('Menu');
          },
        },
      ],
    });
  }

  kachel(lvl) {
    const frei = Spielstand.levelFrei(lvl.nr);
    const erg = Spielstand.ergebnis(lvl.nr);
    const lernBestanden = Spielstand.lerneinheitBestanden(lvl.nr);
    const art = einheitsArt(lvl.nr) === 'lesen' ? '📖' : '🔢';

    return el(
      'button',
      {
        class: `kachel${frei ? '' : ' gesperrt'}`,
        type: 'button',
        title: frei
          ? `${lvl.name} — danach: ${einheitsTitel(lvl.nr)}`
          : 'Erst das Level davor schaffen',
        onClick: () => {
          if (!frei) return;
          tonSpielen('klick');
          overlayLeeren();
          this.scene.start('DinoSelect', { levelNr: lvl.nr });
        },
      },
      [
        el('div', { class: 'nr', text: frei ? String(lvl.nr) : '🔒' }),
        el('div', { class: 'name', text: frei ? lvl.name : `Level ${lvl.nr}` }),
        el('div', { class: 'sterne', text: erg ? sterneText(erg.sterne) : '☆☆☆' }),
        el('div', {
          class: 'sterne',
          style: { fontSize: '13px', color: lernBestanden ? '#1b7f4b' : '#b9b090' },
          text: `${art} ${lernBestanden ? '✔' : '…'}`,
        }),
      ]
    );
  }
}
