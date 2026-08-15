import Phaser from 'phaser';
import { karteZeigen, el, overlayLeeren, toastZeigen } from '../ui/Dialog.js';
import Spielstand from '../state/storage.js';
import { tonSpielen } from '../audio/sfx.js';

/**
 * Minispiel "Fossilien-Ausgrabung".
 *
 * Teil 1: Mit dem Finger (oder der Maus) den Sand wegpinseln.
 * Teil 2: Die freigelegten Knochen an die richtige Stelle ziehen.
 */

const RASTER_X = 24;
const RASTER_Y = 14;

/** Drei Skelette aus einfachen Bausteinen. */
const FOSSILIEN = [
  {
    id: 'trex_schaedel',
    name: 'T-Rex-Schädel',
    teile: [
      { textur: 'kn_schaedel', x: 0, y: -40, winkel: 0 },
      { textur: 'kn_kiefer', x: 10, y: 10, winkel: 0 },
      { textur: 'kn_wirbel', x: -110, y: -20, winkel: 0 },
      { textur: 'kn_wirbel', x: -160, y: -6, winkel: 10 },
      { textur: 'kn_lang', x: -240, y: 10, winkel: 16 },
      { textur: 'kn_zahn', x: 60, y: 30, winkel: 0 },
    ],
  },
  {
    id: 'stego',
    name: 'Stegosaurus-Rücken',
    teile: [
      { textur: 'kn_lang', x: -180, y: 30, winkel: 0 },
      { textur: 'kn_lang', x: 0, y: 30, winkel: 0 },
      { textur: 'kn_lang', x: 180, y: 40, winkel: 12 },
      { textur: 'kn_platte', x: -110, y: -40, winkel: 0 },
      { textur: 'kn_platte', x: -20, y: -60, winkel: 0 },
      { textur: 'kn_platte', x: 70, y: -46, winkel: 0 },
      { textur: 'kn_wirbel', x: 150, y: -10, winkel: 0 },
    ],
  },
  {
    id: 'ammonit',
    name: 'Ammonit & Rippen',
    teile: [
      { textur: 'kn_schnecke', x: -140, y: 0, winkel: 0 },
      { textur: 'kn_rippe', x: 20, y: -30, winkel: 0 },
      { textur: 'kn_rippe', x: 80, y: -20, winkel: 12 },
      { textur: 'kn_rippe', x: 140, y: -6, winkel: 24 },
      { textur: 'kn_wirbel', x: 60, y: 60, winkel: 0 },
      { textur: 'kn_wirbel', x: 130, y: 70, winkel: 0 },
    ],
  },
];

export default class MinigameScene extends Phaser.Scene {
  constructor() {
    super('Minigame');
  }

  init(daten) {
    this.zurueck = daten?.zurueck || 'Menu';
    this.fossilIndex = daten?.fossil ?? Math.floor(Math.random() * FOSSILIEN.length);
  }

  create() {
    this.knochenTexturen();
    this.fossil = FOSSILIEN[this.fossilIndex % FOSSILIEN.length];
    document.getElementById('touch').classList.add('hidden');
    document.getElementById('hud').replaceChildren();

    const b = this.scale.width;
    const h = this.scale.height;
    this.mitte = { x: b / 2, y: h / 2 - 20 };

    this.add.rectangle(0, 0, b, h, 0x2b1d12).setOrigin(0, 0);
    this.add
      .rectangle(b / 2, h / 2, b - 80, h - 130, 0x7a5433)
      .setStrokeStyle(6, 0x4a3220);

    this.zurueckKnopf();
    this.grabungStarten();
  }

  /** Kleiner Zurueck-Knopf oben rechts. */
  zurueckKnopf() {
    const t = this.add
      .text(this.scale.width - 30, 26, '↩ Zurück', {
        fontFamily: 'Verdana, sans-serif',
        fontSize: '20px',
        color: '#fff4d6',
        backgroundColor: '#00000066',
        padding: { x: 12, y: 7 },
      })
      .setOrigin(1, 0.5)
      .setDepth(30)
      .setInteractive({ useHandCursor: true });
    t.on('pointerup', () => {
      tonSpielen('klick');
      this.scene.start(this.zurueck);
    });
  }

  // ------------------------------------------------------ Knochen-Texturen

  knochenTexturen() {
    const male = (key, w, h, fn) => {
      if (this.textures.exists(key)) return;
      const tex = this.textures.createCanvas(key, w, h);
      const ctx = tex.getContext();
      ctx.lineJoin = 'round';
      ctx.strokeStyle = '#b8a986';
      ctx.lineWidth = 3;
      ctx.fillStyle = '#f4ecd8';
      fn(ctx, w, h);
      tex.refresh();
    };

    // Weicher Pinsel-Stempel zum Wegwischen des Sandes
    if (!this.textures.exists('pinsel')) {
      const tex = this.textures.createCanvas('pinsel', 56, 56);
      const ctx = tex.getContext();
      const g = ctx.createRadialGradient(28, 28, 2, 28, 28, 27);
      g.addColorStop(0, 'rgba(255,255,255,1)');
      g.addColorStop(0.7, 'rgba(255,255,255,0.9)');
      g.addColorStop(1, 'rgba(255,255,255,0)');
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, 56, 56);
      tex.refresh();
    }

    male('kn_lang', 140, 34, (ctx, w, h) => {
      ctx.beginPath();
      ctx.arc(20, 12, 12, 0, Math.PI * 2);
      ctx.arc(20, 24, 12, 0, Math.PI * 2);
      ctx.arc(w - 20, 12, 12, 0, Math.PI * 2);
      ctx.arc(w - 20, 24, 12, 0, Math.PI * 2);
      ctx.rect(20, 8, w - 40, 20);
      ctx.fill();
      ctx.stroke();
    });

    male('kn_schaedel', 150, 90, (ctx, w, h) => {
      ctx.beginPath();
      ctx.ellipse(48, 40, 44, 34, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(78, 22);
      ctx.lineTo(w - 8, 36);
      ctx.lineTo(w - 10, 62);
      ctx.lineTo(76, 60);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = '#4a3220';
      ctx.beginPath();
      ctx.ellipse(44, 32, 11, 9, 0, 0, Math.PI * 2);
      ctx.fill();
    });

    male('kn_kiefer', 130, 34, (ctx, w, h) => {
      ctx.beginPath();
      ctx.moveTo(6, 10);
      ctx.lineTo(w - 6, 16);
      ctx.lineTo(w - 6, 28);
      ctx.lineTo(6, 26);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    });

    male('kn_wirbel', 56, 56, (ctx, w, h) => {
      ctx.beginPath();
      ctx.ellipse(28, 34, 18, 16, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(20, 22);
      ctx.lineTo(28, 4);
      ctx.lineTo(36, 22);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    });

    male('kn_rippe', 60, 110, (ctx, w, h) => {
      ctx.beginPath();
      ctx.lineWidth = 12;
      ctx.strokeStyle = '#f4ecd8';
      ctx.moveTo(46, 8);
      ctx.quadraticCurveTo(4, 54, 34, 102);
      ctx.stroke();
      ctx.lineWidth = 3;
      ctx.strokeStyle = '#b8a986';
      ctx.stroke();
    });

    male('kn_platte', 80, 78, (ctx, w, h) => {
      ctx.beginPath();
      ctx.moveTo(8, h - 6);
      ctx.lineTo(w / 2, 6);
      ctx.lineTo(w - 8, h - 6);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    });

    male('kn_zahn', 40, 56, (ctx, w, h) => {
      ctx.beginPath();
      ctx.moveTo(8, 6);
      ctx.lineTo(32, 8);
      ctx.lineTo(20, h - 6);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    });

    male('kn_schnecke', 120, 120, (ctx, w, h) => {
      ctx.beginPath();
      ctx.lineWidth = 13;
      ctx.strokeStyle = '#f4ecd8';
      for (let i = 0; i < 90; i += 1) {
        const t = i / 12;
        const r = 6 + t * 8;
        const x = 60 + Math.cos(t) * r;
        const y = 60 + Math.sin(t) * r;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = '#b8a986';
      ctx.stroke();
    });
  }

  // ------------------------------------------------------------ Teil 1

  grabungStarten() {
    const b = this.scale.width;
    const h = this.scale.height;

    // Schattenrisse der Knochen unter dem Sand
    this.schatten = this.fossil.teile.map((t) =>
      this.add
        .image(this.mitte.x + t.x, this.mitte.y + t.y, t.textur)
        .setAngle(t.winkel)
        .setTint(0x8d7f60)
        .setDepth(2)
    );

    // Sandschicht als RenderTexture, die weggepinselt wird
    this.sand = this.add.renderTexture(40, 60, b - 80, h - 130).setOrigin(0, 0).setDepth(5);
    this.sand.fill(0xb08c5a, 1);
    for (let i = 0; i < 260; i += 1) {
      this.sand.draw(
        'partikel',
        Math.random() * (b - 80),
        Math.random() * (h - 130),
        0.5,
        0x8a6a3f
      );
    }

    this.raster = new Array(RASTER_X * RASTER_Y).fill(false);
    this.freigelegt = 0;

    this.hinweis = this.add
      .text(b / 2, 28, '🖌️  Wische über den Sand, um die Knochen freizulegen!', {
        fontFamily: 'Verdana, sans-serif',
        fontSize: '22px',
        color: '#fff4d6',
      })
      .setOrigin(0.5, 0.5)
      .setDepth(10);

    this.input.on('pointermove', this.pinseln, this);
    this.input.on('pointerdown', this.pinseln, this);
  }

  pinseln(zeiger) {
    if (!this.sand || this.phase2 || !zeiger.isDown) return;
    const lx = zeiger.worldX - this.sand.x;
    const ly = zeiger.worldY - this.sand.y;
    if (lx < 0 || ly < 0 || lx > this.sand.width || ly > this.sand.height) return;

    this.sand.erase('pinsel', lx - 28, ly - 28);

    // Alle Rasterfelder markieren, die der Pinsel wirklich beruehrt hat
    const zellB = this.sand.width / RASTER_X;
    const zellH = this.sand.height / RASTER_Y;
    const von = { x: Math.floor((lx - 26) / zellB), y: Math.floor((ly - 26) / zellH) };
    const bis = { x: Math.floor((lx + 26) / zellB), y: Math.floor((ly + 26) / zellH) };
    for (let gy = Math.max(0, von.y); gy <= Math.min(RASTER_Y - 1, bis.y); gy += 1) {
      for (let gx = Math.max(0, von.x); gx <= Math.min(RASTER_X - 1, bis.x); gx += 1) {
        const idx = gy * RASTER_X + gx;
        if (this.raster[idx]) continue;
        this.raster[idx] = true;
        this.freigelegt += 1;
        if (this.freigelegt % 14 === 0) tonSpielen('klick');
      }
    }
    if (this.freigelegt / this.raster.length > 0.6) this.zusammensetzenStarten();
  }

  // ------------------------------------------------------------ Teil 2

  zusammensetzenStarten() {
    if (this.phase2) return;
    this.phase2 = true;
    this.input.off('pointermove', this.pinseln, this);
    this.input.off('pointerdown', this.pinseln, this);
    this.tweens.add({
      targets: this.sand,
      alpha: 0,
      duration: 500,
      onComplete: () => this.sand.destroy(),
    });
    this.hinweis.setText('🦴  Ziehe jeden Knochen auf seinen Schatten!');
    tonSpielen('checkpoint');

    const h = this.scale.height;
    this.offen = this.fossil.teile.length;

    // Ablage am unteren Rand: gleichmaessig verteilt, versetzt in zwei Reihen,
    // damit sich die Teile nicht ueberdecken.
    const anzahl = this.fossil.teile.length;
    const abstand = (this.scale.width - 220) / Math.max(1, anzahl - 1);

    this.teile = this.fossil.teile.map((t, i) => {
      const start = {
        x: 110 + i * abstand,
        y: h - 130 + (i % 2) * 58,
      };
      const s = this.add
        .image(start.x, start.y, t.textur)
        .setAngle(t.winkel + (Math.random() * 30 - 15))
        .setDepth(20)
        .setInteractive({ draggable: true, useHandCursor: true });
      s.zielX = this.mitte.x + t.x;
      s.zielY = this.mitte.y + t.y;
      s.zielWinkel = t.winkel;
      s.schatten = this.schatten[i];
      s.gesetzt = false;
      return s;
    });

    this.input.on('drag', (zeiger, obj, x, y) => {
      if (obj.gesetzt) return;
      obj.setPosition(x, y);
    });

    this.input.on('dragend', (zeiger, obj) => {
      if (obj.gesetzt) return;
      const abstand = Phaser.Math.Distance.Between(obj.x, obj.y, obj.zielX, obj.zielY);
      if (abstand < 56) {
        obj.gesetzt = true;
        obj.disableInteractive();
        this.tweens.add({
          targets: obj,
          x: obj.zielX,
          y: obj.zielY,
          angle: obj.zielWinkel,
          duration: 200,
        });
        obj.schatten.setTint(0xffffff).setAlpha(0.35);
        tonSpielen('richtig');
        this.offen -= 1;
        if (this.offen <= 0) this.fertig();
      } else {
        tonSpielen('falsch');
      }
    });
  }

  fertig() {
    tonSpielen('ziel');
    const schonGefunden = Spielstand.get().fossilien[this.fossil.id]?.fertig;
    Spielstand.fossilSpeichern(this.fossil.id, { fertig: true });

    this.time.delayedCall(400, () => {
      karteZeigen({
        titel: '🦴 Fossil fertig!',
        untertitel: `Du hast das Skelett "${this.fossil.name}" zusammengesetzt, ${
          Spielstand.get().spielerName
        }!`,
        inhalt: [
          el('div', {
            class: 'hinweis gut',
            text: schonGefunden
              ? 'Dieses Fossil hattest du schon einmal - Übung macht den Meister!'
              : 'Neues Fossil für deine Sammlung. Forscherinnen arbeiten genau so: erst pinseln, dann puzzeln.',
          }),
          el('div', {
            class: 'werte',
            html: Object.keys(Spielstand.get().fossilien)
              .map((f) => `<span>✔ ${f}</span>`)
              .join(''),
          }),
        ],
        knoepfe: [
          {
            text: '🦴 Noch eine Ausgrabung',
            onClick: () => {
              overlayLeeren();
              this.scene.restart({
                zurueck: this.zurueck,
                fossil: (this.fossilIndex + 1) % FOSSILIEN.length,
              });
            },
          },
          {
            text: '↩ Zurück',
            klasse: 'grau',
            onClick: () => {
              overlayLeeren();
              this.scene.start(this.zurueck);
            },
          },
        ],
      });
    });
  }
}
