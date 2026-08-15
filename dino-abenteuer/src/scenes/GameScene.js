import Phaser from 'phaser';
import Player from '../entities/Player.js';
import Enemy from '../entities/Enemy.js';
import Collectible from '../entities/Collectible.js';
import { levelBauen, GRAVITATION, BODEN_Y, WELT_H } from '../game/levelGenerator.js';
import { getLevel, ANZAHL_LEVEL, BIOME } from '../data/levels.js';
import { getDino, ABILITIES } from '../data/dinos.js';
import { kulisseAufbauen } from '../gfx/kulisse.js';
import Hud from '../ui/Hud.js';
import TouchControls from '../ui/TouchControls.js';
import Spielstand from '../state/storage.js';
import { tonSpielen } from '../audio/sfx.js';
import { karteZeigen, el, overlayLeeren, toastZeigen, sterneText } from '../ui/Dialog.js';
import { lerneinheitStarten } from '../learning/QuizController.js';
import { dinoVorschau } from '../gfx/dinoArt.js';

/** Das eigentliche Jump-and-Run. */
export default class GameScene extends Phaser.Scene {
  constructor() {
    super('Game');
  }

  init(daten) {
    this.levelNr = daten?.levelNr || 1;
    this.dinoId = daten?.dinoId || Spielstand.gewaehlterDino();
    this.punkte = 0;
    this.eierGesammelt = 0;
    this.fruechteGesammelt = 0;
    this.startZeit = 0;
    this.beendet = false;
    this.pausiert = false;
  }

  create() {
    this.cfg = getLevel(this.levelNr);
    this.dino = getDino(this.dinoId);
    this.L = levelBauen(this.cfg);

    this.physics.world.setBounds(0, -240, this.L.breite, WELT_H + 640);
    this.physics.world.gravity.y = GRAVITATION;
    this.cameras.main.setBounds(0, 0, this.L.breite, WELT_H);

    this.kulisse = kulisseAufbauen(this, this.cfg.biom);

    this.gruppenAnlegen();
    this.weltBauen();
    this.spielerAnlegen();
    this.zusammenstoesse();
    this.effekteAnlegen();

    this.hud = new Hud({ onPause: () => this.pause() });
    this.hud.sichtbar(true);
    this.touch = new TouchControls();
    this.touch.sichtbar(true);
    this.tastenAnlegen();

    this.checkpointPos = { x: this.L.start.x, y: this.L.start.y };
    this.startZeit = this.time.now;
    this.letzterSprung = false;
    this.letzterSpezial = false;

    this.cameras.main.startFollow(this.spieler, true, 0.12, 0.12, 0, 60);
    this.cameras.main.fadeIn(300);

    toastZeigen(`Level ${this.levelNr}: ${this.cfg.name}`, 2200);

    this.events.once('shutdown', () => {
      this.touch?.zuruecksetzen();
      document.getElementById('touch').classList.add('hidden');
      this.hud?.zerstoeren();
    });
  }

  // ------------------------------------------------------------- Aufbau

  gruppenAnlegen() {
    this.festeWelt = this.physics.add.staticGroup();
    this.broeckelGruppe = this.physics.add.staticGroup();
    this.felsenGruppe = this.physics.add.staticGroup();
    this.federnGruppe = this.physics.add.staticGroup();
    this.moverGruppe = this.physics.add.group({ allowGravity: false, immovable: true });
    this.gegnerGruppe = this.physics.add.group();
    this.sammelGruppe = this.physics.add.staticGroup();
    this.feuerGruppe = this.physics.add.group();
    this.zonen = { wasser: [], leitern: [] };
  }

  /** Erzeugt einen unsichtbaren statischen Koerper (Kollision) an dieser Stelle. */
  koerper(gruppe, x, y, w, h) {
    const zone = this.add.zone(x + w / 2, y + h / 2, w, h);
    this.physics.add.existing(zone, true);
    gruppe.add(zone);
    return zone;
  }

  weltBauen() {
    const b = this.cfg.biom;
    const farben = BIOME[b].farben;

    // Abgruende sichtbar machen (dunkler Schacht hinter den Luecken)
    (this.L.luecken || []).forEach((l) => {
      this.add
        .rectangle(l.x, BODEN_Y - 6, l.w, WELT_H - BODEN_Y + 6, 0x101820, 0.55)
        .setOrigin(0, 0)
        .setDepth(3);
    });

    // Boden: oben eine Grasnarbe (Kachel), darunter durchgehende Erde
    this.L.boden.forEach((s) => {
      const h = WELT_H - s.y;
      if (h > 64) {
        this.add
          .rectangle(s.x, s.y + 64, s.w, h - 64, Phaser.Display.Color.HexStringToColor(farben.boden).color)
          .setOrigin(0, 0)
          .setDepth(4);
      }
      this.add
        .tileSprite(s.x, s.y, s.w, Math.min(64, h), `boden_${b}`)
        .setOrigin(0, 0)
        .setDepth(5);
      this.koerper(this.festeWelt, s.x, s.y, s.w, h);
    });

    // Plattformen
    this.L.plattformen.forEach((p) => {
      this.add.tileSprite(p.x, p.y, p.w, 32, `plattform_${b}`).setOrigin(0, 0).setDepth(6);
      this.koerper(this.festeWelt, p.x, p.y, p.w, 32);
    });

    // Kletterwaende (fest + kletterbar)
    this.L.kletterwaende.forEach((w) => {
      this.add.tileSprite(w.x, w.y, w.w, w.h, 'kletterwand').setOrigin(0, 0).setDepth(4);
      this.zonen.leitern.push(new Phaser.Geom.Rectangle(w.x - 10, w.y, w.w + 20, w.h));
    });

    // Lianen (nur kletterbar, keine Kollision)
    this.L.lianen.forEach((l) => {
      this.add.tileSprite(l.x - 9, l.y, 18, l.h, 'liane').setOrigin(0, 0).setDepth(4);
      this.zonen.leitern.push(new Phaser.Geom.Rectangle(l.x - 16, l.y, 32, l.h));
    });

    // Wasser
    this.L.wasser.forEach((w) => {
      this.add.tileSprite(w.x, w.y + 12, w.w, w.h, 'wasser').setOrigin(0, 0).setDepth(18).setAlpha(0.75);
      const welle = this.add
        .tileSprite(w.x, w.y, w.w, 14, 'wasser_oben')
        .setOrigin(0, 0)
        .setDepth(19);
      this.tweens.add({ targets: welle, alpha: 0.6, duration: 1400, yoyo: true, repeat: -1 });
      this.zonen.wasser.push(new Phaser.Geom.Rectangle(w.x, w.y, w.w, w.h));
    });

    // Broeckelnde Plattformen
    this.L.broeckel.forEach((p) => {
      const s = this.broeckelGruppe.create(p.x + p.w / 2, p.y, 'broeckel');
      s.setDisplaySize(p.w, 20).setDepth(7);
      s.body.setSize(p.w, 20);
      s.body.updateFromGameObject();
      s.gefallen = false;
      s.startY = p.y;
    });

    // Sprungfedern
    this.L.federn.forEach((f) => {
      const s = this.federnGruppe.create(f.x, f.y, 'feder_aus');
      s.setDepth(7);
      s.kraft = f.kraft;
      s.body.setSize(44, 16);
      s.body.updateFromGameObject();
    });

    // Felsbloecke
    this.L.felsen.forEach((f) => {
      const s = this.felsenGruppe.create(f.x, f.y, 'fels');
      s.setDepth(7);
      s.body.setSize(40, 40);
      s.body.updateFromGameObject();
    });

    // Bewegliche Plattformen: Treibholz (waagerecht) und Hebebuehnen (senkrecht)
    this.L.treibhoelzer.forEach((t) => {
      const s = this.moverGruppe.create(t.x, t.y, 'treibholz');
      s.setDepth(8);
      s.body.setSize(104, 18).setOffset(2, 4);
      s.achse = 'x';
      s.von = t.von;
      s.bis = t.bis;
      s.tempo = t.tempo;
      s.setVelocityX(t.tempo * t.richtung);
    });

    this.L.hebebuehnen.forEach((hb) => {
      const s = this.moverGruppe.create(hb.x, hb.y, 'hebebuehne');
      s.setDepth(8);
      s.body.setSize(92, 18).setOffset(2, 2);
      s.achse = 'y';
      s.von = hb.von;
      s.bis = hb.bis;
      s.tempo = hb.tempo;
      s.setVelocityY(-hb.tempo);
    });

    // Feuerball-Schlote
    this.L.feuerbaelle.forEach((fb) => {
      this.add.ellipse(fb.x, fb.y + 6, 46, 16, 0x8a3a20).setDepth(6);
      this.time.delayedCall(fb.verzoegerung, () => {
        this.time.addEvent({
          delay: fb.intervall,
          loop: true,
          callback: () => this.feuerballSpucken(fb),
        });
      });
    });

    // Sammelobjekte
    this.L.eier.forEach((e) => {
      this.sammelGruppe.add(new Collectible(this, e.x, e.y, 'ei', 0, !!e.bonus));
    });
    this.L.fruechte.forEach((f) => {
      this.sammelGruppe.add(new Collectible(this, f.x, f.y, 'frucht', f.sorte));
    });
    this.eierGesamt = this.L.eier.length;
    this.fruechteGesamt = this.L.fruechte.length;

    // Gegner
    this.L.gegner.forEach((g) => this.gegnerGruppe.add(new Enemy(this, g)));

    // Checkpoints
    this.checkpoints = this.L.checkpoints.map((c) => {
      const s = this.physics.add.staticImage(c.x, c.y, 'fahne').setDepth(9);
      s.aktiv = false;
      return s;
    });

    // Ziel
    this.ziel = this.physics.add.staticImage(this.L.ziel.x, this.L.ziel.y, 'nest').setDepth(9);
    this.tweens.add({
      targets: this.ziel,
      scaleX: 1.06,
      scaleY: 0.94,
      duration: 900,
      yoyo: true,
      repeat: -1,
    });

    // Rauchpartikel im Vulkan
    if (this.cfg.rauch) {
      this.add
        .particles(0, 0, 'partikel', {
          x: { min: 0, max: this.L.breite },
          y: BODEN_Y + 10,
          speedY: { min: -40, max: -14 },
          speedX: { min: -18, max: 18 },
          scale: { start: 0.9, end: 3.2 },
          alpha: { start: 0.22, end: 0 },
          lifespan: 5200,
          frequency: 220,
          tint: 0x9b6b55,
        })
        .setDepth(3);
    }
  }

  spielerAnlegen() {
    this.spieler = new Player(this, this.L.start.x, this.L.start.y, this.dino);
    this.spieler.setCollideWorldBounds(true);
  }

  zusammenstoesse() {
    const s = this.spieler;
    this.physics.add.collider(s, this.festeWelt);
    this.physics.add.collider(this.gegnerGruppe, this.festeWelt);
    this.physics.add.collider(this.gegnerGruppe, this.moverGruppe);

    this.physics.add.collider(s, this.broeckelGruppe, (spieler, platte) =>
      this.broeckelBeruehrt(platte)
    );
    this.physics.add.collider(s, this.moverGruppe);
    this.physics.add.collider(s, this.felsenGruppe, (spieler, fels) =>
      this.felsBeruehrt(fels)
    );
    this.physics.add.collider(s, this.federnGruppe, (spieler, feder) =>
      this.federBeruehrt(feder)
    );

    this.physics.add.overlap(s, this.sammelGruppe, (spieler, obj) => this.einsammeln(obj));
    this.physics.add.overlap(s, this.gegnerGruppe, (spieler, gegner) =>
      this.gegnerBeruehrt(gegner)
    );
    this.physics.add.overlap(s, this.feuerGruppe, (spieler, ball) => {
      ball.destroy();
      spieler.schaden(ball.x);
    });
    this.checkpoints.forEach((c) =>
      this.physics.add.overlap(s, c, () => this.checkpointErreicht(c))
    );
    this.physics.add.overlap(s, this.ziel, () => this.levelGeschafft());
  }

  effekteAnlegen() {
    this.staub = this.add
      .particles(0, 0, 'partikel', {
        speed: { min: 30, max: 90 },
        scale: { start: 0.55, end: 0 },
        alpha: { start: 0.7, end: 0 },
        lifespan: 420,
        emitting: false,
      })
      .setDepth(19);

    this.funken = this.add
      .particles(0, 0, 'funke', {
        speed: { min: 60, max: 190 },
        scale: { start: 1, end: 0 },
        lifespan: 500,
        emitting: false,
      })
      .setDepth(22);

    this.blasen = this.add
      .particles(0, 0, 'blase', {
        speedY: { min: -70, max: -30 },
        scale: { start: 0.8, end: 0.2 },
        lifespan: 900,
        emitting: false,
      })
      .setDepth(21);

    this.schildBild = this.add.image(0, 0, 'schild').setDepth(23).setVisible(false);
  }

  tastenAnlegen() {
    this.tasten = this.input.keyboard.addKeys({
      links: Phaser.Input.Keyboard.KeyCodes.LEFT,
      rechts: Phaser.Input.Keyboard.KeyCodes.RIGHT,
      hoch: Phaser.Input.Keyboard.KeyCodes.UP,
      runter: Phaser.Input.Keyboard.KeyCodes.DOWN,
      a: Phaser.Input.Keyboard.KeyCodes.A,
      d: Phaser.Input.Keyboard.KeyCodes.D,
      w: Phaser.Input.Keyboard.KeyCodes.W,
      s: Phaser.Input.Keyboard.KeyCodes.S,
      leer: Phaser.Input.Keyboard.KeyCodes.SPACE,
      shift: Phaser.Input.Keyboard.KeyCodes.SHIFT,
      e: Phaser.Input.Keyboard.KeyCodes.E,
      p: Phaser.Input.Keyboard.KeyCodes.P,
    });
    this.input.keyboard.on('keydown-P', () => this.pause());
    this.input.keyboard.on('keydown-ESC', () => this.pause());
  }

  // ------------------------------------------------------------- Schleife

  update(zeit, delta) {
    if (this.beendet || this.pausiert) return;
    this.kulisse.aktualisieren();

    const e = this.eingabeLesen();
    this.zonenPruefen();
    this.spieler.update(zeit, delta, e);
    this.moverAktualisieren(delta);
    this.sturzPruefen();
    this.schildAktualisieren();

    this.touch.spezialLadestand(this.spieler.faehigkeitLadestand(), this.spieler.faehigkeit.icon);

    this.hud.aktualisieren({
      level: this.levelNr,
      punkte: this.punkte,
      eier: this.eierGesammelt,
      eierGesamt: this.eierGesamt,
      fruechte: this.fruechteGesammelt,
      fruechteGesamt: this.fruechteGesamt,
      herzen: this.spieler.herzen,
      zeit: (zeit - this.startZeit) / 1000,
    });
  }

  eingabeLesen() {
    const t = this.tasten;
    const tc = this.touch.zustand();
    const kletterModus = this.spieler.anLeiter || this.spieler.imWasser;

    const hoch = t.hoch.isDown || t.w.isDown || tc.hoch;
    const runter = t.runter.isDown || t.s.isDown || tc.runter;
    // Beim Klettern/Schwimmen soll "hoch" nicht gleichzeitig springen.
    const sprung = t.leer.isDown || tc.sprung || (!kletterModus && (t.hoch.isDown || t.w.isDown));
    const spezial = t.shift.isDown || t.e.isDown || tc.spezial;

    const eingabe = {
      links: t.links.isDown || t.a.isDown || tc.links,
      rechts: t.rechts.isDown || t.d.isDown || tc.rechts,
      hoch,
      runter,
      sprung,
      spezial,
      sprungGedrueckt: sprung && !this.letzterSprung,
      spezialGedrueckt: spezial && !this.letzterSpezial,
    };
    this.letzterSprung = sprung;
    this.letzterSpezial = spezial;
    return eingabe;
  }

  /** Wasser- und Kletterzonen sind einfache Rechteck-Tests. */
  zonenPruefen() {
    const s = this.spieler;
    const p = new Phaser.Geom.Point(s.x, s.y);

    const warImWasser = s.imWasser;
    s.imWasser = this.zonen.wasser.some((r) => Phaser.Geom.Rectangle.ContainsPoint(r, p));
    if (s.imWasser && !warImWasser) {
      tonSpielen('wasser');
      this.blasen.emitParticleAt(s.x, s.y, 6);
    }
    if (s.imWasser && Math.random() < 0.04) this.blasen.emitParticleAt(s.x, s.y - 10, 1);

    s.anLeiter = this.zonen.leitern.some((r) => Phaser.Geom.Rectangle.ContainsPoint(r, p));
    if (!s.anLeiter && s.klettert) {
      s.klettert = false;
      s.body.setAllowGravity(true);
    }
  }

  moverAktualisieren(delta) {
    const s = this.spieler;
    this.moverGruppe.children.iterate((m) => {
      if (!m) return;
      if (m.achse === 'x') {
        if (m.x <= m.von) m.setVelocityX(m.tempo);
        else if (m.x >= m.bis) m.setVelocityX(-m.tempo);
      } else {
        if (m.y <= m.von) m.setVelocityY(m.tempo);
        else if (m.y >= m.bis) m.setVelocityY(-m.tempo);
      }
      // Spieler mitnehmen, wenn er oben draufsteht
      const stehtDrauf =
        (s.body.blocked.down || s.body.touching.down) &&
        Math.abs(s.y + s.body.halfHeight - (m.y - m.body.halfHeight)) < 26 &&
        Math.abs(s.x - m.x) < m.body.halfWidth + 20;
      if (stehtDrauf) {
        s.x += (m.body.velocity.x * delta) / 1000;
        if (m.achse === 'y') s.y += (m.body.velocity.y * delta) / 1000;
      }
    });
  }

  sturzPruefen() {
    if (this.spieler.y > WELT_H + 80 && this.spieler.lebt) {
      this.spieler.herzen -= 1;
      tonSpielen('aua');
      if (this.spieler.herzen <= 0) this.spielerVerloren();
      else this.zumCheckpoint();
    }
  }

  schildAktualisieren() {
    const an = this.spieler.istGeschuetzt;
    this.schildBild.setVisible(an);
    if (an) this.schildBild.setPosition(this.spieler.x, this.spieler.y);
  }

  // ---------------------------------------------------------- Interaktion

  einsammeln(obj) {
    const wert = obj.einsammeln();
    if (!wert) return;
    this.punkte += wert;
    if (obj.art === 'ei') {
      this.eierGesammelt += 1;
      tonSpielen('ei');
    } else {
      this.fruechteGesammelt += 1;
      tonSpielen('sammeln');
    }
    this.funken.emitParticleAt(obj.x, obj.y, 4);
  }

  gegnerBeruehrt(gegner) {
    const s = this.spieler;
    if (!gegner.lebt || !s.lebt) return;

    if (s.rammt) {
      this.punkte += gegner.wegschubsen(s.blickRichtung);
      this.funken.emitParticleAt(gegner.x, gegner.y, 8);
      return;
    }
    const vonOben = s.body.velocity.y > 60 && s.y < gegner.y - 8;
    if (vonOben) {
      this.punkte += gegner.platt();
      s.setVelocityY(-360);
      this.staub.emitParticleAt(gegner.x, gegner.y, 6);
      return;
    }
    if (s.istGeschuetzt) {
      this.funken.emitParticleAt(s.x, s.y, 4);
      return;
    }
    s.schaden(gegner.x);
  }

  felsBeruehrt(fels) {
    const s = this.spieler;
    if (!s.kannFelsenBrechen) return;
    this.felsZerbrechen(fels);
  }

  felsZerbrechen(fels) {
    if (!fels.active) return;
    this.staub.emitParticleAt(fels.x, fels.y, 10);
    this.funken.emitParticleAt(fels.x, fels.y, 6);
    this.punkte += 30;
    tonSpielen('platt');
    fels.destroy();
  }

  federBeruehrt(feder) {
    const s = this.spieler;
    if (s.body.velocity.y < 0) return;
    s.setVelocityY(-feder.kraft);
    s.doppelSprungFrei = s.dino.ability === 'glide';
    feder.setTexture('feder');
    this.time.delayedCall(220, () => feder.active && feder.setTexture('feder_aus'));
    tonSpielen('feder');
  }

  broeckelBeruehrt(platte) {
    if (platte.gefallen || !this.spieler.body.blocked.down) return;
    platte.gefallen = true;
    this.tweens.add({
      targets: platte,
      x: platte.x + 3,
      duration: 60,
      yoyo: true,
      repeat: 7,
      onComplete: () => {
        this.staub.emitParticleAt(platte.x, platte.y, 8);
        platte.disableBody(true, true);
        this.time.delayedCall(4000, () => {
          platte.enableBody(true, platte.x, platte.startY, true, true);
          platte.gefallen = false;
          platte.setAlpha(1);
        });
      },
    });
  }

  feuerballSpucken(fb) {
    if (this.beendet) return;
    // Nur in Kameranaehe zuenden - spart Rechenzeit
    if (Math.abs(fb.x - this.cameras.main.scrollX - this.cameras.main.width / 2) > 900) return;
    const ball = this.feuerGruppe.create(fb.x, fb.y, 'feuerball');
    ball.setDepth(17);
    ball.body.setCircle(11, 2, 2);
    ball.setVelocityY(-fb.kraft);
    this.funken.emitParticleAt(fb.x, fb.y, 4);
    this.time.delayedCall(4000, () => ball.active && ball.destroy());
  }

  checkpointErreicht(c) {
    if (c.aktiv) return;
    c.aktiv = true;
    c.setTexture('fahne_aktiv');
    this.checkpointPos = { x: c.x, y: c.y - 40 };
    this.punkte += 25;
    tonSpielen('checkpoint');
    toastZeigen('Checkpoint erreicht! 🚩', 1400);
  }

  zumCheckpoint() {
    this.spieler.setPosition(this.checkpointPos.x, this.checkpointPos.y);
    this.spieler.setVelocity(0, 0);
    this.spieler.unverwundbarBis = this.time.now + 1200;
    this.cameras.main.flash(200, 255, 255, 255);
  }

  spielerVerloren() {
    this.spieler.beleben(this.checkpointPos.x, this.checkpointPos.y);
    this.cameras.main.flash(250, 255, 120, 120);
    toastZeigen('Keine Sorge - weiter geht es vom Checkpoint! 💪', 1800);
  }

  // -------------------------------------------------------------- Effekte

  staubwolke(x, y) {
    this.staub?.emitParticleAt(x, y, 5);
  }

  effektSpurt(s) {
    this.funken.emitParticleAt(s.x, s.y, 6);
  }

  effektRammen(s) {
    this.staub.emitParticleAt(s.x - s.blickRichtung * 20, s.y + 10, 8);
    this.cameras.main.shake(160, 0.004);
  }

  effektSchild(s, dauer) {
    this.schildBild.setVisible(true);
    toastZeigen('Panzerschild an! 🛡️', 1200);
  }

  effektSchmettern(s) {
    this.staub.emitParticleAt(s.x + s.blickRichtung * 34, s.y + 6, 8);
    this.cameras.main.shake(140, 0.004);
    this.felsenGruppe.getChildren().slice().forEach((fels) => {
      const dx = fels.x - s.x;
      const dy = Math.abs(fels.y - s.y);
      if (dy < 60 && dx * s.blickRichtung > -20 && Math.abs(dx) < 86) this.felsZerbrechen(fels);
    });
  }

  tonSpielen(name) {
    tonSpielen(name);
  }

  // ------------------------------------------------------------ Pausieren

  pause() {
    if (this.beendet || this.pausiert) return;
    this.pausiert = true;
    this.physics.pause();
    this.touch.zuruecksetzen();

    karteZeigen({
      titel: '⏸ Pause',
      untertitel: `Level ${this.levelNr}: ${this.cfg.name}`,
      inhalt: [
        el('div', { class: 'werte' }, [
          el('span', { text: `⭐ ${this.punkte} Punkte` }),
          el('span', { text: `🥚 ${this.eierGesammelt}/${this.eierGesamt}` }),
          el('span', { text: `🍎 ${this.fruechteGesammelt}/${this.fruechteGesamt}` }),
        ]),
        el('div', {
          class: 'hinweis',
          text: `${this.dino.name} kann: ${this.spieler.faehigkeit.icon} ${this.spieler.faehigkeit.name} — ${this.spieler.faehigkeit.hint}`,
        }),
      ],
      knoepfe: [
        {
          text: '▶ Weiterspielen',
          onClick: () => {
            overlayLeeren();
            this.pausiert = false;
            this.physics.resume();
          },
        },
        {
          text: '🔁 Level neu starten',
          klasse: 'gelb',
          onClick: () => {
            overlayLeeren();
            this.scene.restart({ levelNr: this.levelNr, dinoId: this.dinoId });
          },
        },
        {
          text: '🗺 Levelauswahl',
          klasse: 'blau',
          onClick: () => {
            overlayLeeren();
            this.scene.start('LevelSelect');
          },
        },
      ],
    });
  }

  // ------------------------------------------------------------- Abschluss

  levelGeschafft() {
    if (this.beendet) return;
    this.beendet = true;
    this.physics.pause();
    this.spieler.setVelocity(0, 0);
    this.touch.zuruecksetzen();
    this.touch.sichtbar(false);
    tonSpielen('ziel');
    this.funken.emitParticleAt(this.ziel.x, this.ziel.y - 20, 30);

    const zeit = Math.round((this.time.now - this.startZeit) / 1000);
    const alleEier = this.eierGesammelt >= this.eierGesamt;
    const inZeit = zeit <= this.cfg.zielzeit;
    const sterne = 1 + (alleEier ? 1 : 0) + (inZeit ? 1 : 0);
    const zeitBonus = Math.max(0, this.cfg.zielzeit - zeit) * 3;
    const gesamtPunkte = this.punkte + zeitBonus + sterne * 100;

    Spielstand.ergebnisSpeichern(this.levelNr, {
      punkte: gesamtPunkte,
      sterne,
      zeit,
      eier: this.eierGesammelt,
      fruechte: this.fruechteGesammelt,
    });

    karteZeigen({
      titel: `🎉 Level ${this.levelNr} geschafft!`,
      untertitel: `Super gemacht, ${Spielstand.get().spielerName}!`,
      inhalt: [
        el('div', { class: 'sterne-gross', text: sterneText(sterne) }),
        el('div', { class: 'werte' }, [
          el('span', { text: `⭐ ${gesamtPunkte} Punkte` }),
          el('span', { text: `🥚 ${this.eierGesammelt}/${this.eierGesamt}` }),
          el('span', { text: `🍎 ${this.fruechteGesammelt}/${this.fruechteGesamt}` }),
          el('span', { text: `⏱ ${zeit}s (Ziel ${this.cfg.zielzeit}s)` }),
        ]),
        el('div', {
          class: 'hinweis',
          text: [
            alleEier ? '✔ Alle Eier gefunden' : '○ Es fehlen noch Eier für den zweiten Stern',
            inZeit ? '✔ In der Zielzeit geschafft' : '○ Etwas schneller gibt den dritten Stern',
          ].join('   ·   '),
        }),
      ],
      knoepfe: [
        {
          text: Spielstand.lerneinheitBestanden(this.levelNr)
            ? '📚 Lernaufgabe wiederholen'
            : '📚 Weiter zur Lernaufgabe',
          onClick: () => {
            overlayLeeren();
            lerneinheitStarten(this.levelNr, (bestanden) => this.nachDerLerneinheit(bestanden));
          },
        },
      ],
    });
  }

  nachDerLerneinheit(bestanden) {
    if (!bestanden) {
      karteZeigen({
        titel: 'Kein Problem!',
        untertitel:
          'Die Lernaufgabe kannst du jederzeit noch einmal machen. Das nächste Level wartet solange auf dich.',
        knoepfe: [
          {
            text: '🔁 Lernaufgabe nochmal',
            onClick: () => {
              overlayLeeren();
              lerneinheitStarten(this.levelNr, (b) => this.nachDerLerneinheit(b));
            },
          },
          {
            text: '🗺 Levelauswahl',
            klasse: 'grau',
            onClick: () => {
              overlayLeeren();
              this.scene.start('LevelSelect');
            },
          },
        ],
      });
      return;
    }

    const neuerDino = Spielstand.dinoFuerLevelFreischalten(this.levelNr);
    Spielstand.levelFreischalten(this.levelNr + 1);
    tonSpielen('richtig');

    const inhalt = [];
    if (neuerDino) {
      const bild = dinoVorschau(neuerDino, 130);
      inhalt.push(
        el('div', { class: 'belohnung' }, [
          bild,
          el('div', {}, [
            el('strong', { text: `Neu: ${neuerDino.name} — ${neuerDino.art}` }),
            el('div', { text: neuerDino.fakt }),
            el('div', {
              text: `Kann: ${ABILITIES[neuerDino.ability].icon} ${
                ABILITIES[neuerDino.ability].name
              }`,
            }),
          ]),
        ])
      );
    }
    const naechstes = this.levelNr + 1;
    const fertig = naechstes > ANZAHL_LEVEL;
    if (fertig) {
      inhalt.push(
        el('div', {
          class: 'hinweis gut',
          text: `Du hast alle ${ANZAHL_LEVEL} Level geschafft, ${
            Spielstand.get().spielerName
          }! Alle Dinos gehören jetzt dir. 🏆`,
        })
      );
    }

    karteZeigen({
      titel: neuerDino ? '🎁 Ein neuer Dino!' : '⭐ Level freigeschaltet',
      untertitel: fertig
        ? 'Das ganze Abenteuer ist geschafft!'
        : `Level ${naechstes} ist jetzt offen.`,
      inhalt,
      knoepfe: [
        !fertig && {
          text: `▶ Level ${naechstes} spielen`,
          onClick: () => {
            overlayLeeren();
            this.scene.start('DinoSelect', { levelNr: naechstes });
          },
        },
        {
          text: '🗺 Levelauswahl',
          klasse: 'blau',
          onClick: () => {
            overlayLeeren();
            this.scene.start('LevelSelect');
          },
        },
        {
          text: '🦴 Fossilien ausgraben',
          klasse: 'gelb',
          onClick: () => {
            overlayLeeren();
            this.scene.start('Minigame', { zurueck: 'LevelSelect' });
          },
        },
      ].filter(Boolean),
    });
  }
}
