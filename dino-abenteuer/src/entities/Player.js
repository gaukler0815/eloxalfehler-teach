import Phaser from 'phaser';
import { ABILITIES } from '../data/dinos.js';

/**
 * Der spielbare Dino.
 *
 * Kindgerechte Extras: Coyote-Time (kurz nach der Kante darf man noch
 * springen) und Sprung-Puffer (zu frueh gedrueckt zaehlt trotzdem).
 */
export default class Player extends Phaser.Physics.Arcade.Sprite {
  constructor(scene, x, y, dino) {
    super(scene, x, y, `dino_${dino.id}`, 0);
    scene.add.existing(this);
    scene.physics.add.existing(this);

    this.dino = dino;
    this.setCollideWorldBounds(false);
    this.setDepth(20);
    this.body.setSize(34, 40).setOffset(20, 20);
    this.setOrigin(0.5, 0.5);

    // Zustaende
    this.herzen = 3;
    this.maxHerzen = 3;
    this.blickRichtung = 1;
    this.imWasser = false;
    this.anLeiter = false;
    this.klettert = false;
    this.amBoden = false;
    this.doppelSprungFrei = false;
    this.gleitet = false;
    this.rammtBis = 0;
    this.schildBis = 0;
    this.spurtBis = 0;
    this.unverwundbarBis = 0;
    this.faehigkeitBereitAb = 0;
    this.letzterBodenkontakt = 0;
    this.sprungPuffer = 0;
    this.lebt = true;

    this.faehigkeit = ABILITIES[dino.ability];
    this.setName(dino.name);

    // Dunkle Kontur hinter dem Dino: Sie hebt ihn von gruenen Baeumen,
    // dunklen Hoehlen und rotem Vulkangestein gleichermassen ab.
    this.kontur = scene.add
      .sprite(x, y, `dino_${dino.id}`, 0)
      .setDepth(this.depth - 1)
      .setTint(0x101820)
      .setAlpha(0.45)
      .setScale(1.16);
  }

  /** Kontur dem Dino nachfuehren (Frame, Blickrichtung, Position). */
  konturNachfuehren() {
    if (!this.kontur) return;
    this.kontur.setPosition(this.x, this.y);
    this.kontur.setFrame(this.frame.name);
    this.kontur.setFlipX(this.flipX);
    this.kontur.setAlpha(this.visible ? this.alpha * 0.45 : 0);
  }

  destroy(fromScene) {
    this.kontur?.destroy();
    this.kontur = null;
    super.destroy(fromScene);
  }

  get istGeschuetzt() {
    return this.scene.time.now < this.schildBis;
  }

  get istUnverwundbar() {
    return this.scene.time.now < this.unverwundbarBis || this.istGeschuetzt;
  }

  get rammt() {
    return this.scene.time.now < this.rammtBis;
  }

  get kannFelsenBrechen() {
    return this.dino.ability === 'smash' || this.rammt;
  }

  faehigkeitLadestand() {
    const rest = this.faehigkeitBereitAb - this.scene.time.now;
    if (rest <= 0) return 1;
    return 1 - rest / Math.max(1, this.faehigkeit.cooldown);
  }

  // ------------------------------------------------------------ Bewegung

  update(zeit, delta, e) {
    if (!this.lebt) return;

    const amBoden = this.body.blocked.down || this.body.touching.down;
    if (amBoden) {
      this.letzterBodenkontakt = zeit;
      this.doppelSprungFrei = this.dino.ability === 'glide';
      this.gleitet = false;
    }
    this.amBoden = amBoden;

    const tempoFaktor =
      (this.imWasser ? 0.75 * this.dino.swim : 1) * (zeit < this.spurtBis ? 1.85 : 1);
    const tempo = this.dino.speed * tempoFaktor;

    // Links / Rechts
    if (this.rammt) {
      this.setVelocityX(this.blickRichtung * this.dino.speed * 2.1);
    } else if (e.links) {
      this.setVelocityX(-tempo);
      this.blickRichtung = -1;
    } else if (e.rechts) {
      this.setVelocityX(tempo);
      this.blickRichtung = 1;
    } else {
      this.setVelocityX(this.body.velocity.x * (amBoden ? 0.6 : 0.92));
      if (Math.abs(this.body.velocity.x) < 12) this.setVelocityX(0);
    }
    this.setFlipX(this.blickRichtung < 0);

    // Klettern an Liane / Wand
    if (this.anLeiter && (e.hoch || e.runter || this.klettert)) {
      this.klettert = true;
      this.body.setAllowGravity(false);
      const kletterTempo = 150;
      if (e.hoch) this.setVelocityY(-kletterTempo);
      else if (e.runter) this.setVelocityY(kletterTempo);
      else this.setVelocityY(0);
    } else if (this.klettert) {
      this.klettert = false;
      this.body.setAllowGravity(true);
    }

    // Wasser: Auftrieb + Schwimmen
    if (this.imWasser && !this.klettert) {
      this.body.setAllowGravity(false);
      const schwimm = 130 * this.dino.swim;
      if (e.hoch) this.setVelocityY(-schwimm);
      else if (e.runter) this.setVelocityY(schwimm * 0.8);
      else this.setVelocityY(Math.min(60, this.body.velocity.y * 0.9 + 14));
    } else if (!this.klettert) {
      this.body.setAllowGravity(true);
    }

    // Springen (mit Puffer + Coyote-Time)
    if (e.sprungGedrueckt) this.sprungPuffer = zeit + 140;
    const willSpringen = zeit < this.sprungPuffer;
    const darfCoyote = zeit - this.letzterBodenkontakt < 130;

    if (willSpringen) {
      if (this.imWasser) {
        this.setVelocityY(-190 * this.dino.swim);
        this.sprungPuffer = 0;
      } else if (this.klettert) {
        this.klettert = false;
        this.anLeiter = false;
        this.body.setAllowGravity(true);
        this.setVelocityY(-this.dino.jump * 0.85);
        this.sprungPuffer = 0;
      } else if (amBoden || darfCoyote) {
        this.springen(this.dino.jump);
        this.sprungPuffer = 0;
        this.letzterBodenkontakt = 0;
      } else if (this.doppelSprungFrei) {
        this.doppelSprungFrei = false;
        this.springen(this.dino.jump * 0.88);
        this.sprungPuffer = 0;
        this.scene.staubwolke?.(this.x, this.y + 20);
      }
    }

    // Gleitflug: Sprungtaste halten, waehrend man faellt
    this.gleitet =
      this.dino.ability === 'glide' &&
      !amBoden &&
      !this.imWasser &&
      e.sprung &&
      this.body.velocity.y > 40;
    if (this.gleitet) {
      this.setVelocityY(Math.min(this.body.velocity.y, 95));
    }

    // Kurzer Sprung, wenn die Taste losgelassen wird
    if (!e.sprung && this.body.velocity.y < -170 && !this.imWasser && !this.klettert) {
      this.setVelocityY(this.body.velocity.y * 0.86);
    }

    if (e.spezialGedrueckt) this.faehigkeitAusloesen();

    this.animationWaehlen(amBoden);
    this.blinkenAktualisieren(zeit);
    this.konturNachfuehren();
  }

  springen(kraft) {
    this.setVelocityY(-kraft);
    this.scene.tonSpielen?.('sprung');
    this.scene.staubwolke?.(this.x, this.y + 20);
  }

  animationWaehlen(amBoden) {
    const key = this.dino.id;
    if (this.klettert || this.imWasser) {
      this.anims.play(`${key}_lauf`, true);
      this.anims.msPerFrame = 140;
      return;
    }
    if (!amBoden) {
      this.anims.play(`${key}_sprung`, true);
    } else if (Math.abs(this.body.velocity.x) > 20) {
      this.anims.play(`${key}_lauf`, true);
    } else {
      this.anims.play(`${key}_stehen`, true);
    }
  }

  // ------------------------------------------------------- Spezialfaehigkeit

  faehigkeitAusloesen() {
    const jetzt = this.scene.time.now;
    if (jetzt < this.faehigkeitBereitAb) return false;

    switch (this.dino.ability) {
      case 'ram':
        this.rammtBis = jetzt + 420;
        this.unverwundbarBis = Math.max(this.unverwundbarBis, jetzt + 420);
        this.scene.effektRammen?.(this);
        break;
      case 'dash':
        this.spurtBis = jetzt + 450;
        this.scene.effektSpurt?.(this);
        break;
      case 'shield':
        this.schildBis = jetzt + 5000;
        this.scene.effektSchild?.(this, 5000);
        break;
      case 'jump':
        if (!this.amBoden && !this.imWasser) return false;
        this.springen(this.dino.jump * 1.45);
        break;
      case 'swim':
        this.setVelocityX(this.blickRichtung * this.dino.speed * (this.imWasser ? 2.4 : 1.6));
        if (this.imWasser) this.setVelocityY(-40);
        this.scene.effektSpurt?.(this);
        break;
      case 'smash':
        this.scene.effektSchmettern?.(this);
        break;
      case 'glide':
        if (!this.amBoden && this.doppelSprungFrei) {
          this.doppelSprungFrei = false;
          this.springen(this.dino.jump * 0.9);
        } else if (this.amBoden) {
          this.springen(this.dino.jump);
        }
        break;
      default:
        return false;
    }
    this.faehigkeitBereitAb = jetzt + this.faehigkeit.cooldown;
    this.scene.tonSpielen?.('spezial');
    return true;
  }

  // ---------------------------------------------------------------- Schaden

  schaden(vonX = null) {
    if (!this.lebt || this.istUnverwundbar) return false;
    this.herzen -= 1;
    this.unverwundbarBis = this.scene.time.now + 1400;
    const richtung = vonX === null ? -this.blickRichtung : Math.sign(this.x - vonX) || 1;
    this.setVelocity(richtung * 220, -260);
    this.scene.tonSpielen?.('aua');
    if (this.herzen <= 0) {
      this.lebt = false;
      this.scene.spielerVerloren?.();
    }
    return true;
  }

  blinkenAktualisieren(zeit) {
    if (zeit < this.unverwundbarBis && !this.istGeschuetzt) {
      this.setAlpha(Math.floor(zeit / 90) % 2 === 0 ? 0.35 : 1);
    } else {
      this.setAlpha(1);
    }
  }

  beleben(x, y) {
    this.lebt = true;
    this.herzen = this.maxHerzen;
    this.setPosition(x, y);
    this.setVelocity(0, 0);
    this.setAlpha(1);
    this.klettert = false;
    this.imWasser = false;
    this.body.setAllowGravity(true);
    this.unverwundbarBis = this.scene.time.now + 1200;
  }
}
