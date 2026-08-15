import Phaser from 'phaser';

/**
 * Freundliche Hindernis-Gegner.
 *
 * Sie greifen nicht an, sie laufen bzw. schweben nur im Weg herum.
 * Von oben draufhuepfen laesst sie ploppen und gibt Punkte -
 * seitlich beruehren kostet ein Herz (ausser mit Schild oder Ramm-Sprint).
 */
export default class Enemy extends Phaser.Physics.Arcade.Sprite {
  constructor(scene, daten) {
    super(scene, daten.x, daten.y, `gegner_${daten.typ}`, 0);
    scene.add.existing(this);
    scene.physics.add.existing(this);

    this.daten = daten;
    this.typ = daten.typ;
    this.fliegt = !!daten.fliegt;
    this.von = daten.von;
    this.bis = daten.bis;
    this.tempo = daten.tempo || 50;
    this.richtung = 1;
    this.startY = daten.y;
    this.lebt = true;
    this.punkte = 25;

    this.setDepth(15);
    this.body.setSize(32, 24).setOffset(6, 8);
    this.body.setAllowGravity(!this.fliegt);
    if (this.fliegt) this.body.setImmovable(true);
    this.setVelocityX(this.tempo);
    this.anims.play(`gegner_${daten.typ}_lauf`, true);
  }

  preUpdate(zeit, delta) {
    super.preUpdate(zeit, delta);
    if (!this.lebt) return;

    if (this.x <= this.von) {
      this.richtung = 1;
    } else if (this.x >= this.bis) {
      this.richtung = -1;
    }
    this.setVelocityX(this.tempo * this.richtung);
    this.setFlipX(this.richtung < 0);

    if (this.fliegt) {
      this.y = this.startY + Math.sin((zeit + this.x * 4) / 380) * 26;
    }
  }

  /** Von oben geplaettet - der Gegner huepft davon. */
  platt() {
    if (!this.lebt) return 0;
    this.lebt = false;
    this.body.enable = false;
    this.scene.tonSpielen?.('platt');
    this.scene.tweens.add({
      targets: this,
      scaleY: 0.35,
      scaleX: 1.3,
      alpha: 0,
      y: this.y + 12,
      duration: 260,
      onComplete: () => this.destroy(),
    });
    return this.punkte;
  }

  /** Vom Ramm-Sprint erwischt - fliegt zur Seite weg. */
  wegschubsen(richtung) {
    if (!this.lebt) return 0;
    this.lebt = false;
    this.body.enable = false;
    this.scene.tweens.add({
      targets: this,
      x: this.x + richtung * 140,
      y: this.y - 90,
      angle: richtung * 320,
      alpha: 0,
      duration: 520,
      onComplete: () => this.destroy(),
    });
    return this.punkte;
  }
}
