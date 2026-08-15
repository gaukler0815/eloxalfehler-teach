import Phaser from 'phaser';

/** Punkte pro Sammelobjekt. */
export const PUNKTE = {
  ei: 50,
  bonusEi: 100,
  frucht: 20,
};

/**
 * Sammelbares Objekt (Ei oder Frucht) mit kleiner Schwebe-Animation.
 */
export default class Collectible extends Phaser.Physics.Arcade.Sprite {
  constructor(scene, x, y, art, sorte = 0, bonus = false) {
    const key = art === 'ei' ? 'ei' : `frucht${sorte % 3}`;
    super(scene, x, y, key);
    scene.add.existing(this);
    scene.physics.add.existing(this, true);

    this.art = art;
    this.bonus = bonus;
    this.wert = art === 'ei' ? (bonus ? PUNKTE.bonusEi : PUNKTE.ei) : PUNKTE.frucht;
    this.setDepth(12);
    if (bonus) this.setTint(0xffe680);

    scene.tweens.add({
      targets: this,
      y: y - 7,
      duration: 900 + (Math.abs(Math.round(x)) % 300),
      yoyo: true,
      repeat: -1,
      ease: 'Sine.easeInOut',
    });
  }

  /** Einsammeln: kleine Aufstieg-Animation, dann verschwinden. */
  einsammeln() {
    if (!this.active) return 0;
    const wert = this.wert;
    this.body.enable = false;
    this.scene.tweens.killTweensOf(this);
    this.scene.tweens.add({
      targets: this,
      y: this.y - 42,
      alpha: 0,
      scale: 1.5,
      duration: 300,
      onComplete: () => this.destroy(),
    });
    return wert;
  }
}
