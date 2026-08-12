/* Eloxal Strike — central balancing & game data.
 * Pure data + pure functions only (no DOM, no three.js) so node tests can
 * require() this file directly. Everything gameplay-tunable lives here.
 */
(function () {
  'use strict';

  var config = {
    version: '1.0.0',

    player: {
      hp: 100,
      speed: 5.6,          // m/s walking
      sprintMult: 1.6,
      jumpVel: 7.2,
      gravity: 20,
      eyeHeight: 1.7,
      radius: 0.45,
      regenDelaySec: 4.5,  // seconds without damage before regen starts
      regenPerSec: 9
    },

    // Difficulty presets. All multipliers apply on top of base values.
    difficulties: [
      {
        id: 'azubi',
        name: 'Azubi',
        tagline: 'Erster Tag in der Halle',
        desc: 'Gegner sind träge, Lebensenergie regeneriert schnell. Zum Eingewöhnen.',
        enemyHp: 0.7, enemyDmg: 0.55, enemySpeed: 0.85, spawnMult: 0.75,
        playerRegen: true, scoreMult: 0.8
      },
      {
        id: 'geselle',
        name: 'Geselle',
        tagline: 'Normale Schicht',
        desc: 'Der ausgewogene Standard. Faire Gegner, faire Wertung.',
        enemyHp: 1.0, enemyDmg: 1.0, enemySpeed: 1.0, spawnMult: 1.0,
        playerRegen: true, scoreMult: 1.0
      },
      {
        id: 'schichtleiter',
        name: 'Schichtleiter',
        tagline: 'Doppelschicht, kein Kaffee',
        desc: 'Mehr Gegner, härtere Treffer, langsame Regeneration.',
        enemyHp: 1.35, enemyDmg: 1.5, enemySpeed: 1.12, spawnMult: 1.3,
        playerRegen: true, scoreMult: 1.5
      },
      {
        id: 'korrosius',
        name: 'Korrosius-Modus',
        tagline: 'Der Baron persönlich',
        desc: 'Keine Regeneration. Gnadenlose Horden. Nur für Legenden der Halle.',
        enemyHp: 1.7, enemyDmg: 2.0, enemySpeed: 1.25, spawnMult: 1.6,
        playerRegen: false, scoreMult: 2.2
      }
    ],

    weapons: [
      {
        id: 'anodisierer',
        name: 'Anodisierer MK-1',
        slot: 1,
        auto: false,
        damage: 34,
        headshotMult: 2.0,
        pellets: 1,
        spreadDeg: 0.35,
        fireDelay: 0.24,
        magSize: 12,
        reserve: Infinity,
        reloadSec: 1.1,
        range: 120,
        color: 0x58c7f0
      },
      {
        id: 'streuer',
        name: 'Säure-Streuer',
        slot: 2,
        auto: false,
        damage: 12,
        headshotMult: 1.6,
        pellets: 8,
        spreadDeg: 5.5,
        fireDelay: 0.85,
        magSize: 6,
        reserve: 36,
        reloadSec: 1.9,
        range: 40,
        color: 0x9fe348
      },
      {
        id: 'lichtbogen',
        name: 'Lichtbogen-LMG',
        slot: 3,
        auto: true,
        damage: 11,
        headshotMult: 1.8,
        pellets: 1,
        spreadDeg: 1.6,
        fireDelay: 0.085,
        magSize: 42,
        reserve: 168,
        reloadSec: 2.2,
        range: 90,
        color: 0xffd166
      }
    ],

    enemies: {
      rostling: {
        name: 'Rostling',
        hp: 60, speed: 3.1, damage: 12, attackRange: 1.9, attackDelay: 1.0,
        radius: 0.55, height: 1.4, score: 100, ranged: false,
        color: 0xa8502a, eyeColor: 0xffd166
      },
      brocken: {
        name: 'Blatterbrocken',
        hp: 240, speed: 1.55, damage: 26, attackRange: 2.3, attackDelay: 1.6,
        radius: 0.95, height: 2.3, score: 250, ranged: false,
        color: 0x6e3b1e, eyeColor: 0xff5c39
      },
      sprueher: {
        name: 'Säure-Sprüher',
        hp: 85, speed: 2.3, damage: 16, attackRange: 15, attackDelay: 2.3,
        radius: 0.55, height: 1.6, score: 200, ranged: true,
        projectileSpeed: 14, color: 0x5d7a2a, eyeColor: 0xb7ff4d
      },
      korrosius: {
        name: 'Baron Korrosius',
        hp: 1500, speed: 1.9, damage: 40, attackRange: 3.0, attackDelay: 1.4,
        radius: 1.35, height: 3.4, score: 2000, ranged: false, boss: true,
        color: 0x3f2b1a, eyeColor: 0xff2e2e
      }
    },

    scoring: {
      headshotBonus: 25,
      waveClearBonus: 150
    },

    pickups: {
      health: { amount: 40, respawnAfterWave: true },
      ammo:   { reserveRefill: { streuer: 12, lichtbogen: 84 } }
    },

    intermissionSec: 6,

    /* Composition of a wave. Returns { rostling, sprueher, brocken, korrosius }
     * counts, already scaled by the difficulty's spawnMult. Every 5th wave
     * summons a Baron Korrosius on top of a reduced regular horde. */
    waveFor: function (n, spawnMult) {
      if (n < 1) { n = 1; }
      var m = spawnMult || 1;
      var boss = (n % 5 === 0);
      var base = boss ? 0.6 : 1;
      return {
        rostling: Math.max(1, Math.round((2 + n * 1.6) * m * base)),
        sprueher: Math.round(Math.max(0, n - 1) * 0.8 * m * base),
        brocken: Math.round(Math.max(0, n - 2) * 0.5 * m * base),
        korrosius: boss ? Math.max(1, Math.floor(n / 10) + (n % 10 === 0 ? 0 : 1)) : 0
      };
    },

    totalEnemies: function (wave) {
      return wave.rostling + wave.sprueher + wave.brocken + wave.korrosius;
    },

    /* Hitscan damage falloff: full damage up to half range, then linear decay
     * down to 35% at maximum range. */
    falloff: function (damage, dist, range) {
      if (dist <= range * 0.5) { return damage; }
      if (dist >= range) { return damage * 0.35; }
      var t = (dist - range * 0.5) / (range * 0.5);
      return damage * (1 - t * 0.65);
    }
  };

  if (typeof module !== 'undefined' && module.exports) { module.exports = config; }
  if (typeof window !== 'undefined') {
    window.ES = window.ES || {};
    window.ES.config = config;
  }
})();
