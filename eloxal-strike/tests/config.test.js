/* Eloxal Strike — config sanity tests. Run with: node tests/config.test.js
 * Keeps the balancing data well-formed without needing a browser.
 */
'use strict';

const assert = require('assert');
const path = require('path');
const cfg = require(path.join(__dirname, '..', 'src', 'config.js'));

let passed = 0;
function test(name, fn) {
  fn();
  passed++;
  console.log('  ok - ' + name);
}

test('four difficulties with unique ids and sane multipliers', () => {
  assert.strictEqual(cfg.difficulties.length, 4);
  const ids = new Set(cfg.difficulties.map(d => d.id));
  assert.strictEqual(ids.size, 4);
  for (const d of cfg.difficulties) {
    assert.ok(d.name && d.desc, d.id + ' has name and desc');
    for (const k of ['enemyHp', 'enemyDmg', 'enemySpeed', 'spawnMult', 'scoreMult']) {
      assert.ok(d[k] > 0, d.id + '.' + k + ' > 0');
    }
    assert.strictEqual(typeof d.playerRegen, 'boolean');
  }
});

test('difficulties escalate: harder means tougher enemies and more score', () => {
  for (let i = 1; i < cfg.difficulties.length; i++) {
    const prev = cfg.difficulties[i - 1];
    const cur = cfg.difficulties[i];
    assert.ok(cur.enemyHp >= prev.enemyHp, cur.id + ' hp >= ' + prev.id);
    assert.ok(cur.enemyDmg > prev.enemyDmg, cur.id + ' dmg > ' + prev.id);
    assert.ok(cur.scoreMult > prev.scoreMult, cur.id + ' score > ' + prev.id);
  }
  assert.strictEqual(cfg.difficulties[3].playerRegen, false, 'nightmare has no regen');
});

test('weapons are well-formed', () => {
  assert.strictEqual(cfg.weapons.length, 3);
  const slots = new Set(cfg.weapons.map(w => w.slot));
  assert.deepStrictEqual([...slots].sort(), [1, 2, 3]);
  for (const w of cfg.weapons) {
    assert.ok(w.damage > 0 && w.pellets >= 1 && w.fireDelay > 0, w.id + ' ballistics');
    assert.ok(w.magSize > 0 && w.reloadSec > 0 && w.range > 0, w.id + ' handling');
    assert.ok(w.headshotMult > 1, w.id + ' rewards headshots');
  }
});

test('enemy roster is well-formed', () => {
  const types = Object.keys(cfg.enemies);
  assert.ok(types.length >= 4, 'at least 4 enemy types');
  for (const t of types) {
    const e = cfg.enemies[t];
    assert.ok(e.hp > 0 && e.speed > 0 && e.damage > 0, t + ' stats');
    assert.ok(e.attackRange > 0 && e.attackDelay > 0, t + ' attack');
    assert.ok(e.score > 0 && e.radius > 0 && e.height > 0, t + ' body');
    if (e.ranged) { assert.ok(e.projectileSpeed > 0, t + ' projectile speed'); }
  }
  assert.ok(cfg.enemies.korrosius.boss, 'korrosius is the boss');
});

test('waves grow and every 5th wave summons the Baron', () => {
  let prevTotal = 0;
  for (let n = 1; n <= 12; n++) {
    const w = cfg.waveFor(n, 1);
    const total = cfg.totalEnemies(w);
    assert.ok(total >= 1, 'wave ' + n + ' not empty');
    for (const k of Object.keys(w)) {
      assert.ok(Number.isInteger(w[k]) && w[k] >= 0, 'wave ' + n + '.' + k + ' integer count');
    }
    if (n % 5 === 0) {
      assert.ok(w.korrosius >= 1, 'wave ' + n + ' has a boss');
    } else {
      assert.strictEqual(w.korrosius, 0, 'wave ' + n + ' has no boss');
      assert.ok(total >= prevTotal, 'wave ' + n + ' at least as big as previous regular wave');
      prevTotal = total;
    }
  }
});

test('spawnMult scales wave size', () => {
  const easy = cfg.totalEnemies(cfg.waveFor(6, 0.75));
  const normal = cfg.totalEnemies(cfg.waveFor(6, 1));
  const nightmare = cfg.totalEnemies(cfg.waveFor(6, 1.6));
  assert.ok(easy <= normal, 'easy <= normal');
  assert.ok(normal < nightmare, 'normal < nightmare');
});

test('damage falloff: full up close, reduced but never zero at range', () => {
  assert.strictEqual(cfg.falloff(100, 0, 100), 100);
  assert.strictEqual(cfg.falloff(100, 50, 100), 100);
  const mid = cfg.falloff(100, 75, 100);
  assert.ok(mid < 100 && mid > 35, 'mid-range partial damage, got ' + mid);
  assert.ok(Math.abs(cfg.falloff(100, 100, 100) - 35) < 0.001, '35% at max range');
  assert.ok(cfg.falloff(100, 500, 100) > 0, 'never zero');
});

console.log('config.test.js: ' + passed + ' tests passed');
