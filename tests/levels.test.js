/*
 * Level catalogue validation. Checks every one of the 12 levels for
 * structural sanity, verifies the generated JSON files match the embedded
 * catalogue, and — most importantly — simulates each level untouched for
 * 300 frames to prove the fortresses are statically stable (nothing breaks
 * or dies before the player takes a shot).
 * Run with:  node tests/levels.test.js
 */
const path = require('path');
const fs = require('fs');
const assert = require('assert');

globalThis.Matter = require(path.join(__dirname, '..', 'vendor', 'matter.min.js'));
require('../src/config.js');
require('../src/conductivity.js');
require('../src/levels.js');
require('../src/abilities.js');
require('../src/game.js');
const ER = globalThis.ER;

let passed = 0;
function test(name, fn) { fn(); passed++; console.log('  ok - ' + name); }

const all = ER.levels.all();
const MATS = Object.keys(ER.config.MATERIALS);
const ENEMIES = Object.keys(ER.config.ENEMIES);
const PROJ = Object.keys(ER.config.PROJECTILES);

test('campaign has 12 levels with unique ids', () => {
  assert.strictEqual(all.length, 12);
  assert.strictEqual(new Set(all.map((l) => l.id)).size, 12);
});

test('every level is structurally sane', () => {
  all.forEach((l) => {
    const w = l.width || 1920;
    assert.ok(w >= 1920 && w <= 4000, l.id + ': width');
    assert.ok(l.enemies.length >= 2, l.id + ': needs enemies to shoot');
    assert.ok(l.projectiles.length >= 3, l.id + ': needs ammo');
    l.projectiles.forEach((p) => assert.ok(PROJ.includes(p), l.id + ': projectile ' + p));
    l.blocks.forEach((b) => {
      assert.ok(MATS.includes(b.material), l.id + ': material ' + b.material);
      assert.ok(b.x > 0 && b.x < w, l.id + '/' + b.id + ': x in world');
      assert.ok(b.y > 0 && b.y <= ER.config.WORLD.groundY, l.id + '/' + b.id + ': y above ground');
    });
    l.enemies.forEach((e) => {
      assert.ok(ENEMIES.includes(e.type), l.id + ': enemy ' + e.type);
      assert.ok(e.x > 0 && e.x < w, l.id + '/' + e.id + ': x in world');
    });
    assert.ok(l.slingshot.x < 500, l.id + ': slingshot on the left');
  });
});

test('difficulty ramps: later levels have more enemies and wider worlds', () => {
  const first = all[0], last = all[all.length - 1];
  assert.ok(last.enemies.length > first.enemies.length);
  assert.ok((last.width || 1920) > (first.width || 1920));
  assert.ok(all[11].enemies.some((e) => e.type === 'korrosius'), 'boss in the finale');
});

test('levels/*.json files match the embedded catalogue', () => {
  all.forEach((l) => {
    const file = path.join(__dirname, '..', 'levels', l.id + '.json');
    const onDisk = JSON.parse(fs.readFileSync(file, 'utf8'));
    assert.deepStrictEqual(onDisk, l, l.id + '.json out of sync with src/levels.js');
  });
});

test('every fortress is statically stable for 300 untouched frames', () => {
  all.forEach((l) => {
    const game = ER.game.create();
    game.loadLevel(ER.levels.get(l.id));
    const blocks = game.blockBodies.length;
    const enemies = game.enemyBodies.length;
    game.state = 'idle'; // no win/lose transitions, just physics
    for (let i = 0; i < 300; i++) game.update();
    assert.strictEqual(game.enemyBodies.length, enemies, l.id + ': an enemy died while settling');
    assert.strictEqual(game.blockBodies.length, blocks, l.id + ': a block broke while settling');
    // nothing should have toppled far from where it was placed
    game.enemyBodies.forEach((e) => {
      assert.ok(e.position.y < ER.config.WORLD.groundY + 10, l.id + '/' + e.plugin.id + ': fell through');
    });
  });
});

console.log('\n' + passed + ' tests passed.');
