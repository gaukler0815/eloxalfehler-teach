/*
 * Headless integration test for the game engine. Loads the real Matter.js
 * build and the game modules, then drives them with no browser to prove that
 * shots fly, materials break, the arc clears enemies and scoring works.
 * Run with:  node tests/sim.test.js
 */
const path = require('path');
const assert = require('assert');

// Make the browser-style globals the modules expect.
globalThis.Matter = require(path.join(__dirname, '..', 'vendor', 'matter.min.js'));
require('../src/config.js');
require('../src/conductivity.js');
require('../src/levels.js');
require('../src/abilities.js');
require('../src/game.js');
const ER = globalThis.ER;

let passed = 0;
function test(name, fn) { fn(); passed++; console.log('  ok - ' + name); }

function fire(game, dx, dy) {
  game.beginAim(game.anchor.x, game.anchor.y);
  game.moveAim(game.anchor.x + dx, game.anchor.y + dy);
  game.releaseAim();
}
function stepUntilQuiet(game, max) {
  let n = max || 400;
  while (n-- > 0 && game.state === 'flying') game.update();
}
function removeBlock(game, id) {
  const i = game.blockBodies.findIndex((b) => b.plugin.id === id);
  if (i >= 0) { globalThis.Matter.World.remove(game.world, game.blockBodies[i]); game.blockBodies.splice(i, 1); }
}

// --- shots reach the fortress and can win world 1 level 1 -----------------
test('a slingshot shot can clear w1-l01 within its 3 shots', () => {
  let won = false;
  // pull down-left to launch up-right; coarse search over the aim space.
  for (let dx = -280; dx <= -120 && !won; dx += 40) {
    for (let dy = 60; dy <= 280 && !won; dy += 40) {
      const game = ER.game.create();
      game.loadLevel(ER.levels.get('w1-l01'));
      let guard = 6;
      while (game.state === 'aim' && guard-- > 0) {
        fire(game, dx, dy);
        stepUntilQuiet(game);
      }
      if (game.state === 'won') won = true;
    }
  }
  assert.ok(won, 'expected at least one aim to clear w1-l01');
});

// --- a fast projectile breaks light material ------------------------------
test('a fast shot destroys cardboard', () => {
  const game = ER.game.create();
  game.loadLevel(ER.levels.get('w1-l01'));
  const before = game.blockBodies.length;
  // Launch straight at the cardboard stack region with a body placed close.
  const b = game.spawnProjectile('bolle', 1200, 800, 26, 4);
  b.plugin.punch = 3;
  for (let i = 0; i < 120; i++) game.update();
  assert.ok(game.blockBodies.length < before, 'expected at least one block gone');
});

// --- conductivity arc integration on the Kontaktstelle puzzle -------------
test('arc kills only the rail-side enemy until the chain is completed', () => {
  const game = ER.game.create();
  game.loadLevel(ER.levels.get('w5-l01'));
  const rail = game.blockBodies.find((b) => b.plugin.id === 'rail');

  // First arc: only the Stauber on the rail is finished; plate is insulated.
  game.arcFromRail(rail);
  assert.strictEqual(game.enemyBodies.length, 2, 'lenny + kalki should still stand');
  assert.ok(!game.enemyBodies.find((e) => e.plugin.id === 'e3'), 'stauber gone');

  // Remove the insulating cardboard so the alu plate drops onto the steel.
  removeBlock(game, 'spacer');
  for (let i = 0; i < 150; i++) game.update();

  // Now the plate touches the steel -> the arc reaches the enemies on it.
  game.arcFromRail(rail);
  assert.strictEqual(game.enemyBodies.length, 0, 'chain complete: all enemies down');
});

// --- scoring + level end --------------------------------------------------
test('clearing without spending shots scores a perfect 20 µm', () => {
  const game = ER.game.create();
  game.loadLevel(ER.levels.get('w1-l01'));
  let end = null;
  game.on('levelend', (e) => (end = e));
  game.blastArea({ x: 1440, y: 850 }, 4000, 999, 0); // wipe all enemies
  game.update();
  assert.ok(end && end.won === true);
  assert.strictEqual(end.um, ER.config.SCORING.perfect);
});

test('running out of shots with enemies left is a loss', () => {
  const game = ER.game.create();
  game.loadLevel(ER.levels.get('w1-l01'));
  let end = null;
  game.on('levelend', (e) => (end = e));
  game.queue = [];
  game.currentType = null;
  game.state = 'flying';
  for (let i = 0; i < 20; i++) game.update();
  assert.ok(end && end.won === false, 'expected a loss');
});

console.log('\n' + passed + ' tests passed.');
