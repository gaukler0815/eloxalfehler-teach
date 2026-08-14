// Logic tests for the shared simulation modules. Run with: npm test
import test from 'node:test';
import assert from 'node:assert/strict';

import { buildTrack } from '../public/shared/track.js';
import { makeKartState, stepKart } from '../public/shared/kartphysics.js';
import { RaceSim } from '../public/shared/racesim.js';
import { CHARACTERS, itemWeights, ITEMS } from '../public/shared/config.js';

test('track is a closed loop with sane length', () => {
  const track = buildTrack();
  assert.ok(track.length > 1500 && track.length < 6000, `length ${track.length}`);
  const a = track.sample(0);
  const b = track.sample(track.length); // wraps to 0
  assert.ok(Math.hypot(a.x - b.x, a.z - b.z) < 1);
  assert.equal(track.startGrid.length, 8);
  assert.equal(track.itemBoxes.length, 12);
});

test('closestS finds positions along the whole lap', () => {
  const track = buildTrack();
  for (let f = 0; f < 1; f += 0.05) {
    const s = f * track.length;
    const p = track.sample(s);
    const found = track.closestS(p.x, p.z, s + 5);
    const d = Math.abs(found - s);
    assert.ok(Math.min(d, track.length - d) < 3, `s=${s} found=${found}`);
  }
});

test('kart accelerates forward and respects the wall', () => {
  const track = buildTrack();
  const grid = track.startGrid[0];
  const k = makeKartState(grid.x, grid.z, grid.heading);
  k.trackS = grid.s;
  for (let i = 0; i < 120; i++) stepKart(k, { throttle: 1, steer: 0, drift: false }, 1 / 60, track);
  assert.ok(k.speed > 20, `speed ${k.speed}`);
  // Steer hard into the wall for a while – the kart must stay near the road.
  for (let i = 0; i < 600; i++) stepKart(k, { throttle: 1, steer: 1, drift: false }, 1 / 60, track);
  const lat = Math.abs(track.lateralOffset(k.x, k.z, track.closestS(k.x, k.z, k.trackS)));
  assert.ok(lat < 20, `lateral ${lat}`);
});

test('a full bot race finishes with ranks and lap events', () => {
  const track = buildTrack();
  // Deterministic rng so the test cannot flake.
  let seed = 42;
  const rng = () => (seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31;
  const sim = new RaceSim(track, { laps: 1, rng });
  for (let i = 0; i < 8; i++) {
    sim.addRacer({ id: 'bot' + i, name: 'Bot ' + i, charId: CHARACTERS[i].id, isBot: true });
  }
  sim.start();
  const seen = new Set();
  let guard = 0;
  while (sim.phase !== 'finished' && guard++ < 60 * 600) {
    sim.step(1 / 60);
    for (const ev of sim.drainEvents()) seen.add(ev.type);
  }
  assert.equal(sim.phase, 'finished', 'race must end');
  assert.ok(seen.has('go'));
  assert.ok(seen.has('finish'));
  const ranks = sim.racers.map((r) => r.rank).sort((a, b) => a - b);
  assert.deepEqual(ranks, [1, 2, 3, 4, 5, 6, 7, 8]);
});

test('items: pickup, roulette and firing produce projectiles', () => {
  const track = buildTrack();
  const sim = new RaceSim(track, { laps: 3 });
  const r = sim.addRacer({ id: 'p1', name: 'Tester', charId: 'al' });
  sim.start();
  while (sim.phase !== 'racing') sim.step(1 / 60);
  // Teleport onto an item box.
  const box = sim.boxes[0];
  r.kart.x = box.x; r.kart.z = box.z; r.kart.trackS = box.s;
  sim.step(1 / 60);
  assert.ok(r.rouletteT > 0, 'roulette must start');
  for (let i = 0; i < 120; i++) sim.step(1 / 60);
  assert.ok(r.item, 'item assigned after roulette');
  assert.ok(ITEMS[r.item], 'item id is known');
  r.item = 'bolt';
  sim.useItem('p1');
  assert.equal(sim.projectiles.length, 1);
  assert.equal(sim.projectiles[0].type, 'bolt');
});

test('bolt hit spins the victim, shield blocks it', () => {
  const track = buildTrack();
  const sim = new RaceSim(track, { laps: 3 });
  const a = sim.addRacer({ id: 'a', name: 'A', charId: 'al' });
  const b = sim.addRacer({ id: 'b', name: 'B', charId: 'bolle' });
  sim.start();
  while (sim.phase !== 'racing') sim.step(1 / 60);
  // Place B right in front of A and fire.
  const dirX = Math.sin(a.kart.heading), dirZ = Math.cos(a.kart.heading);
  b.kart.x = a.kart.x + dirX * 20; b.kart.z = a.kart.z + dirZ * 20;
  a.item = 'bolt';
  sim.useItem('a');
  for (let i = 0; i < 90; i++) sim.step(1 / 60);
  assert.ok(b.kart.spinT > 0, 'B must be spun out');
  // Again with shield.
  b.kart.spinT = 0; b.kart.shieldT = 5;
  a.item = 'bolt';
  sim.useItem('a');
  for (let i = 0; i < 90; i++) sim.step(1 / 60);
  assert.equal(b.kart.spinT, 0, 'shield must block');
  assert.equal(b.kart.shieldT, 0, 'shield is consumed');
});

test('item weights favour turbo/seeker at the back', () => {
  const front = itemWeights(0, 8);
  const back = itemWeights(7, 8);
  assert.ok(back.turbo > front.turbo);
  assert.ok(back.seeker > front.seeker);
  assert.ok(front.barrel > back.barrel);
});
