/*
 * Unit tests for the conductivity core mechanic.
 * Run with:  node tests/conductivity.test.js
 * No framework, no dependencies — plain assertions so it works anywhere.
 */
const assert = require('assert');
const C = require('../src/conductivity.js');

let passed = 0;
function test(name, fn) {
  fn();
  passed++;
  console.log('  ok - ' + name);
}

// --- buildEdges -----------------------------------------------------------
test('buildEdges links overlapping boxes only', () => {
  const boxes = [
    { id: 'a', minX: 0, minY: 0, maxX: 10, maxY: 10 },
    { id: 'b', minX: 10, minY: 0, maxX: 20, maxY: 10 }, // touches a
    { id: 'c', minX: 100, minY: 0, maxX: 110, maxY: 10 } // far away
  ];
  const edges = C.buildEdges(boxes, 2);
  assert.strictEqual(edges.length, 1);
  assert.deepStrictEqual(edges[0].sort(), ['a', 'b']);
});

// --- energize --------------------------------------------------------------
test('arc runs through a conductive chain', () => {
  const nodes = [
    { id: 'rail', conductive: true },
    { id: 'steel', conductive: true },
    { id: 'alu', conductive: true }
  ];
  const edges = [['rail', 'steel'], ['steel', 'alu']];
  const e = C.energize({ nodes, edges, sources: ['rail'] });
  assert.ok(e.has('rail') && e.has('steel') && e.has('alu'));
  assert.strictEqual(e.size, 3);
});

test('insulator breaks the chain', () => {
  const nodes = [
    { id: 'rail', conductive: true },
    { id: 'plastic', conductive: false }, // Kunststoff isolates
    { id: 'alu', conductive: true }
  ];
  const edges = [['rail', 'plastic'], ['plastic', 'alu']];
  const e = C.energize({ nodes, edges, sources: ['rail'] });
  assert.ok(e.has('rail'));
  assert.ok(!e.has('plastic'));
  assert.ok(!e.has('alu')); // never reached, insulator in between
});

test('a non-conductive source carries nothing', () => {
  const nodes = [{ id: 'carton', conductive: false }];
  const e = C.energize({ nodes, edges: [], sources: ['carton'] });
  assert.strictEqual(e.size, 0);
});

test('branching chain energizes both branches', () => {
  const nodes = [
    { id: 'rail', conductive: true },
    { id: 'l', conductive: true },
    { id: 'r', conductive: true },
    { id: 'iso', conductive: false }
  ];
  const edges = [['rail', 'l'], ['rail', 'r'], ['r', 'iso']];
  const e = C.energize({ nodes, edges, sources: ['rail'] });
  assert.ok(e.has('l') && e.has('r'));
  assert.ok(!e.has('iso'));
});

// --- resolveEnemies --------------------------------------------------------
test('enemies on energized parts die, others survive', () => {
  const energized = new Set(['steel', 'alu']);
  const enemies = [
    { id: 'lenny', on: 'alu' },     // dies
    { id: 'kalki', on: 'plastic' }, // survives (isolated)
    { id: 'fetti', on: 'steel' }    // dies
  ];
  const dead = C.resolveEnemies({ enemies, energized }).sort();
  assert.deepStrictEqual(dead, ['fetti', 'lenny']);
});

// The blueprint level 5-8 "Kontaktstelle": after Bürsti removes the plastic,
// Kalki's alu plate drops onto the steel and the whole chain lights up.
test('blueprint contact-point puzzle resolves after plastic removed', () => {
  const nodes = [
    { id: 'rail', conductive: true },
    { id: 'steeltower', conductive: true },
    { id: 'aluplate', conductive: true }
  ];
  const edges = [['rail', 'steeltower'], ['steeltower', 'aluplate']];
  const energized = C.energize({ nodes, edges, sources: ['rail'] });
  const dead = C.resolveEnemies({
    enemies: [{ id: 'lenny', on: 'aluplate' }, { id: 'kalki', on: 'steeltower' }],
    energized
  });
  assert.strictEqual(dead.length, 2);
});

console.log('\n' + passed + ' tests passed.');
