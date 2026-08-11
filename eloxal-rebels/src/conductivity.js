/*
 * conductivity.js — THE CORE MECHANIC.
 *
 * Eloxieren funktioniert nur, wo Strom fließt. When a shot hits a power rail
 * (Stromschiene), an arc runs through every touching conductive part. Aluminium
 * and steel conduct; plastic, cardboard and protective film insulate and break
 * the chain. Enemies sitting on an energized part are finished instantly.
 *
 * This module is deliberately PURE: it works on plain data (nodes, edges,
 * boxes) and knows nothing about Matter.js, canvas or the DOM, so it can be
 * unit-tested on its own (see tests/conductivity.test.js). The game layer feeds
 * it the current contact graph and consumes the result.
 */
(function (global) {
  'use strict';

  /**
   * Build undirected adjacency edges from axis-aligned bounding boxes.
   * Two parts are "touching" when their boxes overlap (or nearly touch within
   * `tolerance`). Good enough for stacked, resting fortress parts.
   *
   * @param {Array<{id:string, minX:number, minY:number, maxX:number, maxY:number}>} boxes
   * @param {number} tolerance world units of slack (default 6)
   * @returns {Array<[string,string]>}
   */
  function buildEdges(boxes, tolerance) {
    const t = tolerance == null ? 6 : tolerance;
    const edges = [];
    for (let i = 0; i < boxes.length; i++) {
      for (let j = i + 1; j < boxes.length; j++) {
        const a = boxes[i], b = boxes[j];
        const overlapX = a.minX - t <= b.maxX && b.minX - t <= a.maxX;
        const overlapY = a.minY - t <= b.maxY && b.minY - t <= a.maxY;
        if (overlapX && overlapY) edges.push([a.id, b.id]);
      }
    }
    return edges;
  }

  /**
   * Spread the arc from the source parts through conductive, touching parts.
   * Insulators are never energized and never carry the arc further.
   *
   * @param {Object} params
   * @param {Array<{id:string, conductive:boolean}>} params.nodes
   * @param {Array<[string,string]>} params.edges
   * @param {Array<string>} params.sources ids that carry current (the rails hit)
   * @returns {Set<string>} ids of all energized parts
   */
  function energize({ nodes, edges, sources }) {
    const conductive = new Map();
    nodes.forEach((n) => conductive.set(n.id, !!n.conductive));

    // adjacency list restricted to nodes we know about
    const adj = new Map();
    nodes.forEach((n) => adj.set(n.id, []));
    edges.forEach(([a, b]) => {
      if (adj.has(a) && adj.has(b)) {
        adj.get(a).push(b);
        adj.get(b).push(a);
      }
    });

    const energized = new Set();
    const queue = [];
    // A source only conducts if it is itself conductive (a rail is).
    sources.forEach((s) => {
      if (conductive.get(s)) {
        energized.add(s);
        queue.push(s);
      }
    });

    while (queue.length) {
      const cur = queue.shift();
      const neighbours = adj.get(cur) || [];
      for (const next of neighbours) {
        if (energized.has(next)) continue;
        if (conductive.get(next)) {
          energized.add(next);
          queue.push(next);
        }
        // insulator: chain stops here, it is not added
      }
    }
    return energized;
  }

  /**
   * Decide which enemies die from an arc. An enemy dies when the part it rests
   * on / touches is energized.
   *
   * @param {Object} params
   * @param {Array<{id:string, on:string}>} params.enemies enemy id + carrier part id
   * @param {Set<string>} params.energized result of energize()
   * @returns {Array<string>} ids of enemies that are finished
   */
  function resolveEnemies({ enemies, energized }) {
    return enemies.filter((e) => energized.has(e.on)).map((e) => e.id);
  }

  const api = { buildEdges, energize, resolveEnemies };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api; // Node: for the unit test
  }
  if (typeof global !== 'undefined') {
    const ER = (global.ER = global.ER || {});
    ER.conductivity = api; // Browser: game layer
  }
})(typeof window !== 'undefined' ? window : globalThis);
