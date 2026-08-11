/*
 * game.js — the playable core: Matter.js world, slingshot, impacts, scoring,
 * win/lose and the hook into the conductivity module. It carries no rendering
 * and no DOM code, so it can be driven headlessly (see tests/sim.test.js).
 * main.js wires pointer input and the renderer to the small API below.
 */
(function (global) {
  'use strict';
  const ER = (global.ER = global.ER || {});
  const Matter = global.Matter;

  function create() {
    const { WORLD, PHYSICS, MATERIALS, PROJECTILES, ENEMIES, SCORING } = ER.config;
    const M = Matter;

    const engine = M.Engine.create();
    engine.gravity.y = PHYSICS.gravityY;
    const world = engine.world;

    const game = {
      engine, world, M, WORLD,
      level: null,
      anchor: { x: 340, y: 720 },
      queue: [],
      currentType: null,
      shotsTotal: 0,
      shotsUsed: 0,
      aiming: false,
      aimPos: { x: 340, y: 720 },
      state: 'idle', // idle | aim | flying | won | lost
      activeBodies: [],
      blockBodies: [],
      enemyBodies: [],
      pellets: [],
      effects: [],
      arcs: [],
      lasarUsed: false,
      abilityUsedThisShot: false,
      settleCounter: 0,
      um: 0,
      flashText: '',
      flashTtl: 0,
      _listeners: {}
    };

    // --- tiny event bus ----------------------------------------------------
    game.on = (name, cb) => {
      (game._listeners[name] = game._listeners[name] || []).push(cb);
    };
    game.emit = (name, payload) => {
      (game._listeners[name] || []).forEach((cb) => cb(payload));
    };

    // --- geometry helpers --------------------------------------------------
    function boxOf(body) {
      const b = body.bounds;
      return { id: body.plugin.id, minX: b.min.x, minY: b.min.y, maxX: b.max.x, maxY: b.max.y };
    }
    function overlap(a, b, t) {
      return a.minX - t <= b.maxX && b.minX - t <= a.maxX &&
             a.minY - t <= b.maxY && b.minY - t <= a.maxY;
    }
    function bodyBox(body) {
      const b = body.bounds;
      return { minX: b.min.x, minY: b.min.y, maxX: b.max.x, maxY: b.max.y };
    }
    function relSpeed(a, b) {
      return Math.hypot(a.velocity.x - b.velocity.x, a.velocity.y - b.velocity.y);
    }
    function segDist(p, a, b) {
      const dx = b.x - a.x, dy = b.y - a.y;
      const len2 = dx * dx + dy * dy || 1;
      let t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / len2;
      t = Math.max(0, Math.min(1, t));
      return Math.hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy));
    }

    // --- world building ----------------------------------------------------
    function clearWorld() {
      M.Composite.clear(world, false, true);
      game.blockBodies = [];
      game.enemyBodies = [];
      game.activeBodies = [];
      game.pellets = [];
      game.effects = [];
      game.arcs = [];
      // Ground and side walls (static).
      const ground = M.Bodies.rectangle(WORLD.width / 2, WORLD.groundY + 80,
        WORLD.width * 3, 200, { isStatic: true, friction: 0.9, plugin: { kind: 'ground' } });
      const leftWall = M.Bodies.rectangle(-120, WORLD.height / 2, 240, WORLD.height * 3,
        { isStatic: true, plugin: { kind: 'wall' } });
      const rightWall = M.Bodies.rectangle(WORLD.width + 120, WORLD.height / 2, 240, WORLD.height * 3,
        { isStatic: true, plugin: { kind: 'wall' } });
      M.World.add(world, [ground, leftWall, rightWall]);
    }

    function addBlock(spec) {
      const mat = MATERIALS[spec.material];
      const body = M.Bodies.rectangle(spec.x, spec.y, spec.w, spec.h, {
        isStatic: !!mat.rail,
        density: mat.density,
        restitution: mat.restitution,
        friction: mat.friction,
        frictionStatic: 1,
        plugin: {}
      });
      if (spec.angle) M.Body.setAngle(body, (spec.angle * Math.PI) / 180);
      body.plugin = {
        kind: 'block', id: spec.id, material: spec.material, matDef: mat,
        hp: mat.hp, maxHp: mat.hp, w: spec.w, h: spec.h, softened: false
      };
      M.World.add(world, body);
      game.blockBodies.push(body);
      return body;
    }

    function addEnemy(spec) {
      const def = ENEMIES[spec.type] || ENEMIES.stauber;
      const body = M.Bodies.circle(spec.x, spec.y, def.radius, {
        density: 0.0016, restitution: 0.1, friction: 0.6, plugin: {}
      });
      body.plugin = { kind: 'enemy', id: spec.id, type: spec.type, hp: 1 };
      M.World.add(world, body);
      game.enemyBodies.push(body);
      return body;
    }

    game.loadLevel = function (data) {
      game.level = data;
      clearWorld();
      game.anchor = { x: data.slingshot.x, y: data.slingshot.y };
      (data.blocks || []).forEach(addBlock);
      (data.enemies || []).forEach(addEnemy);
      game.queue = (data.projectiles || []).slice();
      game.shotsTotal = game.queue.length;
      game.shotsUsed = 0;
      game.lasarUsed = false;
      game.um = 0;
      game.flashText = '';
      game.flashTtl = 0;
      prepareShot();
    };

    function prepareShot() {
      game.aiming = false;
      game.abilityUsedThisShot = false;
      game.activeBodies = [];
      if (game.queue.length === 0) {
        game.currentType = null;
        game.state = 'flying'; // let update() resolve win/lose once settled
        return;
      }
      game.currentType = game.queue.shift();
      game.aimPos = { x: game.anchor.x, y: game.anchor.y };
      game.state = 'aim';
    }

    // --- projectiles -------------------------------------------------------
    game.spawnProjectile = function (type, x, y, vx, vy, scaleMul) {
      const def = PROJECTILES[type];
      const r = def.radius * (scaleMul || 1);
      const body = M.Bodies.circle(x, y, r, {
        restitution: 0.25, friction: 0.5, frictionAir: 0.006,
        density: 0.0016 * def.mass, plugin: {}
      });
      body.plugin = { kind: 'projectile', type, punch: 1, abilityUsed: false };
      M.Body.setVelocity(body, { x: vx, y: vy });
      M.World.add(world, body);
      game.activeBodies.push(body);
      return body;
    };

    game.spawnPellet = function (x, y, vx, vy) {
      const body = M.Bodies.circle(x, y, 7, {
        restitution: 0.2, friction: 0.4, frictionAir: 0.01,
        density: 0.0008, plugin: {}
      });
      body.plugin = { kind: 'pellet', ttl: PHYSICS.pelletTtl };
      M.Body.setVelocity(body, { x: vx, y: vy });
      M.World.add(world, body);
      game.pellets.push(body);
      return body;
    };

    game.removeActive = function (body) {
      const i = game.activeBodies.indexOf(body);
      if (i >= 0) game.activeBodies.splice(i, 1);
      M.World.remove(world, body);
    };

    // --- input from main.js ------------------------------------------------
    game.beginAim = function (x, y) {
      if (game.state !== 'aim' || !game.currentType) return false;
      if (Math.hypot(x - game.anchor.x, y - game.anchor.y) > PHYSICS.grabRadius) return false;
      game.aiming = true;
      game.moveAim(x, y);
      return true;
    };
    game.moveAim = function (x, y) {
      if (!game.aiming) return;
      let dx = x - game.anchor.x, dy = y - game.anchor.y;
      const d = Math.hypot(dx, dy);
      if (d > PHYSICS.maxPull) { dx = (dx / d) * PHYSICS.maxPull; dy = (dy / d) * PHYSICS.maxPull; }
      game.aimPos = { x: game.anchor.x + dx, y: game.anchor.y + dy };
    };
    game.releaseAim = function () {
      if (!game.aiming) return;
      game.aiming = false;
      const dx = game.anchor.x - game.aimPos.x;
      const dy = game.anchor.y - game.aimPos.y;
      if (Math.hypot(dx, dy) < 24) { game.aimPos = { x: game.anchor.x, y: game.anchor.y }; return; }
      const type = game.currentType;
      game.currentType = null;
      game.spawnProjectile(type, game.aimPos.x, game.aimPos.y,
        dx * PHYSICS.launchPower, dy * PHYSICS.launchPower);
      game.shotsUsed++;
      game.settleCounter = 0;
      game.state = 'flying';
    };
    // A tap/click in flight triggers the ability of the launched projectile.
    game.tap = function () {
      if (game.state !== 'flying' || game.abilityUsedThisShot) return;
      const body = game.activeBodies.find((b) => !b.plugin.noAbility && !b.plugin.abilityUsed);
      if (!body) return;
      body.plugin.abilityUsed = true;
      game.abilityUsedThisShot = true;
      ER.abilities.trigger(game, body);
    };

    // --- ability support ---------------------------------------------------
    game.flashLabel = function (t) { game.flashText = t; game.flashTtl = 90; };
    game.spawnEffect = function (kind, pos, radius, to) {
      game.effects.push({ kind, x: pos.x, y: pos.y, radius: radius || 0, to: to || null, ttl: kind === 'cut' ? 16 : 26, max: kind === 'cut' ? 16 : 26 });
    };
    game.nearestBlockInFront = function (body, range) {
      let best = null, bestD = range;
      game.blockBodies.forEach((b) => {
        if (b.position.x < body.position.x - 40) return;
        const d = Math.hypot(b.position.x - body.position.x, b.position.y - body.position.y);
        if (d < bestD) { bestD = d; best = b; }
      });
      return best;
    };
    game.softenArea = function (pos, radius) {
      game.blockBodies.forEach((b) => {
        if (b.plugin.matDef.rail) return;
        if (Math.hypot(b.position.x - pos.x, b.position.y - pos.y) <= radius) {
          b.plugin.hp = Math.max(4, b.plugin.hp * 0.35);
          b.plugin.softened = true;
        }
      });
    };
    game.blastArea = function (pos, radius, dmg, impulse) {
      const all = game.blockBodies.concat(game.enemyBodies, game.activeBodies);
      all.forEach((b) => {
        const dx = b.position.x - pos.x, dy = b.position.y - pos.y;
        const dist = Math.hypot(dx, dy);
        if (dist > radius) return;
        const fall = 1 - dist / radius;
        if (!b.isStatic && impulse) {
          const n = dist || 1;
          M.Body.applyForce(b, b.position, { x: (dx / n) * impulse * b.mass * fall, y: (dy / n) * impulse * b.mass * fall - 0.01 * b.mass });
        }
        if (b.plugin.kind === 'block') damageBlock(b, dmg * fall);
        else if (b.plugin.kind === 'enemy') killEnemy(b);
      });
    };
    game.cutLine = function (from, to) {
      game.blockBodies.slice().forEach((b) => {
        if (segDist(b.position, from, to) < 40) destroyBlock(b, true);
      });
      game.enemyBodies.slice().forEach((e) => {
        if (segDist(e.position, from, to) < 44) killEnemy(e);
      });
    };

    // --- damage / destruction ---------------------------------------------
    function damageBlock(body, dmg) {
      if (!body.plugin || body.plugin.kind !== 'block') return;
      if (body.plugin.matDef.rail) return; // indestructible power rail
      body.plugin.hp -= dmg;
      if (body.plugin.hp <= 0) destroyBlock(body);
    }
    function destroyBlock(body, silent) {
      const i = game.blockBodies.indexOf(body);
      if (i < 0) return;
      game.blockBodies.splice(i, 1);
      const mat = body.plugin.matDef;
      if (mat.explosive) {
        game.spawnEffect('boom', body.position, PHYSICS.barrelBlastRadius);
        M.World.remove(world, body);
        game.blastArea(body.position, PHYSICS.barrelBlastRadius, 40, 0.05);
        return;
      }
      if (!silent && mat.brittle) game.spawnEffect('shatter', body.position, 60);
      M.World.remove(world, body);
    }
    function killEnemy(body) {
      const i = game.enemyBodies.indexOf(body);
      if (i < 0) return;
      game.enemyBodies.splice(i, 1);
      game.spawnEffect('pop', body.position, 40);
      M.World.remove(world, body);
    }

    // --- the arc (conductivity) -------------------------------------------
    game.arcFromRail = function (railBody) {
      const nodes = game.blockBodies.map((b) => ({ id: b.plugin.id, conductive: !!b.plugin.matDef.conductive }));
      const boxes = game.blockBodies.map(boxOf);
      const edges = ER.conductivity.buildEdges(boxes, 12);
      const energized = ER.conductivity.energize({ nodes, edges, sources: [railBody.plugin.id] });

      // Kill enemies touching any energized part.
      const energizedBodies = game.blockBodies.filter((b) => energized.has(b.plugin.id));
      game.enemyBodies.slice().forEach((e) => {
        const eb = bodyBox(e);
        for (const bb of energizedBodies) {
          if (overlap(eb, bodyBox(bb), 16)) { killEnemy(e); break; }
        }
      });

      // Visual: arc segments along energized adjacency + a flash.
      const centres = {};
      energizedBodies.forEach((b) => (centres[b.plugin.id] = { x: b.position.x, y: b.position.y }));
      const segs = [];
      edges.forEach(([a, b]) => {
        if (centres[a] && centres[b]) segs.push({ a: centres[a], b: centres[b] });
      });
      game.arcs.push({ segs, ttl: 22, max: 22, source: { x: railBody.position.x, y: railBody.position.y } });
      game.flashLabel('LICHTBOGEN');
      game.emit('arc', { count: energizedBodies.length });
    };

    // --- collisions --------------------------------------------------------
    M.Events.on(engine, 'collisionStart', (ev) => {
      ev.pairs.forEach((pair) => handlePair(pair.bodyA, pair.bodyB));
    });

    function handlePair(a, b) {
      const rv = relSpeed(a, b);
      handleSide(a, b, rv);
      handleSide(b, a, rv);
    }
    function handleSide(body, other, rv) {
      const p = body.plugin || {};
      const op = other.plugin || {};

      // Lift projectile detonates on first solid contact.
      if (op.kind === 'projectile' && op.lift && (p.kind === 'block' || p.kind === 'ground' || p.kind === 'wall')) {
        game.spawnEffect('boom', other.position, 170);
        game.blastArea(other.position, 170, 30, 0.05);
        game.removeActive(other);
        return;
      }

      if (p.kind === 'block') {
        // Rail hit by a shot or pellet -> arc.
        if (p.matDef.rail && (op.kind === 'projectile' || op.kind === 'pellet')) {
          game.arcFromRail(body);
          return;
        }
        if (p.matDef.rail) return;

        let dmg = rv * PHYSICS.dmgFactor * (op.punch || 1);
        if (p.matDef.brittle) dmg *= 3;
        if (op.kind === 'pellet') dmg = p.matDef.light ? rv * 1.4 : rv * 0.15;
        if (p.softened) dmg *= 1.8;
        if (dmg > 0.4) damageBlock(body, dmg);
      } else if (p.kind === 'enemy') {
        const threshold = p.type === 'kalki' ? PHYSICS.kalkiKillImpact : PHYSICS.enemyKillImpact;
        const boost = op.kind === 'projectile' ? (op.punch || 1) : 1;
        if (op.kind === 'pellet' && p.type !== 'stauber') return; // grit only clears dust
        if (rv * boost >= threshold) killEnemy(body);
      }
    }

    // --- main step ---------------------------------------------------------
    game.update = function () {
      M.Engine.update(engine, 1000 / 60);

      // Bubbles lift.
      game.activeBodies.forEach((b) => {
        if (b.plugin.lift) M.Body.setVelocity(b, { x: b.velocity.x * 0.985, y: -9 });
      });

      // Pellet lifetimes.
      game.pellets = game.pellets.filter((p) => {
        p.plugin.ttl--;
        if (p.plugin.ttl <= 0 || p.position.y > PHYSICS.offWorldY) { M.World.remove(world, p); return false; }
        return true;
      });

      // Off-world cleanup.
      game.blockBodies.slice().forEach((b) => { if (b.position.y > PHYSICS.offWorldY) destroyBlock(b, true); });
      game.enemyBodies.slice().forEach((e) => { if (e.position.y > PHYSICS.offWorldY) killEnemy(e); });
      game.activeBodies.slice().forEach((b) => { if (b.position.y > PHYSICS.offWorldY || b.position.x > WORLD.width + 400 || b.position.x < -400) game.removeActive(b); });

      // Effects / arcs / flash countdown.
      game.effects = game.effects.filter((e) => (--e.ttl) > 0);
      game.arcs = game.arcs.filter((a) => (--a.ttl) > 0);
      if (game.flashTtl > 0) game.flashTtl--;

      // Win as soon as the last enemy is gone.
      if (game.enemyBodies.length === 0 && game.state !== 'won' && game.state !== 'lost') {
        return win();
      }

      // Resolve the end of a shot.
      if (game.state === 'flying') {
        const active = game.activeBodies;
        const quiet = active.length === 0 || active.every((b) => b.speed < PHYSICS.settleSpeed);
        if (quiet) {
          if (++game.settleCounter > (active.length === 0 ? 6 : PHYSICS.settleFrames)) endShot();
        } else {
          game.settleCounter = 0;
        }
      }
    };

    function endShot() {
      game.settleCounter = 0;
      if (game.enemyBodies.length === 0) return win();
      if (game.queue.length === 0 && game.currentType === null) return lose();
      prepareShot();
    }

    function grade(leftover) {
      if (leftover >= 2) return SCORING.perfect;
      if (leftover === 1) return SCORING.good;
      return SCORING.pass;
    }

    function win() {
      if (game.state === 'won') return;
      const leftover = Math.max(0, game.shotsTotal - game.shotsUsed);
      game.um = grade(leftover);
      game.state = 'won';
      game.emit('levelend', { won: true, um: game.um, leftover, level: game.level });
    }
    function lose() {
      if (game.state === 'lost') return;
      game.state = 'lost';
      game.um = 0;
      game.emit('levelend', { won: false, um: 0, level: game.level });
    }

    // --- read-only helpers for the renderer / aiming ----------------------
    game.predictPath = function () {
      if (game.state !== 'aim' || !game.aiming) return [];
      const dx = game.anchor.x - game.aimPos.x;
      const dy = game.anchor.y - game.aimPos.y;
      let vx = dx * PHYSICS.launchPower, vy = dy * PHYSICS.launchPower;
      let px = game.aimPos.x, py = game.aimPos.y;
      const pts = [];
      for (let i = 0; i < 48; i++) {
        pts.push({ x: px, y: py });
        px += vx; py += vy; vy += PHYSICS.previewGravity;
        if (py > WORLD.groundY) break;
      }
      return pts;
    };

    return game;
  }

  ER.game = { create };
})(typeof window !== 'undefined' ? window : globalThis);
