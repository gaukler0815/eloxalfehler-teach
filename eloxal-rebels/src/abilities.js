/*
 * abilities.js — the one special move each rebel has, triggered by a tap/click
 * in flight (Lasar once per level). Each function receives the game and the
 * currently active projectile body and manipulates the Matter world through the
 * small helper API the game exposes. No rendering here.
 */
(function (global) {
  'use strict';
  const ER = (global.ER = global.ER || {});
  const Matter = global.Matter;

  function unit(v) {
    const m = Math.hypot(v.x, v.y) || 1;
    return { x: v.x / m, y: v.y / m };
  }

  const abilities = {
    // Ali has no ability.
    ali: null,

    // Bolle: aim down and dive, punching through one layer instead of bouncing.
    dive(game, body) {
      const v = body.velocity;
      const dir = unit({ x: v.x, y: Math.abs(v.y) + 6 });
      const speed = Math.max(Math.hypot(v.x, v.y), 16);
      Matter.Body.setVelocity(body, { x: dir.x * speed * 0.9, y: speed * 1.15 });
      body.plugin.punch = 3.2; // triples damage, resists breaking apart
      game.flashLabel('BOLLE — STURZFLUG');
    },

    // Rippi: split into three cooling fins that fan out.
    split(game, body) {
      const v = body.velocity;
      const base = Math.atan2(v.y, v.x);
      const speed = Math.max(Math.hypot(v.x, v.y), 10);
      const pos = { x: body.position.x, y: body.position.y };
      game.removeActive(body);
      [-0.3, 0, 0.3].forEach((da) => {
        const a = base + da;
        const b = game.spawnProjectile('rippi', pos.x, pos.y,
          Math.cos(a) * speed, Math.sin(a) * speed, 0.62);
        b.plugin.noAbility = true;
      });
      game.flashLabel('RIPPI — STREUUNG');
    },

    // Titania: yank the nearest edge in front downward to topple towers.
    hook(game, body) {
      const target = game.nearestBlockInFront(body, 520);
      if (target) {
        const m = target.mass;
        Matter.Body.applyForce(target, { x: target.position.x, y: target.bounds.min.y },
          { x: -0.9 * m, y: 1.4 * m });
      }
      game.flashLabel('TITANIA — ZUG');
    },

    // Bubbles: reverse gravity and rise; detonate on first ceiling contact.
    lift(game, body) {
      body.plugin.lift = true;
      body.frictionAir = 0.001;
      game.flashLabel('BUBBLES — AUFTRIEB');
    },

    // Säuri: burst and etch everything in the radius — parts lose strength.
    acid(game, body) {
      game.softenArea(body.position, ER.config.PHYSICS.acidRadius);
      game.blastArea(body.position, ER.config.PHYSICS.acidRadius * 0.6, 4, 0.02);
      game.spawnEffect('acid', body.position, ER.config.PHYSICS.acidRadius);
      game.removeActive(body);
      game.flashLabel('SÄURI — FLÄCHE');
    },

    // Bürsti: blow a cone of blasting grit forward. Clears light material fast.
    blast(game, body) {
      const dir = unit(body.velocity.x || body.velocity.y ? body.velocity : { x: 1, y: -0.2 });
      const speed = Math.max(Math.hypot(body.velocity.x, body.velocity.y), 14);
      for (let i = 0; i < 9; i++) {
        const spread = (i - 4) * 0.09;
        const cos = Math.cos(spread), sin = Math.sin(spread);
        const dx = dir.x * cos - dir.y * sin;
        const dy = dir.x * sin + dir.y * cos;
        const p = game.spawnPellet(body.position.x + dx * 30, body.position.y + dy * 30,
          dx * speed * 1.25, dy * speed * 1.25);
        void p;
      }
      game.flashLabel('BÜRSTI — SCHROT');
    },

    // Lasar: freeze and cut a line through everything it crosses. Once per level.
    cut(game, body) {
      if (game.lasarUsed) return;
      game.lasarUsed = true;
      const dir = unit(body.velocity.x || body.velocity.y ? body.velocity : { x: 1, y: 0 });
      const from = { x: body.position.x, y: body.position.y };
      const to = { x: from.x + dir.x * 900, y: from.y + dir.y * 900 };
      game.cutLine(from, to);
      game.spawnEffect('cut', from, 0, to);
      game.flashLabel('LASAR — SCHNITT');
    }
  };

  // Dispatch by ability key from config.PROJECTILES[type].ability.
  function trigger(game, body) {
    if (body.plugin && body.plugin.noAbility) return;
    const type = body.plugin && body.plugin.type;
    const def = ER.config.PROJECTILES[type];
    if (!def || !def.ability) return;
    const fn = abilities[def.ability];
    if (fn) fn(game, body);
  }

  ER.abilities = { trigger, table: abilities };
})(typeof window !== 'undefined' ? window : globalThis);
