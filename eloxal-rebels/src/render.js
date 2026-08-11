/*
 * render.js — Canvas 2D renderer. Draws the fixed 1920x1080 world, scaled and
 * letterboxed to the window. Figures are simple shapes in the Eloxal palette
 * for now; the finished SVG characters from the design bible drop in later.
 * All game rules live in game.js; this file only draws state.
 */
(function (global) {
  'use strict';
  const ER = (global.ER = global.ER || {});

  function create(canvas, game) {
    const ctx = canvas.getContext('2d');
    const { WORLD, PALETTE, PROJECTILES, ENEMIES } = ER.config;
    const view = { scale: 1, ox: 0, oy: 0, wCss: 0, hCss: 0, dpr: 1 };

    function resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const wCss = canvas.clientWidth || window.innerWidth;
      const hCss = canvas.clientHeight || window.innerHeight;
      canvas.width = Math.round(wCss * dpr);
      canvas.height = Math.round(hCss * dpr);
      const scale = Math.min(wCss / WORLD.width, hCss / WORLD.height);
      view.scale = scale;
      view.ox = (wCss - WORLD.width * scale) / 2;
      view.oy = (hCss - WORLD.height * scale) / 2;
      view.wCss = wCss; view.hCss = hCss; view.dpr = dpr;
    }

    function toWorld(cssX, cssY) {
      return { x: (cssX - view.ox) / view.scale, y: (cssY - view.oy) / view.scale };
    }

    // --- primitives --------------------------------------------------------
    function poly(vertices, fill, stroke, lw) {
      ctx.beginPath();
      ctx.moveTo(vertices[0].x, vertices[0].y);
      for (let i = 1; i < vertices.length; i++) ctx.lineTo(vertices[i].x, vertices[i].y);
      ctx.closePath();
      if (fill) { ctx.fillStyle = fill; ctx.fill(); }
      if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw || 5; ctx.lineJoin = 'round'; ctx.stroke(); }
    }
    function circle(x, y, r, fill, stroke, lw) {
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
      if (fill) { ctx.fillStyle = fill; ctx.fill(); }
      if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw || 5; ctx.stroke(); }
    }
    function shade(hex, amt) {
      const n = parseInt(hex.slice(1), 16);
      let r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255;
      r = Math.max(0, Math.min(255, r + amt));
      g = Math.max(0, Math.min(255, g + amt));
      b = Math.max(0, Math.min(255, b + amt));
      return 'rgb(' + r + ',' + g + ',' + b + ')';
    }

    // --- scene layers ------------------------------------------------------
    function background() {
      const g = ctx.createLinearGradient(0, 0, 0, WORLD.height);
      g.addColorStop(0, PALETTE.hallMid);
      g.addColorStop(0.6, PALETTE.hallDark);
      g.addColorStop(1, '#140F1C');
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, WORLD.width, WORLD.height);

      // Parallax: distant tanks (bath row).
      ctx.fillStyle = 'rgba(30,116,220,0.10)';
      for (let i = 0; i < 7; i++) ctx.fillRect(120 + i * 260, 470, 200, 430);
      ctx.strokeStyle = 'rgba(143,211,255,0.10)'; ctx.lineWidth = 3;
      for (let i = 0; i < 7; i++) ctx.strokeRect(120 + i * 260, 470, 200, 430);

      // Bath glow near the floor.
      const bg = ctx.createLinearGradient(0, WORLD.groundY - 60, 0, WORLD.groundY);
      bg.addColorStop(0, 'rgba(143,211,255,0)');
      bg.addColorStop(1, 'rgba(143,211,255,0.12)');
      ctx.fillStyle = bg;
      ctx.fillRect(0, WORLD.groundY - 60, WORLD.width, 60);
    }

    function ground() {
      ctx.fillStyle = '#0E0A16';
      ctx.fillRect(0, WORLD.groundY, WORLD.width, WORLD.height - WORLD.groundY);
      ctx.strokeStyle = PALETTE.line; ctx.lineWidth = 6;
      ctx.beginPath(); ctx.moveTo(0, WORLD.groundY); ctx.lineTo(WORLD.width, WORLD.groundY); ctx.stroke();
    }

    function drawBlock(body) {
      const p = body.plugin;
      const mat = p.matDef;
      let fill = mat.color;
      if (p.softened) fill = shade(fill, -30);
      poly(body.vertices, fill, PALETTE.line, 5);
      // simple two-step volume: lighter top strip
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(body.vertices[0].x, body.vertices[0].y);
      for (let i = 1; i < body.vertices.length; i++) ctx.lineTo(body.vertices[i].x, body.vertices[i].y);
      ctx.closePath(); ctx.clip();
      ctx.fillStyle = 'rgba(255,255,255,0.12)';
      ctx.fillRect(body.bounds.min.x, body.bounds.min.y, body.bounds.max.x - body.bounds.min.x, (body.bounds.max.y - body.bounds.min.y) * 0.4);
      // damage cracks: darken as hp drops
      if (!mat.rail && p.hp < p.maxHp) {
        ctx.fillStyle = 'rgba(23,19,31,' + (0.5 * (1 - p.hp / p.maxHp)).toFixed(2) + ')';
        ctx.fillRect(body.bounds.min.x, body.bounds.min.y, body.bounds.max.x - body.bounds.min.x, body.bounds.max.y - body.bounds.min.y);
      }
      ctx.restore();
      // rail hazard stripes
      if (mat.rail) {
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(body.vertices[0].x, body.vertices[0].y);
        for (let i = 1; i < body.vertices.length; i++) ctx.lineTo(body.vertices[i].x, body.vertices[i].y);
        ctx.closePath(); ctx.clip();
        ctx.strokeStyle = PALETTE.line; ctx.lineWidth = 8;
        for (let x = body.bounds.min.x - 40; x < body.bounds.max.x; x += 26) {
          ctx.beginPath(); ctx.moveTo(x, body.bounds.max.y); ctx.lineTo(x + 40, body.bounds.min.y); ctx.stroke();
        }
        ctx.restore();
      }
    }

    function drawEnemy(body) {
      const def = ENEMIES[body.plugin.type] || ENEMIES.stauber;
      const x = body.position.x, y = body.position.y, r = def.radius;
      circle(x, y, r, def.color, PALETTE.line, 5);
      circle(x, y + r * 0.15, r * 0.7, shade(def.color, -18), null);
      // eyes
      const ex = r * 0.34, ey = -r * 0.12;
      circle(x - ex, y + ey, r * 0.22, '#fff', PALETTE.line, 3);
      circle(x + ex, y + ey, r * 0.22, '#fff', PALETTE.line, 3);
      circle(x - ex + 2, y + ey, r * 0.1, PALETTE.line, null);
      circle(x + ex + 2, y + ey, r * 0.1, PALETTE.line, null);
      // angry brow
      ctx.strokeStyle = PALETTE.line; ctx.lineWidth = 4;
      ctx.beginPath(); ctx.moveTo(x - ex * 1.5, y - r * 0.4); ctx.lineTo(x - ex * 0.3, y - r * 0.2);
      ctx.moveTo(x + ex * 1.5, y - r * 0.4); ctx.lineTo(x + ex * 0.3, y - r * 0.2); ctx.stroke();
    }

    function drawProjectileBody(body) {
      const type = body.plugin.type;
      const def = PROJECTILES[type] || PROJECTILES.ali;
      const r = body.circleRadius || def.radius;
      circle(body.position.x, body.position.y, r, def.color, PALETTE.line, 5);
      circle(body.position.x - r * 0.3, body.position.y - r * 0.3, r * 0.25, 'rgba(255,255,255,0.7)', null);
      // contact point marker
      circle(body.position.x + r * 0.55, body.position.y, r * 0.16, PALETTE.line, null);
    }

    function drawSlingshot() {
      const a = game.anchor;
      const held = game.state === 'aim' && game.currentType;
      const pos = held ? game.aimPos : a;
      // back band
      ctx.strokeStyle = '#5A4A2A'; ctx.lineWidth = 12; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(a.x + 26, a.y - 70); ctx.lineTo(pos.x, pos.y); ctx.stroke();
      // held projectile
      if (held) {
        const def = PROJECTILES[game.currentType];
        circle(pos.x, pos.y, def.radius, def.color, PALETTE.line, 5);
        circle(pos.x - def.radius * 0.3, pos.y - def.radius * 0.3, def.radius * 0.25, 'rgba(255,255,255,0.7)', null);
      }
      // fork
      ctx.strokeStyle = PALETTE.titan; ctx.lineWidth = 16; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(a.x - 24, a.y + 120); ctx.lineTo(a.x - 24, a.y - 70);
      ctx.moveTo(a.x + 26, a.y + 120); ctx.lineTo(a.x + 26, a.y - 70); ctx.stroke();
      ctx.strokeStyle = PALETTE.line; ctx.lineWidth = 4;
      ctx.beginPath(); ctx.moveTo(a.x - 24, a.y + 120); ctx.lineTo(a.x - 24, a.y - 70);
      ctx.moveTo(a.x + 26, a.y + 120); ctx.lineTo(a.x + 26, a.y - 70); ctx.stroke();
      // front band
      if (held) {
        ctx.strokeStyle = '#7A6636'; ctx.lineWidth = 12;
        ctx.beginPath(); ctx.moveTo(a.x - 24, a.y - 70); ctx.lineTo(pos.x, pos.y); ctx.stroke();
      }
    }

    function drawTrajectory() {
      const pts = game.predictPath();
      if (!pts.length) return;
      ctx.fillStyle = 'rgba(255,255,255,0.6)';
      for (let i = 2; i < pts.length; i += 2) {
        ctx.beginPath(); ctx.arc(pts[i].x, pts[i].y, 5, 0, Math.PI * 2); ctx.fill();
      }
    }

    function drawEffects() {
      game.effects.forEach((e) => {
        const t = e.ttl / e.max;
        if (e.kind === 'boom') {
          circle(e.x, e.y, e.radius * (1 - t) * 0.9 + 20, null, 'rgba(245,168,28,' + t.toFixed(2) + ')', 10);
        } else if (e.kind === 'acid') {
          circle(e.x, e.y, e.radius * (1 - t), 'rgba(126,217,87,' + (0.35 * t).toFixed(2) + ')', 'rgba(126,217,87,' + t.toFixed(2) + ')', 6);
        } else if (e.kind === 'pop') {
          circle(e.x, e.y, 30 * (1 - t) + 8, null, 'rgba(227,58,44,' + t.toFixed(2) + ')', 8);
        } else if (e.kind === 'shatter') {
          ctx.strokeStyle = 'rgba(245,168,28,' + t.toFixed(2) + ')'; ctx.lineWidth = 5;
          for (let k = 0; k < 6; k++) {
            const a = (k / 6) * Math.PI * 2, d = (1 - t) * 60;
            ctx.beginPath(); ctx.moveTo(e.x, e.y); ctx.lineTo(e.x + Math.cos(a) * d, e.y + Math.sin(a) * d); ctx.stroke();
          }
        } else if (e.kind === 'cut' && e.to) {
          ctx.strokeStyle = 'rgba(227,58,44,' + t.toFixed(2) + ')'; ctx.lineWidth = 10; ctx.lineCap = 'round';
          ctx.beginPath(); ctx.moveTo(e.x, e.y); ctx.lineTo(e.to.x, e.to.y); ctx.stroke();
        }
      });
    }

    function drawArcs() {
      game.arcs.forEach((arc) => {
        const t = arc.ttl / arc.max;
        ctx.strokeStyle = 'rgba(143,211,255,' + t.toFixed(2) + ')';
        ctx.shadowColor = PALETTE.arc; ctx.shadowBlur = 24; ctx.lineWidth = 7; ctx.lineCap = 'round';
        arc.segs.forEach((s) => {
          ctx.beginPath();
          ctx.moveTo(s.a.x, s.a.y);
          const mx = (s.a.x + s.b.x) / 2 + (Math.random() - 0.5) * 22;
          const my = (s.a.y + s.b.y) / 2 + (Math.random() - 0.5) * 22;
          ctx.quadraticCurveTo(mx, my, s.b.x, s.b.y);
          ctx.stroke();
        });
        ctx.shadowBlur = 0;
      });
    }

    function drawFlash() {
      if (game.flashTtl <= 0 || !game.flashText) return;
      ctx.save();
      ctx.globalAlpha = Math.min(1, game.flashTtl / 30);
      ctx.font = '700 64px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.lineWidth = 8; ctx.strokeStyle = PALETTE.line;
      ctx.strokeText(game.flashText, WORLD.width / 2, 160);
      ctx.fillStyle = PALETTE.arc;
      ctx.fillText(game.flashText, WORLD.width / 2, 160);
      ctx.restore();
    }

    function draw() {
      ctx.setTransform(view.dpr, 0, 0, view.dpr, 0, 0);
      ctx.fillStyle = '#0B0812';
      ctx.fillRect(0, 0, view.wCss, view.hCss);
      ctx.translate(view.ox, view.oy);
      ctx.scale(view.scale, view.scale);

      background();
      ground();
      game.blockBodies.forEach(drawBlock);
      game.enemyBodies.forEach(drawEnemy);
      game.pellets.forEach((p) => circle(p.position.x, p.position.y, 7, PALETTE.gold, PALETTE.line, 2));
      game.activeBodies.forEach(drawProjectileBody);
      drawSlingshot();
      drawTrajectory();
      drawEffects();
      drawArcs();
      drawFlash();
    }

    resize();
    window.addEventListener('resize', resize);
    return { draw, resize, toWorld, view };
  }

  ER.render = { create };
})(typeof window !== 'undefined' ? window : globalThis);
