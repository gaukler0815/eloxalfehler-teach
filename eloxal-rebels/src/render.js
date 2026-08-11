/*
 * render.js — Canvas 2D renderer for the fixed 1920x1080 world, scaled and
 * letterboxed to the window. This revision adds the "juice": a soft-follow
 * camera with zoom and impact shake, a three-layer parallax hall with living
 * anodizing baths, projectile trails, per-material block detail with damage
 * cracks, and the character art from characters.js. All game rules stay in
 * game.js; this file only draws state (plus purely visual camera bookkeeping).
 */
(function (global) {
  'use strict';
  const ER = (global.ER = global.ER || {});

  function create(canvas, game) {
    const ctx = canvas.getContext('2d');
    const { WORLD, PALETTE, PROJECTILES, ENEMIES } = ER.config;
    const view = { scale: 1, ox: 0, oy: 0, wCss: 0, hCss: 0, dpr: 1 };

    // --- camera (visual only) ---------------------------------------------
    const ZOOM = 1.14; // slightly tighter than "fit" so the camera can travel
    const cam = { x: WORLD.width / 2, y: WORLD.height / 2, shake: 0, flash: 0 };
    const maxPanX = WORLD.width * (1 - 1 / ZOOM) / 2;
    const maxPanY = WORLD.height * (1 - 1 / ZOOM) / 2;

    game.on('impact', (e) => { cam.shake = Math.min(14, Math.max(cam.shake, e.rv * 0.55)); });
    game.on('boom', () => { cam.shake = 16; });
    game.on('arc', () => { cam.shake = 12; cam.flash = 10; });

    function camTarget() {
      const active = game.activeBodies[0];
      if (active) return { x: active.position.x, y: active.position.y };
      if (game.state === 'aim') return { x: game.anchor.x + 330, y: game.anchor.y - 60 };
      return { x: WORLD.width / 2, y: WORLD.height / 2 };
    }
    function updateCamera() {
      const t = camTarget();
      const tx = Math.max(WORLD.width / 2 - maxPanX, Math.min(WORLD.width / 2 + maxPanX, t.x));
      const ty = Math.max(WORLD.height / 2 - maxPanY, Math.min(WORLD.height / 2 + maxPanY, t.y));
      cam.x += (tx - cam.x) * 0.07;
      cam.y += (ty - cam.y) * 0.07;
      cam.shake *= 0.86;
      if (cam.flash > 0) cam.flash--;
    }

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

    // Pointer position -> world coordinates (camera-aware).
    function toWorld(cssX, cssY) {
      const bx = (cssX - view.ox) / view.scale;
      const by = (cssY - view.oy) / view.scale;
      return {
        x: (bx - WORLD.width / 2) / ZOOM + cam.x,
        y: (by - WORLD.height / 2) / ZOOM + cam.y
      };
    }

    // --- primitives ---------------------------------------------------------
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
    function hashOf(str) {
      let h = 0;
      for (let i = 0; i < str.length; i++) h = ((h << 5) - h + str.charCodeAt(i)) | 0;
      return Math.abs(h);
    }
    function clipBody(body) {
      ctx.beginPath();
      ctx.moveTo(body.vertices[0].x, body.vertices[0].y);
      for (let i = 1; i < body.vertices.length; i++) ctx.lineTo(body.vertices[i].x, body.vertices[i].y);
      ctx.closePath(); ctx.clip();
    }

    // --- background: three parallax layers of the anodizing hall ------------
    function parallax(depth) {
      // parallax offset relative to camera pan; depth 0 = infinitely far
      return {
        x: (WORLD.width / 2 - cam.x) * (1 - depth),
        y: (WORLD.height / 2 - cam.y) * (1 - depth) * 0.5
      };
    }

    function background(t) {
      const g = ctx.createLinearGradient(0, 0, 0, WORLD.height);
      g.addColorStop(0, '#332A47');
      g.addColorStop(0.55, PALETTE.hallDark);
      g.addColorStop(1, '#140F1C');
      ctx.fillStyle = g;
      ctx.fillRect(-200, -200, WORLD.width + 400, WORLD.height + 400);

      // Layer 1 (far): roof trusses, crane rail, hanging chains
      const p1 = parallax(0.25);
      ctx.save(); ctx.translate(p1.x, p1.y);
      ctx.strokeStyle = 'rgba(143,211,255,0.07)'; ctx.lineWidth = 10;
      for (let x = -100; x < WORLD.width + 200; x += 320) {
        ctx.beginPath(); ctx.moveTo(x, 40); ctx.lineTo(x + 160, 150); ctx.lineTo(x + 320, 40); ctx.stroke();
      }
      ctx.strokeStyle = 'rgba(143,211,255,0.09)'; ctx.lineWidth = 14;
      ctx.beginPath(); ctx.moveTo(-100, 160); ctx.lineTo(WORLD.width + 200, 160); ctx.stroke();
      // crane trolley + hook
      const cx = 700 + Math.sin(t * 0.0012) * 500;
      ctx.fillStyle = 'rgba(143,211,255,0.10)';
      ctx.fillRect(cx - 60, 150, 120, 40);
      ctx.strokeStyle = 'rgba(143,211,255,0.10)'; ctx.lineWidth = 5;
      ctx.beginPath(); ctx.moveTo(cx, 190); ctx.lineTo(cx, 320); ctx.stroke();
      ctx.beginPath(); ctx.arc(cx, 335, 16, -0.5, Math.PI + 0.5); ctx.stroke();
      ctx.restore();

      // Layer 2 (mid): the bath row with rims, liquid and rising bubbles
      const p2 = parallax(0.55);
      ctx.save(); ctx.translate(p2.x, p2.y);
      for (let i = 0; i < 8; i++) {
        const bx = 60 + i * 260, by = 480, bw = 210, bh = 420;
        ctx.fillStyle = 'rgba(20,26,44,0.85)';
        ctx.fillRect(bx, by, bw, bh);
        // liquid with a slow shimmer
        const lg = ctx.createLinearGradient(0, by + 40, 0, by + bh);
        const tone = i % 3 === 0 ? '30,116,220' : (i % 3 === 1 ? '126,182,245' : '76,140,200');
        lg.addColorStop(0, 'rgba(' + tone + ',0.34)');
        lg.addColorStop(1, 'rgba(' + tone + ',0.10)');
        ctx.fillStyle = lg;
        ctx.fillRect(bx + 8, by + 42 + Math.sin(t * 0.02 + i) * 3, bw - 16, bh - 50);
        // rim
        ctx.fillStyle = 'rgba(180,169,150,0.25)';
        ctx.fillRect(bx - 6, by, bw + 12, 14);
        // bubbles
        ctx.fillStyle = 'rgba(200,230,255,0.35)';
        for (let b = 0; b < 4; b++) {
          const ph = ((t * (0.8 + b * 0.23) + i * 97 + b * 53) % 340);
          const bxx = bx + 30 + ((i * 37 + b * 61) % (bw - 60));
          ctx.beginPath(); ctx.arc(bxx, by + bh - 20 - ph, 3 + (b % 3), 0, Math.PI * 2); ctx.fill();
        }
        // electrode bar over some baths
        if (i % 2 === 0) {
          ctx.fillStyle = 'rgba(245,168,28,0.30)';
          ctx.fillRect(bx + 20, by - 26, bw - 40, 12);
        }
      }
      ctx.restore();

      // occasional steam wisp rising from the baths (ambient particles)
      if (ER.particles && t % 24 === 0) {
        ER.particles.steam(200 + Math.random() * 1500, 520);
      }

      // Layer 3 glow near the floor
      const bg2 = ctx.createLinearGradient(0, WORLD.groundY - 70, 0, WORLD.groundY);
      bg2.addColorStop(0, 'rgba(143,211,255,0)');
      bg2.addColorStop(1, 'rgba(143,211,255,0.14)');
      ctx.fillStyle = bg2;
      ctx.fillRect(-200, WORLD.groundY - 70, WORLD.width + 400, 70);
    }

    function ground() {
      ctx.fillStyle = '#0E0A16';
      ctx.fillRect(-200, WORLD.groundY, WORLD.width + 400, WORLD.height - WORLD.groundY + 200);
      // grating lines
      ctx.strokeStyle = 'rgba(143,211,255,0.06)'; ctx.lineWidth = 3;
      for (let x = -100; x < WORLD.width + 200; x += 46) {
        ctx.beginPath(); ctx.moveTo(x, WORLD.groundY + 8); ctx.lineTo(x - 18, WORLD.height + 100); ctx.stroke();
      }
      ctx.strokeStyle = PALETTE.line; ctx.lineWidth = 6;
      ctx.beginPath(); ctx.moveTo(-200, WORLD.groundY); ctx.lineTo(WORLD.width + 400, WORLD.groundY); ctx.stroke();
    }

    // --- blocks with per-material detail ------------------------------------
    function drawBlock(body, t) {
      const p = body.plugin;
      const mat = p.matDef;
      let fill = mat.color;
      if (p.softened) fill = shade(fill, -30);
      poly(body.vertices, fill, PALETTE.line, 5);

      const bMinX = body.bounds.min.x, bMinY = body.bounds.min.y;
      const bw = body.bounds.max.x - bMinX, bh = body.bounds.max.y - bMinY;

      ctx.save();
      clipBody(body);

      // top light + bath reflection from below (the two light sources)
      ctx.fillStyle = 'rgba(255,255,255,0.14)';
      ctx.fillRect(bMinX, bMinY, bw, bh * 0.32);
      ctx.fillStyle = 'rgba(143,211,255,0.08)';
      ctx.fillRect(bMinX, bMinY + bh * 0.8, bw, bh * 0.2);

      // material signature
      ctx.translate(body.position.x, body.position.y);
      ctx.rotate(body.angle);
      const hw = (p.w || bw) / 2, hh = (p.h || bh) / 2;
      if (p.material === 'kartonage') {
        ctx.strokeStyle = 'rgba(122,96,58,0.5)'; ctx.lineWidth = 3;
        for (let y = -hh + 12; y < hh; y += 16) {
          ctx.beginPath();
          for (let x = -hw; x <= hw; x += 8) {
            const yy = y + Math.sin(x * 0.6) * 2;
            if (x === -hw) ctx.moveTo(x, yy); else ctx.lineTo(x, yy);
          }
          ctx.stroke();
        }
        ctx.fillStyle = 'rgba(180,150,90,0.6)';
        ctx.fillRect(-hw * 0.35, -hh, hw * 0.7, 10);
      } else if (p.material === 'kunststoff') {
        ctx.strokeStyle = 'rgba(255,255,255,0.35)'; ctx.lineWidth = 6; ctx.lineCap = 'round';
        ctx.beginPath(); ctx.moveTo(-hw * 0.6, -hh * 0.55); ctx.lineTo(hw * 0.2, hh * 0.35); ctx.stroke();
      } else if (p.material === 'aluminium') {
        ctx.strokeStyle = 'rgba(255,255,255,0.16)'; ctx.lineWidth = 2;
        for (let x = -hw + 8; x < hw; x += 12) {
          ctx.beginPath(); ctx.moveTo(x, -hh + 5); ctx.lineTo(x, hh - 5); ctx.stroke();
        }
        ctx.fillStyle = shade('#C9D2DC', -50);
        [[-hw + 9, -hh + 9], [hw - 9, -hh + 9], [-hw + 9, hh - 9], [hw - 9, hh - 9]].forEach(([x, y]) => {
          ctx.beginPath(); ctx.arc(x, y, 3.5, 0, Math.PI * 2); ctx.fill();
        });
      } else if (p.material === 'stahl') {
        ctx.strokeStyle = 'rgba(23,19,31,0.5)'; ctx.lineWidth = 5;
        ctx.beginPath(); ctx.moveTo(-hw, -hh); ctx.lineTo(hw, hh);
        ctx.moveTo(hw, -hh); ctx.lineTo(-hw, hh); ctx.stroke();
        ctx.fillStyle = shade('#8A94A0', -46);
        [[-hw + 10, -hh + 10], [hw - 10, -hh + 10], [-hw + 10, hh - 10], [hw - 10, hh - 10]].forEach(([x, y]) => {
          ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.fill();
        });
      } else if (p.material === 'fehlcharge') {
        ctx.fillStyle = 'rgba(255,255,255,0.32)';
        ctx.beginPath();
        ctx.moveTo(-hw * 0.7, -hh); ctx.lineTo(-hw * 0.3, -hh); ctx.lineTo(hw * 0.3, hh); ctx.lineTo(-hw * 0.1, hh);
        ctx.closePath(); ctx.fill();
      } else if (p.material === 'saeurefass') {
        ctx.strokeStyle = 'rgba(23,19,31,0.55)'; ctx.lineWidth = 4;
        [-hh * 0.4, hh * 0.25].forEach((y) => {
          ctx.beginPath(); ctx.moveTo(-hw, y); ctx.lineTo(hw, y); ctx.stroke();
        });
        // acid drop symbol
        ctx.fillStyle = '#17131F';
        ctx.beginPath();
        ctx.moveTo(0, -hh * 0.28);
        ctx.bezierCurveTo(hw * 0.3, 0, hw * 0.24, hh * 0.28, 0, hh * 0.3);
        ctx.bezierCurveTo(-hw * 0.24, hh * 0.28, -hw * 0.3, 0, 0, -hh * 0.28);
        ctx.fill();
      }
      ctx.restore();

      // damage cracks (deterministic per block, more as hp drops)
      if (!mat.rail && p.hp < p.maxHp) {
        const dmg = 1 - p.hp / p.maxHp;
        const h = hashOf(p.id || 'x');
        ctx.save();
        clipBody(body);
        ctx.strokeStyle = 'rgba(23,19,31,' + (0.35 + dmg * 0.45).toFixed(2) + ')';
        ctx.lineWidth = 3;
        const n = dmg > 0.6 ? 3 : (dmg > 0.25 ? 2 : 1);
        for (let c = 0; c < n; c++) {
          let x = bMinX + ((h >> (c * 3)) % 100) / 100 * bw;
          let y = bMinY;
          ctx.beginPath(); ctx.moveTo(x, y);
          for (let s = 0; s < 4; s++) {
            x += (((h >> (c * 4 + s)) % 21) - 10) * 2.2;
            y += bh / 4;
            ctx.lineTo(x, y);
          }
          ctx.stroke();
        }
        ctx.restore();
      }

      // power rail: hazard stripes + pulsing glow
      if (mat.rail) {
        ctx.save();
        clipBody(body);
        ctx.strokeStyle = PALETTE.line; ctx.lineWidth = 8;
        for (let x = bMinX - 40; x < bMinX + bw; x += 26) {
          ctx.beginPath(); ctx.moveTo(x, bMinY + bh); ctx.lineTo(x + 40, bMinY); ctx.stroke();
        }
        ctx.restore();
        const pulse = 0.35 + 0.25 * Math.sin(t * 0.1);
        ctx.save();
        ctx.shadowColor = PALETTE.arc; ctx.shadowBlur = 18;
        ctx.strokeStyle = 'rgba(143,211,255,' + pulse.toFixed(2) + ')';
        ctx.lineWidth = 3;
        poly(body.vertices, null, 'rgba(143,211,255,' + pulse.toFixed(2) + ')', 3);
        ctx.restore();
      }
    }

    // --- entities ------------------------------------------------------------
    function drawEnemy(body, t) {
      const def = ENEMIES[body.plugin.type] || ENEMIES.stauber;
      ER.characters.drawEnemy(
        ctx, body.plugin.type, body.position.x, body.position.y, def.radius,
        body.velocity.x, body.velocity.y, t, body.id, body.plugin.squashT || 0
      );
      if (body.plugin.squashT > 0) body.plugin.squashT--;
    }

    function drawProjectileBody(body, t) {
      const type = body.plugin.type;
      const def = PROJECTILES[type] || PROJECTILES.ali;
      const r = body.circleRadius || def.radius;
      // motion trail
      const trail = body.plugin.trail || [];
      for (let i = 0; i < trail.length; i++) {
        const k = i / trail.length;
        ctx.globalAlpha = k * 0.25;
        circle(trail[i].x, trail[i].y, r * (0.3 + k * 0.5), def.color, null);
      }
      ctx.globalAlpha = 1;
      ER.characters.drawProjectile(
        ctx, type, body.position.x, body.position.y, r,
        body.velocity.x, body.velocity.y, t, body.id, body.plugin.squashT || 0
      );
      if (body.plugin.squashT > 0) body.plugin.squashT--;
    }

    function drawSlingshot(t) {
      const a = game.anchor;
      const held = game.state === 'aim' && game.currentType;
      const pos = held ? game.aimPos : a;
      const pull = held ? Math.hypot(pos.x - a.x, pos.y - a.y) : 0;
      // idle wobble after release
      const wob = (!held && game.state !== 'idle') ? Math.sin(t * 0.55) * Math.max(0, 8 - (t % 60) * 0.4) : 0;
      // back band
      ctx.strokeStyle = '#5A4A2A'; ctx.lineWidth = 12 - Math.min(5, pull * 0.015); ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(a.x + 26, a.y - 70); ctx.lineTo(pos.x + wob, pos.y); ctx.stroke();
      // held character leans back while pulled
      if (held) {
        const def = PROJECTILES[game.currentType];
        ER.characters.drawProjectile(ctx, game.currentType, pos.x, pos.y, def.radius,
          (a.x - pos.x) * 0.05, (a.y - pos.y) * 0.05, t, 5, 0);
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
        ctx.strokeStyle = '#7A6636'; ctx.lineWidth = 12 - Math.min(5, pull * 0.015);
        ctx.beginPath(); ctx.moveTo(a.x - 24, a.y - 70); ctx.lineTo(pos.x, pos.y); ctx.stroke();
      }
      // waiting queue sits next to the slingshot, fidgeting
      const queue = game.queue || [];
      for (let i = 0; i < Math.min(queue.length, 4); i++) {
        const def = PROJECTILES[queue[i]] || PROJECTILES.ali;
        const qx = a.x - 120 - i * 64;
        const hop = Math.max(0, Math.sin(t * 0.12 + i * 1.7)) * 6;
        ER.characters.drawProjectile(ctx, queue[i], qx, WORLD.groundY - def.radius * 0.8 - hop,
          def.radius * 0.8, 0.6, 0, t, 11 + i, 0);
      }
    }

    function drawTrajectory(t) {
      const pts = game.predictPath();
      if (!pts.length) return;
      for (let i = 2; i < pts.length; i += 2) {
        const march = (i * 0.5 + t * 0.25) % 4;
        ctx.globalAlpha = 0.25 + 0.45 * Math.max(0, 1 - march);
        ctx.fillStyle = '#fff';
        ctx.beginPath(); ctx.arc(pts[i].x, pts[i].y, 5, 0, Math.PI * 2); ctx.fill();
      }
      ctx.globalAlpha = 1;
    }

    function drawEffects() {
      game.effects.forEach((e) => {
        const k = e.ttl / e.max;
        if (e.kind === 'boom') {
          circle(e.x, e.y, e.radius * (1 - k) * 0.9 + 20, null, 'rgba(245,168,28,' + k.toFixed(2) + ')', 10);
          circle(e.x, e.y, e.radius * (1 - k) * 0.55 + 10, null, 'rgba(255,255,255,' + (k * 0.7).toFixed(2) + ')', 5);
        } else if (e.kind === 'acid') {
          circle(e.x, e.y, e.radius * (1 - k), 'rgba(126,217,87,' + (0.35 * k).toFixed(2) + ')', 'rgba(126,217,87,' + k.toFixed(2) + ')', 6);
        } else if (e.kind === 'pop') {
          circle(e.x, e.y, 30 * (1 - k) + 8, null, 'rgba(227,58,44,' + k.toFixed(2) + ')', 8);
        } else if (e.kind === 'shatter') {
          ctx.strokeStyle = 'rgba(245,168,28,' + k.toFixed(2) + ')'; ctx.lineWidth = 5;
          for (let s = 0; s < 6; s++) {
            const a = (s / 6) * Math.PI * 2, d = (1 - k) * 60;
            ctx.beginPath(); ctx.moveTo(e.x, e.y); ctx.lineTo(e.x + Math.cos(a) * d, e.y + Math.sin(a) * d); ctx.stroke();
          }
        } else if (e.kind === 'cut' && e.to) {
          ctx.strokeStyle = 'rgba(227,58,44,' + k.toFixed(2) + ')'; ctx.lineWidth = 10; ctx.lineCap = 'round';
          ctx.beginPath(); ctx.moveTo(e.x, e.y); ctx.lineTo(e.to.x, e.to.y); ctx.stroke();
        }
      });
    }

    function drawArcs() {
      game.arcs.forEach((arc) => {
        const k = arc.ttl / arc.max;
        ctx.strokeStyle = 'rgba(143,211,255,' + k.toFixed(2) + ')';
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
      const k = Math.min(1, game.flashTtl / 30);
      const popIn = 1 + Math.max(0, (game.flashTtl - 80) / 10) * 0.4;
      ctx.globalAlpha = k;
      ctx.translate(WORLD.width / 2, 160);
      ctx.scale(popIn, popIn);
      ctx.font = '800 64px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.lineWidth = 9; ctx.strokeStyle = PALETTE.line;
      ctx.strokeText(game.flashText, 0, 0);
      ctx.fillStyle = PALETTE.arc;
      ctx.fillText(game.flashText, 0, 0);
      ctx.restore();
    }

    // --- frame ---------------------------------------------------------------
    function draw(t) {
      t = t || 0;
      updateCamera();
      ctx.setTransform(view.dpr, 0, 0, view.dpr, 0, 0);
      ctx.fillStyle = '#0B0812';
      ctx.fillRect(0, 0, view.wCss, view.hCss);
      ctx.translate(view.ox, view.oy);
      ctx.scale(view.scale, view.scale);

      // camera transform (zoom around the camera focus + shake)
      const sx = (Math.random() - 0.5) * cam.shake;
      const sy = (Math.random() - 0.5) * cam.shake;
      ctx.translate(WORLD.width / 2 + sx, WORLD.height / 2 + sy);
      ctx.scale(ZOOM, ZOOM);
      ctx.translate(-cam.x, -cam.y);

      background(t);
      ground();
      game.blockBodies.forEach((b) => drawBlock(b, t));
      game.enemyBodies.forEach((b) => drawEnemy(b, t));
      game.pellets.forEach((p) => circle(p.position.x, p.position.y, 7, PALETTE.gold, PALETTE.line, 2));
      game.activeBodies.forEach((b) => drawProjectileBody(b, t));
      drawSlingshot(t);
      drawTrajectory(t);
      if (ER.particles) ER.particles.draw(ctx);
      drawEffects();
      drawArcs();

      // arc white-flash overlay (drawn in world space, covers view)
      if (cam.flash > 0) {
        ctx.fillStyle = 'rgba(200,235,255,' + (cam.flash / 10 * 0.35).toFixed(2) + ')';
        ctx.fillRect(cam.x - WORLD.width, cam.y - WORLD.height, WORLD.width * 2, WORLD.height * 2);
      }

      drawFlash();
    }

    resize();
    window.addEventListener('resize', resize);
    return { draw, resize, toWorld, view };
  }

  ER.render = { create };
})(typeof window !== 'undefined' ? window : globalThis);
