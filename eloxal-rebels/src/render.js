/*
 * render.js — Canvas 2D renderer with an Angry-Birds-style camera: zoomed out
 * while aiming so the whole (possibly wide) level is visible, zooming in to
 * follow the shot, an intro pan across the fortress when a level loads, and
 * impact shake. Backgrounds span the level's world width in three parallax
 * layers, with the JACOBI ELOXAL hall signage, light shafts and a vignette.
 * All game rules stay in game.js; this file only draws state.
 */
(function (global) {
  'use strict';
  const ER = (global.ER = global.ER || {});

  function create(canvas, game) {
    const ctx = canvas.getContext('2d');
    const { WORLD, PALETTE, PROJECTILES, ENEMIES } = ER.config;
    const view = { scale: 1, ox: 0, oy: 0, wCss: 0, hCss: 0, dpr: 1 };

    // --- camera (visual only) ---------------------------------------------
    const cam = { x: WORLD.width / 2, y: WORLD.height / 2, zoom: 1.0, shake: 0, flash: 0, introT: 0 };

    game.on('impact', (e) => { cam.shake = Math.min(14, Math.max(cam.shake, e.rv * 0.55)); });
    game.on('boom', () => { cam.shake = 16; });
    game.on('arc', () => { cam.shake = 12; cam.flash = 10; });
    game.on('levelloaded', (e) => {
      // start on the fortress, then ease over to the slingshot
      cam.x = e.focusX; cam.y = Math.min(e.focusY, 700); cam.zoom = 1.1; cam.introT = 55;
    });

    function levelW() { return game.levelWidth || WORLD.width; }
    function fitZoom() { return Math.min(1.06, (WORLD.width / levelW()) * 0.985); }

    function camGoal() {
      if (cam.introT > 0) return { x: cam.x, y: cam.y, zoom: 1.08 }; // hold on the fortress
      const active = game.activeBodies[0];
      if (active) return { x: active.position.x, y: active.position.y, zoom: 1.12 };
      if (game.state === 'aim') return { x: levelW() / 2, y: WORLD.height / 2, zoom: fitZoom() };
      return { x: levelW() / 2, y: WORLD.height / 2, zoom: fitZoom() };
    }

    function updateCamera() {
      if (cam.introT > 0) cam.introT--;
      const goal = camGoal();
      // ease zoom first, then derive the pan limits from it
      cam.zoom += (goal.zoom - cam.zoom) * 0.06;
      const halfW = (WORLD.width / 2) / cam.zoom;
      const halfH = (WORLD.height / 2) / cam.zoom;
      let tx = goal.x, ty = goal.y;
      // keep the view inside the level horizontally
      if (levelW() <= halfW * 2) tx = levelW() / 2;
      else tx = Math.max(halfW, Math.min(levelW() - halfW, tx));
      // keep the ground near the bottom of the view
      ty = Math.min(ty, WORLD.groundY + 120 - halfH);
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
        x: (bx - WORLD.width / 2) / cam.zoom + cam.x,
        y: (by - WORLD.height / 2) / cam.zoom + cam.y
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

    // --- the company hall sign (vector, no image files) ----------------------
    function drawLogoSign(x, y, s, alpha) {
      ctx.save();
      ctx.translate(x, y);
      ctx.scale(s, s);
      ctx.globalAlpha = alpha;
      // round JX mark
      const hg = ctx.createLinearGradient(-34, -34, 34, 34);
      hg.addColorStop(0, '#1E74DC'); hg.addColorStop(1, '#8FD3FF');
      ctx.fillStyle = hg;
      ctx.beginPath(); ctx.arc(0, 0, 34, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = PALETTE.line; ctx.lineWidth = 5; ctx.stroke();
      ctx.fillStyle = '#fff';
      ctx.font = '800 30px system-ui, sans-serif';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('JX', 0, 2);
      // wordmark
      ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
      ctx.fillStyle = '#C9D2DC';
      ctx.font = '800 44px system-ui, sans-serif';
      ctx.fillText('JACOBI', 52, -2);
      ctx.fillStyle = '#F5A81C';
      ctx.font = '700 30px system-ui, sans-serif';
      ctx.fillText('E L O X A L', 53, 32);
      ctx.restore();
    }

    // --- background: three parallax layers of the anodizing hall ------------
    function parallax(depth) {
      return {
        x: (levelW() / 2 - cam.x) * (1 - depth),
        y: (WORLD.height / 2 - cam.y) * (1 - depth) * 0.5
      };
    }

    function background(t) {
      const lw = levelW();
      const g = ctx.createLinearGradient(0, -1400, 0, WORLD.height);
      g.addColorStop(0, '#241D33');
      g.addColorStop(0.62, '#332A47');
      g.addColorStop(0.85, PALETTE.hallDark);
      g.addColorStop(1, '#140F1C');
      ctx.fillStyle = g;
      ctx.fillRect(-600, -1400, lw + 1200, WORLD.height + 1800);

      // Layer 1 (far): roof trusses, crane rail, wall signage, light shafts
      const p1 = parallax(0.25);
      ctx.save(); ctx.translate(p1.x, p1.y);
      ctx.strokeStyle = 'rgba(143,211,255,0.07)'; ctx.lineWidth = 10;
      for (let x = -300; x < lw + 300; x += 320) {
        ctx.beginPath(); ctx.moveTo(x, 40); ctx.lineTo(x + 160, 150); ctx.lineTo(x + 320, 40); ctx.stroke();
      }
      ctx.strokeStyle = 'rgba(143,211,255,0.09)'; ctx.lineWidth = 14;
      ctx.beginPath(); ctx.moveTo(-300, 160); ctx.lineTo(lw + 300, 160); ctx.stroke();
      // crane trolley + hook, slowly patrolling the hall
      const cx = lw / 2 + Math.sin(t * 0.0012) * lw * 0.3;
      ctx.fillStyle = 'rgba(143,211,255,0.10)';
      ctx.fillRect(cx - 60, 150, 120, 40);
      ctx.strokeStyle = 'rgba(143,211,255,0.10)'; ctx.lineWidth = 5;
      ctx.beginPath(); ctx.moveTo(cx, 190); ctx.lineTo(cx, 320); ctx.stroke();
      ctx.beginPath(); ctx.arc(cx, 335, 16, -0.5, Math.PI + 0.5); ctx.stroke();
      // hall signage every ~1700 units
      for (let x = 860; x < lw; x += 1700) {
        drawLogoSign(x, 290, 1.15, 0.4);
      }
      // light shafts from the roof windows
      ctx.fillStyle = 'rgba(200,225,255,0.045)';
      for (let x = 300; x < lw + 300; x += 900) {
        ctx.beginPath();
        ctx.moveTo(x, 60); ctx.lineTo(x + 240, 60);
        ctx.lineTo(x + 560, WORLD.groundY); ctx.lineTo(x + 180, WORLD.groundY);
        ctx.closePath(); ctx.fill();
      }
      ctx.restore();

      // Layer 2 (mid): the bath row with rims, liquid and rising bubbles
      const p2 = parallax(0.55);
      ctx.save(); ctx.translate(p2.x, p2.y);
      for (let i = 0, bx = 60; bx < lw + 200; i++, bx += 260) {
        const by = 480, bw = 210, bh = 420;
        ctx.fillStyle = 'rgba(20,26,44,0.85)';
        ctx.fillRect(bx, by, bw, bh);
        const lg = ctx.createLinearGradient(0, by + 40, 0, by + bh);
        const tone = i % 3 === 0 ? '30,116,220' : (i % 3 === 1 ? '126,182,245' : '76,140,200');
        lg.addColorStop(0, 'rgba(' + tone + ',0.34)');
        lg.addColorStop(1, 'rgba(' + tone + ',0.10)');
        ctx.fillStyle = lg;
        ctx.fillRect(bx + 8, by + 42 + Math.sin(t * 0.02 + i) * 3, bw - 16, bh - 50);
        ctx.fillStyle = 'rgba(180,169,150,0.25)';
        ctx.fillRect(bx - 6, by, bw + 12, 14);
        ctx.fillStyle = 'rgba(200,230,255,0.35)';
        for (let b = 0; b < 4; b++) {
          const ph = ((t * (0.8 + b * 0.23) + i * 97 + b * 53) % 340);
          const bxx = bx + 30 + ((i * 37 + b * 61) % (bw - 60));
          ctx.beginPath(); ctx.arc(bxx, by + bh - 20 - ph, 3 + (b % 3), 0, Math.PI * 2); ctx.fill();
        }
        if (i % 2 === 0) {
          ctx.fillStyle = 'rgba(245,168,28,0.30)';
          ctx.fillRect(bx + 20, by - 26, bw - 40, 12);
        }
      }
      ctx.restore();

      // occasional steam wisp rising from the baths (ambient particles)
      if (ER.particles && t % 22 === 0) {
        ER.particles.steam(150 + Math.random() * (lw - 300), 520);
      }

      // Layer 3 glow near the floor
      const bg2 = ctx.createLinearGradient(0, WORLD.groundY - 70, 0, WORLD.groundY);
      bg2.addColorStop(0, 'rgba(143,211,255,0)');
      bg2.addColorStop(1, 'rgba(143,211,255,0.14)');
      ctx.fillStyle = bg2;
      ctx.fillRect(-600, WORLD.groundY - 70, lw + 1200, 70);
    }

    function ground() {
      const lw = levelW();
      ctx.fillStyle = '#0E0A16';
      ctx.fillRect(-600, WORLD.groundY, lw + 1200, WORLD.height - WORLD.groundY + 900);
      ctx.strokeStyle = 'rgba(143,211,255,0.06)'; ctx.lineWidth = 3;
      for (let x = -300; x < lw + 300; x += 46) {
        ctx.beginPath(); ctx.moveTo(x, WORLD.groundY + 8); ctx.lineTo(x - 18, WORLD.height + 400); ctx.stroke();
      }
      ctx.strokeStyle = PALETTE.line; ctx.lineWidth = 6;
      ctx.beginPath(); ctx.moveTo(-600, WORLD.groundY); ctx.lineTo(lw + 600, WORLD.groundY); ctx.stroke();
    }

    // --- blocks with per-material detail (unchanged visual language) --------
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
      ctx.fillStyle = 'rgba(255,255,255,0.14)';
      ctx.fillRect(bMinX, bMinY, bw, bh * 0.32);
      ctx.fillStyle = 'rgba(143,211,255,0.08)';
      ctx.fillRect(bMinX, bMinY + bh * 0.8, bw, bh * 0.2);

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
        ctx.fillStyle = '#17131F';
        ctx.beginPath();
        ctx.moveTo(0, -hh * 0.28);
        ctx.bezierCurveTo(hw * 0.3, 0, hw * 0.24, hh * 0.28, 0, hh * 0.3);
        ctx.bezierCurveTo(-hw * 0.24, hh * 0.28, -hw * 0.3, 0, 0, -hh * 0.28);
        ctx.fill();
      }
      ctx.restore();

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
        poly(body.vertices, null, 'rgba(143,211,255,' + pulse.toFixed(2) + ')', 3);
        ctx.restore();
      }
    }

    // --- entities ------------------------------------------------------------
    function nearestProjectileDist(body) {
      let d = 1e9;
      game.activeBodies.forEach((p) => {
        d = Math.min(d, Math.hypot(p.position.x - body.position.x, p.position.y - body.position.y));
      });
      return d;
    }

    function drawEnemy(body, t) {
      const def = ENEMIES[body.plugin.type] || ENEMIES.stauber;
      const fear = nearestProjectileDist(body) < 340;
      ER.characters.drawEnemy(
        ctx, body.plugin.type, body.position.x, body.position.y, def.radius,
        body.velocity.x, body.velocity.y, t, body.id, body.plugin.squashT || 0, fear
      );
      // hurt flash (boss phases)
      if (body.plugin.hurtT > 0) {
        ctx.globalAlpha = (body.plugin.hurtT / 30) * 0.4;
        circle(body.position.x, body.position.y, def.radius * 1.1, '#E33A2C', null);
        ctx.globalAlpha = 1;
      }
      // hit-point pips over multi-phase enemies
      if ((body.plugin.maxHp || 1) > 1) {
        const n = body.plugin.maxHp;
        for (let i = 0; i < n; i++) {
          const px = body.position.x + (i - (n - 1) / 2) * 26;
          const py = body.position.y - def.radius - 26;
          circle(px, py, 9, i < body.plugin.hp ? '#E33A2C' : 'rgba(23,19,31,0.5)', PALETTE.line, 3);
        }
      }
      if (body.plugin.squashT > 0) body.plugin.squashT--;
    }

    function drawProjectileBody(body, t) {
      const type = body.plugin.type;
      const def = PROJECTILES[type] || PROJECTILES.ali;
      const r = body.circleRadius || def.radius;
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
      const wob = (!held && game.state !== 'idle') ? Math.sin(t * 0.55) * Math.max(0, 8 - (t % 60) * 0.4) : 0;
      ctx.strokeStyle = '#5A4A2A'; ctx.lineWidth = 12 - Math.min(5, pull * 0.015); ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(a.x + 26, a.y - 70); ctx.lineTo(pos.x + wob, pos.y); ctx.stroke();
      if (held) {
        const def = PROJECTILES[game.currentType];
        ER.characters.drawProjectile(ctx, game.currentType, pos.x, pos.y, def.radius,
          (a.x - pos.x) * 0.05, (a.y - pos.y) * 0.05, t, 5, 0);
      }
      ctx.strokeStyle = PALETTE.titan; ctx.lineWidth = 16; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(a.x - 24, a.y + 120); ctx.lineTo(a.x - 24, a.y - 70);
      ctx.moveTo(a.x + 26, a.y + 120); ctx.lineTo(a.x + 26, a.y - 70); ctx.stroke();
      ctx.strokeStyle = PALETTE.line; ctx.lineWidth = 4;
      ctx.beginPath(); ctx.moveTo(a.x - 24, a.y + 120); ctx.lineTo(a.x - 24, a.y - 70);
      ctx.moveTo(a.x + 26, a.y + 120); ctx.lineTo(a.x + 26, a.y - 70); ctx.stroke();
      if (held) {
        ctx.strokeStyle = '#7A6636'; ctx.lineWidth = 12 - Math.min(5, pull * 0.015);
        ctx.beginPath(); ctx.moveTo(a.x - 24, a.y - 70); ctx.lineTo(pos.x, pos.y); ctx.stroke();
      }
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
      const dotR = 5 / Math.min(1, cam.zoom); // keep dots visible when zoomed out
      for (let i = 2; i < pts.length; i += 2) {
        const march = (i * 0.5 + t * 0.25) % 4;
        ctx.globalAlpha = 0.25 + 0.45 * Math.max(0, 1 - march);
        ctx.fillStyle = '#fff';
        ctx.beginPath(); ctx.arc(pts[i].x, pts[i].y, dotR, 0, Math.PI * 2); ctx.fill();
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
      ctx.translate(cam.x, cam.y - 300 / cam.zoom);
      ctx.scale(popIn / cam.zoom, popIn / cam.zoom);
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

      const sx = (Math.random() - 0.5) * cam.shake;
      const sy = (Math.random() - 0.5) * cam.shake;
      ctx.save();
      ctx.translate(WORLD.width / 2 + sx, WORLD.height / 2 + sy);
      ctx.scale(cam.zoom, cam.zoom);
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

      if (cam.flash > 0) {
        ctx.fillStyle = 'rgba(200,235,255,' + (cam.flash / 10 * 0.35).toFixed(2) + ')';
        ctx.fillRect(cam.x - WORLD.width * 2, cam.y - WORLD.height * 2, WORLD.width * 4, WORLD.height * 4);
      }
      drawFlash();
      ctx.restore();

      // vignette in screen space, drawn over the letterboxed stage
      const vg = ctx.createRadialGradient(
        WORLD.width / 2, WORLD.height / 2, WORLD.height * 0.55,
        WORLD.width / 2, WORLD.height / 2, WORLD.height * 1.05
      );
      vg.addColorStop(0, 'rgba(11,8,18,0)');
      vg.addColorStop(1, 'rgba(11,8,18,0.55)');
      ctx.fillStyle = vg;
      ctx.fillRect(0, 0, WORLD.width, WORLD.height);
    }

    resize();
    window.addEventListener('resize', resize);
    return { draw, resize, toWorld, view };
  }

  ER.render = { create };
})(typeof window !== 'undefined' ? window : globalThis);
