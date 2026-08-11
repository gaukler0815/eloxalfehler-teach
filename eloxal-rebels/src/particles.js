/*
 * particles.js — the dust, shards, sparks and confetti that make hits feel
 * physical. A single pooled array of typed particles; the game emits events,
 * main.js maps them to spawns, render.js calls draw() inside the camera
 * transform. No game rules live here.
 */
(function (global) {
  'use strict';
  const ER = (global.ER = global.ER || {});

  const MAX = 420; // hard cap so phones never drown in particles
  const pool = [];
  const GRAV = 0.34;

  function push(p) {
    if (pool.length >= MAX) pool.shift();
    p.age = 0;
    pool.push(p);
  }
  function rand(a, b) { return a + Math.random() * (b - a); }

  // --- spawn helpers -------------------------------------------------------

  // Soft dust puff, e.g. a projectile thumping into cardboard.
  function dust(x, y, strength) {
    const n = Math.min(10, 3 + Math.round(strength / 3));
    for (let i = 0; i < n; i++) {
      push({
        type: 'smoke', x: x + rand(-14, 14), y: y + rand(-10, 10),
        vx: rand(-1.6, 1.6), vy: rand(-1.8, -0.3),
        size: rand(9, 22), life: rand(26, 46), color: '201,210,220'
      });
    }
  }

  // Angular debris when a block breaks. Colored like its material.
  function shards(x, y, colorHex, n) {
    for (let i = 0; i < (n || 10); i++) {
      const a = rand(0, Math.PI * 2), s = rand(3, 11);
      push({
        type: 'shard', x, y,
        vx: Math.cos(a) * s, vy: Math.sin(a) * s - 4,
        size: rand(5, 13), rot: rand(0, Math.PI * 2), vr: rand(-0.3, 0.3),
        life: rand(34, 60), color: colorHex
      });
    }
  }

  // Electric sparks along the arc chain segments.
  function sparks(segs) {
    segs.forEach((seg) => {
      for (let i = 0; i < 5; i++) {
        const t = Math.random();
        const x = seg.a.x + (seg.b.x - seg.a.x) * t;
        const y = seg.a.y + (seg.b.y - seg.a.y) * t;
        const a = rand(0, Math.PI * 2), s = rand(3, 9);
        push({
          type: 'spark', x, y,
          vx: Math.cos(a) * s, vy: Math.sin(a) * s - 2,
          life: rand(12, 26), color: '#8FD3FF'
        });
      }
    });
  }

  // Round burst, e.g. an enemy popping.
  function burst(x, y, colorHex, n, speed) {
    for (let i = 0; i < (n || 12); i++) {
      const a = (i / (n || 12)) * Math.PI * 2 + rand(-0.2, 0.2);
      const s = (speed || 7) * rand(0.5, 1.1);
      push({
        type: 'dot', x, y,
        vx: Math.cos(a) * s, vy: Math.sin(a) * s - 2,
        size: rand(4, 9), life: rand(20, 38), color: colorHex
      });
    }
  }

  // Floating text ("+20 µm", "ZACK").
  function pop(x, y, text, colorHex) {
    push({ type: 'text', x, y, vx: 0, vy: -1.4, life: 55, text, color: colorHex || '#F5A81C', size: 40 });
  }

  // Level-won confetti in the Eloxal palette.
  function confetti(x, y) {
    const colors = ['#1E74DC', '#F5A81C', '#E33A2C', '#C9D2DC', '#B4A996'];
    for (let i = 0; i < 60; i++) {
      const a = rand(-Math.PI, 0);
      const s = rand(6, 16);
      push({
        type: 'shard', x: x + rand(-120, 120), y,
        vx: Math.cos(a) * s * 0.5, vy: Math.sin(a) * s,
        size: rand(6, 12), rot: rand(0, Math.PI * 2), vr: rand(-0.4, 0.4),
        life: rand(60, 110), color: colors[i % colors.length], flutter: true
      });
    }
  }

  // Gentle rising steam wisp (ambient, spawned by the renderer).
  function steam(x, y) {
    push({
      type: 'smoke', x: x + rand(-16, 16), y,
      vx: rand(-0.3, 0.3), vy: rand(-1.0, -0.5),
      size: rand(14, 30), life: rand(60, 110), color: '160,190,220'
    });
  }

  // --- simulation ----------------------------------------------------------
  function update() {
    for (let i = pool.length - 1; i >= 0; i--) {
      const p = pool[i];
      p.age++;
      if (p.age >= p.life) { pool.splice(i, 1); continue; }
      p.x += p.vx;
      p.y += p.vy;
      if (p.type === 'shard' || p.type === 'dot') {
        p.vy += p.flutter ? GRAV * 0.35 : GRAV;
        p.vx *= p.flutter ? 0.96 : 0.99;
        if (p.rot != null) p.rot += p.vr;
        if (p.flutter) p.vx += Math.sin(p.age * 0.3) * 0.25;
      } else if (p.type === 'smoke') {
        p.vy *= 0.98;
        p.size *= 1.015;
      } else if (p.type === 'spark') {
        p.vy += GRAV * 0.5;
      }
    }
  }

  // --- rendering (called inside the world/camera transform) ----------------
  function draw(ctx) {
    for (const p of pool) {
      const k = 1 - p.age / p.life;
      if (p.type === 'dot') {
        ctx.globalAlpha = k;
        ctx.fillStyle = p.color;
        ctx.beginPath(); ctx.arc(p.x, p.y, p.size * k + 1, 0, Math.PI * 2); ctx.fill();
      } else if (p.type === 'shard') {
        ctx.save();
        ctx.globalAlpha = Math.min(1, k * 1.6);
        ctx.translate(p.x, p.y); ctx.rotate(p.rot);
        ctx.fillStyle = p.color;
        ctx.fillRect(-p.size / 2, -p.size / 4, p.size, p.size / 2);
        ctx.restore();
      } else if (p.type === 'smoke') {
        ctx.globalAlpha = 0.28 * k;
        ctx.fillStyle = 'rgba(' + p.color + ',1)';
        ctx.beginPath(); ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2); ctx.fill();
      } else if (p.type === 'spark') {
        ctx.globalAlpha = k;
        ctx.strokeStyle = p.color; ctx.lineWidth = 3; ctx.lineCap = 'round';
        ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(p.x - p.vx * 1.6, p.y - p.vy * 1.6); ctx.stroke();
      } else if (p.type === 'text') {
        ctx.globalAlpha = Math.min(1, k * 1.5);
        ctx.font = '800 ' + p.size + 'px system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.lineWidth = 7; ctx.strokeStyle = '#17131F';
        ctx.strokeText(p.text, p.x, p.y);
        ctx.fillStyle = p.color;
        ctx.fillText(p.text, p.x, p.y);
      }
    }
    ctx.globalAlpha = 1;
  }

  function clear() { pool.length = 0; }

  ER.particles = { dust, shards, sparks, burst, pop, confetti, steam, update, draw, clear, _pool: pool };
})(typeof window !== 'undefined' ? window : globalThis);
