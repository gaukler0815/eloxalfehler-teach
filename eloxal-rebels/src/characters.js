/*
 * characters.js — the cast. Every rebel and every defect gets a distinct
 * silhouette, a face with eyes that look and blink, and squash & stretch
 * driven by velocity (design bible: animation through deformation, never
 * frame sequences; contour #17131F everywhere; volume via 2-3 color steps).
 * Pure drawing: takes a ctx, a position/velocity and the frame counter.
 */
(function (global) {
  'use strict';
  const ER = (global.ER = global.ER || {});
  const LINE = '#17131F';

  function shade(hex, amt) {
    const n = parseInt(hex.slice(1), 16);
    const r = Math.max(0, Math.min(255, ((n >> 16) & 255) + amt));
    const g = Math.max(0, Math.min(255, ((n >> 8) & 255) + amt));
    const b = Math.max(0, Math.min(255, (n & 255) + amt));
    return 'rgb(' + r + ',' + g + ',' + b + ')';
  }
  function seedOf(id) { return ((id * 2654435761) >>> 0) % 997; }

  // Squash & stretch factor from velocity: fast = stretched along motion.
  function deform(ctx, x, y, vx, vy, extraSquash) {
    const speed = Math.hypot(vx || 0, vy || 0);
    let k = Math.min(0.24, speed * 0.011);
    if (extraSquash > 0) k = -Math.min(0.3, extraSquash * 0.03); // impact squash
    const ang = speed > 0.5 ? Math.atan2(vy, vx) : 0;
    ctx.translate(x, y);
    ctx.rotate(ang);
    ctx.scale(1 + k, 1 - k);
    ctx.rotate(-ang);
  }

  // One eye. look = -1..1 both axes; blink 0..1 (1 = closed).
  function eye(ctx, x, y, r, lookX, lookY, blink) {
    ctx.fillStyle = '#fff';
    ctx.strokeStyle = LINE; ctx.lineWidth = Math.max(2.5, r * 0.18);
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    if (blink > 0.6) {
      ctx.beginPath(); ctx.moveTo(x - r, y); ctx.lineTo(x + r, y);
      ctx.lineWidth = Math.max(3, r * 0.3); ctx.stroke();
      return;
    }
    ctx.fillStyle = LINE;
    ctx.beginPath();
    ctx.arc(x + lookX * r * 0.36, y + lookY * r * 0.36, r * 0.42, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    ctx.arc(x + lookX * r * 0.36 - r * 0.14, y + lookY * r * 0.36 - r * 0.14, r * 0.13, 0, Math.PI * 2);
    ctx.fill();
  }

  function faceParams(t, id, vx, vy) {
    const seed = seedOf(id || 1);
    const phase = (t + seed) % 190;
    const blink = phase < 7 ? 1 : 0;
    const speed = Math.hypot(vx || 0, vy || 0);
    const lookX = speed > 1 ? (vx / (speed || 1)) : 0.55;
    const lookY = speed > 1 ? (vy / (speed || 1)) * 0.7 : 0.1;
    return { blink, lookX, lookY, seed };
  }

  function outlined(ctx, drawPath, fill, lw) {
    drawPath();
    ctx.fillStyle = fill; ctx.fill();
    ctx.strokeStyle = LINE; ctx.lineWidth = lw || 5; ctx.lineJoin = 'round'; ctx.stroke();
  }
  function glossy(ctx, x, y, r) {
    ctx.fillStyle = 'rgba(255,255,255,0.55)';
    ctx.beginPath(); ctx.ellipse(x - r * 0.32, y - r * 0.38, r * 0.3, r * 0.18, -0.6, 0, Math.PI * 2); ctx.fill();
  }

  /*
   * PROJECTILES — each rebel is recognisable at a glance.
   */
  const projectileArt = {
    // Ali: the plucky sheet-metal square with a big grin. The everyman.
    ali(ctx, r, f) {
      outlined(ctx, () => {
        const w = r * 1.05;
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(-w, -w, w * 2, w * 2, r * 0.35);
        else ctx.rect(-w, -w, w * 2, w * 2);
      }, '#C9D2DC');
      ctx.fillStyle = shade('#C9D2DC', -22);
      ctx.fillRect(-r * 1.05, r * 0.35, r * 2.1, r * 0.6);
      glossy(ctx, 0, -r * 0.1, r * 1.15);
      eye(ctx, -r * 0.38, -r * 0.18, r * 0.3, f.lookX, f.lookY, f.blink);
      eye(ctx, r * 0.38, -r * 0.18, r * 0.3, f.lookX, f.lookY, f.blink);
      // determined smile
      ctx.strokeStyle = LINE; ctx.lineWidth = 4; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.arc(0, r * 0.28, r * 0.34, 0.15 * Math.PI, 0.85 * Math.PI); ctx.stroke();
    },
    // Bolle: heavy turned part, ring grooves, unibrow. Angry tank.
    bolle(ctx, r, f) {
      outlined(ctx, () => { ctx.beginPath(); ctx.arc(0, 0, r, 0, Math.PI * 2); }, '#B4A996');
      ctx.strokeStyle = shade('#B4A996', -40); ctx.lineWidth = 3;
      for (let i = 1; i <= 2; i++) { ctx.beginPath(); ctx.arc(0, 0, r * (1 - i * 0.22), 0, Math.PI * 2); ctx.stroke(); }
      ctx.fillStyle = shade('#B4A996', -26);
      ctx.beginPath(); ctx.arc(0, r * 0.45, r * 0.5, 0, Math.PI); ctx.fill();
      glossy(ctx, 0, 0, r);
      eye(ctx, -r * 0.34, -r * 0.1, r * 0.26, f.lookX, f.lookY, f.blink);
      eye(ctx, r * 0.34, -r * 0.1, r * 0.26, f.lookX, f.lookY, f.blink);
      ctx.strokeStyle = LINE; ctx.lineWidth = 6; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(-r * 0.62, -r * 0.46); ctx.lineTo(r * 0.62, -r * 0.46); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(-r * 0.22, r * 0.42); ctx.lineTo(r * 0.22, r * 0.42); ctx.lineWidth = 4; ctx.stroke();
    },
    // Rippi: cooling fins on top like a mohawk. Cheeky.
    rippi(ctx, r, f) {
      ctx.fillStyle = '#C9D2DC'; ctx.strokeStyle = LINE; ctx.lineWidth = 4;
      for (let i = -2; i <= 2; i++) {
        ctx.beginPath();
        ctx.rect(i * r * 0.34 - r * 0.1, -r * 1.35, r * 0.2, r * 0.6);
        ctx.fill(); ctx.stroke();
      }
      outlined(ctx, () => {
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(-r, -r * 0.85, r * 2, r * 1.75, r * 0.3);
        else ctx.rect(-r, -r * 0.85, r * 2, r * 1.75);
      }, '#C9D2DC');
      glossy(ctx, 0, -r * 0.2, r);
      eye(ctx, -r * 0.36, -r * 0.15, r * 0.28, f.lookX, f.lookY, f.blink);
      eye(ctx, r * 0.36, -r * 0.15, r * 0.28, f.lookX, f.lookY, f.blink);
      ctx.strokeStyle = LINE; ctx.lineWidth = 4; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.arc(-r * 0.05, r * 0.3, r * 0.3, 0.1 * Math.PI, 0.8 * Math.PI); ctx.stroke();
    },
    // Titania: titanium hook on the head, calm confident eyes.
    titania(ctx, r, f) {
      ctx.strokeStyle = LINE; ctx.lineWidth = 8; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.arc(0, -r * 1.15, r * 0.42, Math.PI * 0.9, Math.PI * 2.05); ctx.stroke();
      ctx.strokeStyle = '#B4A996'; ctx.lineWidth = 5;
      ctx.beginPath(); ctx.arc(0, -r * 1.15, r * 0.42, Math.PI * 0.9, Math.PI * 2.05); ctx.stroke();
      outlined(ctx, () => { ctx.beginPath(); ctx.arc(0, 0, r, 0, Math.PI * 2); }, '#B4A996');
      ctx.fillStyle = shade('#B4A996', -22);
      ctx.beginPath(); ctx.arc(0, r * 0.4, r * 0.62, 0, Math.PI); ctx.fill();
      glossy(ctx, 0, 0, r);
      eye(ctx, -r * 0.32, -r * 0.12, r * 0.27, f.lookX, f.lookY, f.blink);
      eye(ctx, r * 0.32, -r * 0.12, r * 0.27, f.lookX, f.lookY, f.blink);
      // lashes
      ctx.strokeStyle = LINE; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(-r * 0.55, -r * 0.38); ctx.lineTo(-r * 0.42, -r * 0.3);
      ctx.moveTo(r * 0.55, -r * 0.38); ctx.lineTo(r * 0.42, -r * 0.3); ctx.stroke();
      ctx.beginPath(); ctx.arc(0, r * 0.32, r * 0.22, 0.15 * Math.PI, 0.85 * Math.PI); ctx.lineWidth = 4; ctx.stroke();
    },
    // Bubbles: translucent hydrogen bubble, huge happy eyes.
    bubbles(ctx, r, f) {
      ctx.globalAlpha = 0.85;
      outlined(ctx, () => { ctx.beginPath(); ctx.arc(0, 0, r, 0, Math.PI * 2); }, '#7FB6F5', 4);
      ctx.globalAlpha = 1;
      ctx.strokeStyle = 'rgba(255,255,255,0.8)'; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.arc(0, 0, r * 0.72, -1.2, 0.3); ctx.stroke();
      glossy(ctx, 0, 0, r * 1.15);
      eye(ctx, -r * 0.3, -r * 0.08, r * 0.32, f.lookX, f.lookY - 0.25, f.blink);
      eye(ctx, r * 0.3, -r * 0.08, r * 0.32, f.lookX, f.lookY - 0.25, f.blink);
      ctx.strokeStyle = LINE; ctx.lineWidth = 4; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.arc(0, r * 0.3, r * 0.28, 0.1 * Math.PI, 0.9 * Math.PI); ctx.stroke();
    },
    // Säuri: green acid teardrop, mischievous grin.
    saeuri(ctx, r, f) {
      outlined(ctx, () => {
        ctx.beginPath();
        ctx.moveTo(0, -r * 1.4);
        ctx.bezierCurveTo(r * 1.05, -r * 0.25, r * 0.95, r * 0.9, 0, r);
        ctx.bezierCurveTo(-r * 0.95, r * 0.9, -r * 1.05, -r * 0.25, 0, -r * 1.4);
      }, '#7ED957');
      ctx.fillStyle = shade('#7ED957', -34);
      ctx.beginPath(); ctx.ellipse(0, r * 0.55, r * 0.55, r * 0.3, 0, 0, Math.PI * 2); ctx.fill();
      glossy(ctx, 0, -r * 0.3, r);
      eye(ctx, -r * 0.3, -r * 0.05, r * 0.26, f.lookX, f.lookY, f.blink);
      eye(ctx, r * 0.3, -r * 0.05, r * 0.26, f.lookX, f.lookY, f.blink);
      ctx.strokeStyle = LINE; ctx.lineWidth = 4; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.arc(r * 0.05, r * 0.32, r * 0.3, 0.2 * Math.PI, 0.95 * Math.PI); ctx.stroke();
    },
    // Bürsti: blasting nozzle with bristles, wild look.
    buersti(ctx, r, f) {
      ctx.strokeStyle = LINE; ctx.lineWidth = 3.5; ctx.lineCap = 'round';
      for (let i = -3; i <= 3; i++) {
        const a = -Math.PI / 2 + i * 0.22;
        ctx.beginPath();
        ctx.moveTo(Math.cos(a) * r * 0.9, Math.sin(a) * r * 0.9);
        ctx.lineTo(Math.cos(a) * r * 1.4, Math.sin(a) * r * 1.4);
        ctx.stroke();
      }
      outlined(ctx, () => { ctx.beginPath(); ctx.arc(0, 0, r, 0, Math.PI * 2); }, '#F5A81C');
      ctx.fillStyle = shade('#F5A81C', -30);
      ctx.beginPath(); ctx.arc(0, r * 0.45, r * 0.55, 0, Math.PI); ctx.fill();
      glossy(ctx, 0, 0, r);
      eye(ctx, -r * 0.33, -r * 0.1, r * 0.28, f.lookX, f.lookY, f.blink);
      eye(ctx, r * 0.33, -r * 0.1, r * 0.28, f.lookX, f.lookY, 0); // one eye never blinks: wired
      ctx.strokeStyle = LINE; ctx.lineWidth = 4;
      ctx.beginPath(); ctx.moveTo(-r * 0.2, r * 0.4); ctx.lineTo(r * 0.25, r * 0.32); ctx.stroke();
    },
    // Lasar: cyclops laser visor. Cool customer.
    lasar(ctx, r, f) {
      outlined(ctx, () => {
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(-r, -r * 0.9, r * 2, r * 1.8, r * 0.45);
        else ctx.rect(-r, -r * 0.9, r * 2, r * 1.8);
      }, '#E33A2C');
      ctx.fillStyle = shade('#E33A2C', -38);
      ctx.fillRect(-r, r * 0.25, r * 2, r * 0.55);
      // visor
      ctx.fillStyle = LINE;
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(-r * 0.75, -r * 0.5, r * 1.5, r * 0.62, r * 0.3);
      else ctx.rect(-r * 0.75, -r * 0.5, r * 1.5, r * 0.62);
      ctx.fill();
      const scan = f.blink ? 0 : Math.sin((f.seed + Date.now() * 0.004)) * r * 0.4;
      ctx.fillStyle = '#FF6B5E';
      ctx.beginPath(); ctx.arc(scan, -r * 0.19, r * 0.16, 0, Math.PI * 2); ctx.fill();
      glossy(ctx, 0, -r * 0.5, r);
      ctx.strokeStyle = LINE; ctx.lineWidth = 4; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(-r * 0.3, r * 0.5); ctx.lineTo(r * 0.3, r * 0.5); ctx.stroke();
    }
  };

  /*
   * ENEMIES — the defect gang. All grumpy, all recognisable from the shop floor.
   */
  const enemyArt = {
    // Stauber: fuzzy dust ball.
    stauber(ctx, r, f, t) {
      outlined(ctx, () => {
        ctx.beginPath();
        for (let i = 0; i <= 14; i++) {
          const a = (i / 14) * Math.PI * 2;
          const rr = r * (1 + 0.14 * Math.sin(a * 5 + f.seed + t * 0.03));
          const x = Math.cos(a) * rr, y = Math.sin(a) * rr;
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.closePath();
      }, '#6E6A7A', 4);
      ctx.fillStyle = shade('#6E6A7A', -18);
      ctx.beginPath(); ctx.arc(0, r * 0.3, r * 0.6, 0, Math.PI); ctx.fill();
      eye(ctx, -r * 0.3, -r * 0.1, r * 0.24, -f.lookX, f.lookY, f.blink);
      eye(ctx, r * 0.3, -r * 0.1, r * 0.24, -f.lookX, f.lookY, f.blink);
      ctx.strokeStyle = LINE; ctx.lineWidth = 4; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(-r * 0.5, -r * 0.42); ctx.lineTo(-r * 0.15, -r * 0.28);
      ctx.moveTo(r * 0.5, -r * 0.42); ctx.lineTo(r * 0.15, -r * 0.28); ctx.stroke();
      ctx.beginPath(); ctx.arc(0, r * 0.45, r * 0.2, Math.PI * 1.15, Math.PI * 1.85); ctx.stroke();
    },
    // Lochfraß-Lenny: pitted green troublemaker with a drill snout.
    lenny(ctx, r, f) {
      outlined(ctx, () => { ctx.beginPath(); ctx.arc(0, 0, r, 0, Math.PI * 2); }, '#3B7A57');
      ctx.fillStyle = shade('#3B7A57', -30);
      [[-0.45, 0.3, 0.2], [0.5, 0.15, 0.16], [0.1, 0.55, 0.24], [-0.2, -0.5, 0.13]].forEach(([px, py, pr]) => {
        ctx.beginPath(); ctx.arc(px * r, py * r, pr * r, 0, Math.PI * 2); ctx.fill();
      });
      eye(ctx, -r * 0.3, -r * 0.15, r * 0.24, -f.lookX, f.lookY, f.blink);
      eye(ctx, r * 0.3, -r * 0.15, r * 0.24, -f.lookX, f.lookY, f.blink);
      // drill mouth
      ctx.fillStyle = '#8A94A0'; ctx.strokeStyle = LINE; ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(-r * 0.25, r * 0.35); ctx.lineTo(r * 0.25, r * 0.35); ctx.lineTo(0, r * 0.75); ctx.closePath();
      ctx.fill(); ctx.stroke();
      ctx.strokeStyle = LINE; ctx.lineWidth = 4; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(-r * 0.5, -r * 0.45); ctx.lineTo(-r * 0.12, -r * 0.32);
      ctx.moveTo(r * 0.5, -r * 0.45); ctx.lineTo(r * 0.12, -r * 0.32); ctx.stroke();
    },
    // Fetti: oily film, droopy lids, self-satisfied smirk.
    fetti(ctx, r, f) {
      outlined(ctx, () => { ctx.beginPath(); ctx.ellipse(0, r * 0.1, r * 1.1, r * 0.9, 0, 0, Math.PI * 2); }, '#C7B24A');
      const g = ctx.createLinearGradient(-r, -r, r, r);
      g.addColorStop(0, 'rgba(255,255,255,0.35)');
      g.addColorStop(0.5, 'rgba(126,182,245,0.25)');
      g.addColorStop(1, 'rgba(227,58,44,0.18)');
      ctx.fillStyle = g;
      ctx.beginPath(); ctx.ellipse(0, r * 0.1, r * 1.05, r * 0.85, 0, 0, Math.PI * 2); ctx.fill();
      eye(ctx, -r * 0.32, -r * 0.12, r * 0.26, -f.lookX, f.lookY, Math.max(f.blink, 0.0));
      eye(ctx, r * 0.32, -r * 0.12, r * 0.26, -f.lookX, f.lookY, Math.max(f.blink, 0.0));
      // droopy lids
      ctx.fillStyle = '#C7B24A'; ctx.strokeStyle = LINE; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.arc(-r * 0.32, -r * 0.2, r * 0.27, Math.PI, 0); ctx.fill(); ctx.stroke();
      ctx.beginPath(); ctx.arc(r * 0.32, -r * 0.2, r * 0.27, Math.PI, 0); ctx.fill(); ctx.stroke();
      ctx.strokeStyle = LINE; ctx.lineWidth = 4; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.arc(r * 0.08, r * 0.35, r * 0.28, Math.PI * 1.05, Math.PI * 1.8); ctx.stroke();
    },
    // Kalki: crystalline armor, smug because he re-hardens.
    kalki(ctx, r, f) {
      outlined(ctx, () => {
        ctx.beginPath();
        const pts = 7;
        for (let i = 0; i <= pts; i++) {
          const a = (i / pts) * Math.PI * 2 - Math.PI / 2;
          const rr = r * (i % 2 === 0 ? 1.15 : 0.88);
          const x = Math.cos(a) * rr, y = Math.sin(a) * rr;
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.closePath();
      }, '#E7E2D6', 4);
      ctx.fillStyle = 'rgba(255,255,255,0.5)';
      ctx.beginPath(); ctx.moveTo(-r * 0.5, -r * 0.5); ctx.lineTo(0, -r * 0.9); ctx.lineTo(r * 0.15, -r * 0.3); ctx.closePath(); ctx.fill();
      eye(ctx, -r * 0.28, -r * 0.05, r * 0.22, -f.lookX, f.lookY, f.blink);
      eye(ctx, r * 0.28, -r * 0.05, r * 0.22, -f.lookX, f.lookY, f.blink);
      ctx.strokeStyle = LINE; ctx.lineWidth = 4; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(-r * 0.45, -r * 0.35); ctx.lineTo(-r * 0.1, -r * 0.22);
      ctx.moveTo(r * 0.45, -r * 0.35); ctx.lineTo(r * 0.1, -r * 0.22); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(-r * 0.2, r * 0.42); ctx.lineTo(r * 0.2, r * 0.42); ctx.stroke();
    },
    // Baron Korrosius: the white-rust boss with a rusty crown.
    korrosius(ctx, r, f) {
      // crown
      ctx.fillStyle = '#9A5B2B'; ctx.strokeStyle = LINE; ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(-r * 0.6, -r * 0.85);
      ctx.lineTo(-r * 0.6, -r * 1.25); ctx.lineTo(-r * 0.3, -r * 1.0);
      ctx.lineTo(0, -r * 1.3); ctx.lineTo(r * 0.3, -r * 1.0);
      ctx.lineTo(r * 0.6, -r * 1.25); ctx.lineTo(r * 0.6, -r * 0.85);
      ctx.closePath(); ctx.fill(); ctx.stroke();
      outlined(ctx, () => {
        ctx.beginPath();
        for (let i = 0; i <= 16; i++) {
          const a = (i / 16) * Math.PI * 2;
          const rr = r * (1 + 0.08 * Math.sin(a * 6 + f.seed));
          if (i === 0) ctx.moveTo(Math.cos(a) * rr, Math.sin(a) * rr);
          else ctx.lineTo(Math.cos(a) * rr, Math.sin(a) * rr);
        }
        ctx.closePath();
      }, '#F0EAF2', 5);
      ctx.fillStyle = shade('#F0EAF2', -26);
      [[-0.4, 0.4, 0.22], [0.45, 0.28, 0.18], [0.05, -0.45, 0.15]].forEach(([px, py, pr]) => {
        ctx.beginPath(); ctx.arc(px * r, py * r, pr * r, 0, Math.PI * 2); ctx.fill();
      });
      eye(ctx, -r * 0.28, -r * 0.08, r * 0.2, -f.lookX, f.lookY, f.blink);
      eye(ctx, r * 0.28, -r * 0.08, r * 0.2, -f.lookX, f.lookY, f.blink);
      ctx.strokeStyle = LINE; ctx.lineWidth = 5; ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(-r * 0.48, -r * 0.32); ctx.lineTo(-r * 0.1, -r * 0.2);
      ctx.moveTo(r * 0.48, -r * 0.32); ctx.lineTo(r * 0.1, -r * 0.2); ctx.stroke();
      ctx.beginPath(); ctx.arc(0, r * 0.5, r * 0.25, Math.PI * 1.1, Math.PI * 1.9); ctx.stroke();
    }
  };

  /*
   * Public API. Draws with squash & stretch applied from velocity; `squashT`
   * (set by the game on impacts) flips the deformation into a squash.
   */
  function drawProjectile(ctx, type, x, y, r, vx, vy, t, id, squashT) {
    const f = faceParams(t, id, vx, vy);
    ctx.save();
    deform(ctx, x, y, vx, vy, squashT || 0);
    (projectileArt[type] || projectileArt.ali)(ctx, r, f);
    ctx.restore();
  }

  function drawEnemy(ctx, type, x, y, r, vx, vy, t, id, squashT, fear) {
    const f = faceParams(t, id, vx, vy);
    if (fear) f.blink = 0; // scared enemies don't blink
    const bob = Math.sin((t + f.seed) * 0.05) * (fear ? 4 : 2); // trembling when scared
    const breathe = 1 + (fear ? 0.05 : 0.025) * Math.sin((t + f.seed) * (fear ? 0.4 : 0.08));
    ctx.save();
    ctx.translate(x, y + bob);
    ctx.scale(breathe, 2 - breathe);
    if (squashT > 0) { const k = Math.min(0.3, squashT * 0.03); ctx.scale(1 + k, 1 - k); }
    (enemyArt[type] || enemyArt.stauber)(ctx, r, f, t);
    if (fear) {
      // open "uh-oh" mouth over whatever the resting face was
      ctx.fillStyle = LINE;
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.ellipse(0, r * 0.42, r * 0.2, r * 0.26, 0, 0, Math.PI * 2);
      ctx.fill(); ctx.stroke();
      // sweat drop
      ctx.fillStyle = '#8FD3FF';
      ctx.beginPath();
      ctx.moveTo(r * 0.78, -r * 0.72);
      ctx.bezierCurveTo(r * 0.95, -r * 0.5, r * 0.9, -r * 0.38, r * 0.76, -r * 0.38);
      ctx.bezierCurveTo(r * 0.62, -r * 0.38, r * 0.6, -r * 0.52, r * 0.78, -r * 0.72);
      ctx.fill();
    }
    ctx.restore();
  }

  // For HUD chips and the held slingshot character: idle face, no motion.
  function drawStatic(ctx, type, x, y, r, t) {
    drawProjectile(ctx, type, x, y, r, 0, 0, t || 0, 7, 0);
  }

  ER.characters = { drawProjectile, drawEnemy, drawStatic };
})(typeof window !== 'undefined' ? window : globalThis);
