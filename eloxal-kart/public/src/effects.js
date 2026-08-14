// Particle effects: drift sparks, offroad dust, hit explosions, shield pop
// and finish confetti. One pooled Points-based system per effect color class.

import * as THREE from 'three';

const MAX = 600;

export class Effects {
  constructor(scene) {
    this.geo = new THREE.BufferGeometry();
    this.positions = new Float32Array(MAX * 3);
    this.colors = new Float32Array(MAX * 3);
    this.geo.setAttribute('position', new THREE.BufferAttribute(this.positions, 3));
    this.geo.setAttribute('color', new THREE.BufferAttribute(this.colors, 3));
    // Soft round sprite so particles do not render as hard squares.
    const c = document.createElement('canvas');
    c.width = c.height = 64;
    const g = c.getContext('2d');
    const grad = g.createRadialGradient(32, 32, 4, 32, 32, 30);
    grad.addColorStop(0, 'rgba(255,255,255,1)');
    grad.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = grad;
    g.fillRect(0, 0, 64, 64);
    this.mat = new THREE.PointsMaterial({
      size: 0.8, vertexColors: true, transparent: true, opacity: 0.9,
      depthWrite: false, sizeAttenuation: true,
      map: new THREE.CanvasTexture(c), alphaTest: 0.05,
      blending: THREE.AdditiveBlending, // fade-to-black doubles as fade-out
    });
    this.points = new THREE.Points(this.geo, this.mat);
    this.points.frustumCulled = false;
    scene.add(this.points);
    this.particles = []; // { x,y,z, vx,vy,vz, life, maxLife, r,g,b, gravity }
  }

  spawn(n, opts) {
    for (let i = 0; i < n; i++) {
      if (this.particles.length >= MAX) this.particles.shift();
      const spread = opts.spread ?? 1;
      this.particles.push({
        x: opts.x + (Math.random() - 0.5) * (opts.jitter ?? 0.5),
        y: opts.y ?? 0.4,
        z: opts.z + (Math.random() - 0.5) * (opts.jitter ?? 0.5),
        vx: (opts.vx ?? 0) + (Math.random() - 0.5) * spread,
        vy: (opts.vy ?? 2) + Math.random() * spread,
        vz: (opts.vz ?? 0) + (Math.random() - 0.5) * spread,
        life: 0,
        maxLife: (opts.life ?? 0.6) * (0.6 + Math.random() * 0.8),
        r: opts.color.r, g: opts.color.g, b: opts.color.b,
        gravity: opts.gravity ?? 6,
      });
    }
  }

  driftSparks(x, z, heading, driftDir, level) {
    // level 0: blue, 1: orange, 2: purple-white (big boost ready)
    const col = level >= 2 ? { r: 1, g: 0.5, b: 1 } : level >= 1 ? { r: 1, g: 0.6, b: 0.15 } : { r: 0.3, g: 0.7, b: 1 };
    const bx = x - Math.sin(heading) * 1.4, bz = z - Math.cos(heading) * 1.4;
    this.spawn(2, { x: bx, z: bz, y: 0.25, color: col, vy: 1.5, spread: 3, life: 0.35, gravity: 4 });
  }

  dust(x, z) {
    this.spawn(1, { x, z, y: 0.3, color: { r: 0.85, g: 0.72, b: 0.5 }, vy: 1.2, spread: 1.5, life: 0.8, gravity: 0.5 });
  }

  boostTrail(x, z, heading) {
    const bx = x - Math.sin(heading) * 2.2, bz = z - Math.cos(heading) * 2.2;
    this.spawn(2, { x: bx, z: bz, y: 0.8, color: { r: 1, g: 0.55, b: 0.1 }, vy: 0.5, spread: 1.2, life: 0.4, gravity: -1 });
  }

  explosion(x, z) {
    this.spawn(26, { x, z, y: 1, color: { r: 1, g: 0.65, b: 0.1 }, vy: 5, spread: 9, life: 0.7, gravity: 9 });
    this.spawn(14, { x, z, y: 1, color: { r: 0.4, g: 0.4, b: 0.42 }, vy: 4, spread: 5, life: 1, gravity: 3 });
  }

  shieldPop(x, z) {
    this.spawn(20, { x, z, y: 1.2, color: { r: 0.45, g: 0.8, b: 1 }, vy: 3, spread: 6, life: 0.5, gravity: 4 });
  }

  pickup(x, z) {
    this.spawn(12, { x, z, y: 1.6, color: { r: 1, g: 0.85, b: 0.3 }, vy: 3.5, spread: 4, life: 0.5, gravity: 5 });
  }

  confetti(x, z) {
    for (const c of [{ r: 1, g: 0.2, b: 0.3 }, { r: 0.3, g: 0.9, b: 0.4 }, { r: 1, g: 0.85, b: 0.2 }, { r: 0.3, g: 0.6, b: 1 }]) {
      this.spawn(10, { x, z, y: 4, color: c, vy: 6, spread: 8, life: 1.6, gravity: 5 });
    }
  }

  update(dt) {
    const P = this.particles;
    for (let i = P.length - 1; i >= 0; i--) {
      const p = P[i];
      p.life += dt;
      if (p.life >= p.maxLife) { P.splice(i, 1); continue; }
      p.vy -= p.gravity * dt;
      p.x += p.vx * dt; p.y += p.vy * dt; p.z += p.vz * dt;
      if (p.y < 0.05) { p.y = 0.05; p.vy *= -0.3; }
    }
    for (let i = 0; i < MAX; i++) {
      if (i < P.length) {
        const p = P[i];
        const fade = 1 - p.life / p.maxLife;
        this.positions.set([p.x, p.y, p.z], i * 3);
        this.colors.set([p.r * fade, p.g * fade, p.b * fade], i * 3);
      } else {
        this.positions.set([0, -100, 0], i * 3);
      }
    }
    this.geo.attributes.position.needsUpdate = true;
    this.geo.attributes.color.needsUpdate = true;
  }
}
