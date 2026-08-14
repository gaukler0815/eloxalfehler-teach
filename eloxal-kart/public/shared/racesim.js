// The complete race simulation: racers, bots, progress/laps/ranks, item
// boxes, items, projectiles and hazards. Runs authoritatively on the server
// for online races and directly in the browser for solo races – identical
// rules in both modes.
//
// Racers marked `external: true` (online human players) are NOT integrated by
// stepKart here; their positions are written in from network reports. All
// race rules (laps, items, hits, spins) still apply to them authoritatively.

import { PHYS, RACE, PROJ, ITEMS, itemWeights, characterById } from './config.js';
import { makeKartState, stepKart, resolveKartCollision } from './kartphysics.js';

function wrapDelta(d, len) {
  d = d % len;
  if (d > len / 2) d -= len;
  if (d < -len / 2) d += len;
  return d;
}

export class RaceSim {
  constructor(track, opts = {}) {
    this.track = track;
    this.laps = opts.laps ?? RACE.laps;
    this.phase = 'waiting'; // waiting | countdown | racing | finished
    this.time = 0;          // race clock, starts at GO
    this.phaseT = 0;
    this.racers = [];
    this.projectiles = [];
    this.hazards = [];
    this.events = [];
    this.nextProjId = 1;
    this.nextHazardId = 1;
    this.finishedCount = 0;
    this.firstFinishTime = null;
    this.boxes = track.itemBoxes.map((b) => ({ ...b, active: true, respawnT: 0 }));
    this.rng = opts.rng || Math.random;
  }

  addRacer({ id, name, charId, isBot = false, external = false }) {
    const grid = this.track.startGrid[this.racers.length % this.track.startGrid.length];
    const char = characterById(charId);
    const kart = makeKartState(grid.x, grid.z, grid.heading, char);
    kart.trackS = grid.s;
    const racer = {
      id, name, char, isBot, external,
      kart,
      item: null,
      rouletteT: 0,
      dist: 0,
      lastS: grid.s,
      lap: 1,
      rank: this.racers.length + 1,
      finished: false,
      finishTime: null,
      itemUsedAt: -99,
      bot: isBot ? {
        skill: 0.82 + this.rng() * 0.22,
        targetLat: (this.rng() * 2 - 1) * 4,
        latTimer: 2 + this.rng() * 3,
        itemTimer: 1 + this.rng() * 2.5,
      } : null,
    };
    this.racers.push(racer);
    return racer;
  }

  getRacer(id) { return this.racers.find((r) => r.id === id); }

  start() {
    this.phase = 'countdown';
    this.phaseT = 0;
    this.time = 0;
    this.emit({ type: 'start' });
  }

  emit(ev) { this.events.push(ev); }
  drainEvents() { const e = this.events; this.events = []; return e; }

  // ------------------------------------------------------------------ items

  useItem(id) {
    const r = this.getRacer(id);
    if (!r || !r.item || r.finished || this.phase !== 'racing' || r.kart.spinT > 0) return;
    const item = r.item;
    r.item = null;
    r.itemUsedAt = this.time;
    const k = r.kart;
    const dirX = Math.sin(k.heading), dirZ = Math.cos(k.heading);

    if (item === 'turbo') {
      k.boostT = Math.max(k.boostT, 1.6);
    } else if (item === 'shield') {
      k.shieldT = PROJ.shieldTime;
    } else if (item === 'bolt') {
      this.projectiles.push({
        id: this.nextProjId++, type: 'bolt', owner: id,
        x: k.x + dirX * 3, z: k.z + dirZ * 3,
        dx: dirX, dz: dirZ,
        speed: PROJ.boltSpeed + Math.max(0, k.speed),
        life: PROJ.boltLife, bounces: PROJ.boltBounces, age: 0,
        trackS: k.trackS,
      });
    } else if (item === 'seeker') {
      // Target: the racer directly ahead in the standings.
      const ahead = this.racers
        .filter((o) => !o.finished && o.id !== id && o.dist > r.dist)
        .sort((a, b) => a.dist - b.dist)[0];
      this.projectiles.push({
        id: this.nextProjId++, type: 'seeker', owner: id,
        s: (k.trackS + 5) % this.track.length,
        lat: this.track.lateralOffset(k.x, k.z, k.trackS),
        target: ahead ? ahead.id : null,
        life: PROJ.seekerLife, age: 0,
        x: k.x, z: k.z,
      });
    } else if (item === 'barrel') {
      const bx = k.x - dirX * 4, bz = k.z - dirZ * 4;
      this.hazards.push({
        id: this.nextHazardId++, type: 'barrel', owner: id,
        x: bx, z: bz, life: PROJ.barrelLife, age: 0,
      });
    }
    this.emit({ type: 'fire', id, item });
  }

  rollItem(r) {
    const w = itemWeights(r.rank - 1, this.racers.length);
    let sum = 0;
    for (const k in w) sum += w[k];
    let pick = this.rng() * sum;
    for (const k in w) {
      pick -= w[k];
      if (pick <= 0) return k;
    }
    return 'bolt';
  }

  hitRacer(r, byId, item) {
    const k = r.kart;
    if (k.shieldT > 0) {
      k.shieldT = 0;
      this.emit({ type: 'shieldBreak', id: r.id });
      return;
    }
    k.spinT = PHYS.spinDuration;
    k.speed *= PHYS.spinSpeedFactor;
    k.drifting = false;
    this.emit({ type: 'hit', id: r.id, by: byId, item });
  }

  // ------------------------------------------------------------------ step

  step(dt, inputs = {}) {
    this.phaseT += dt;

    if (this.phase === 'countdown') {
      if (this.phaseT >= RACE.countdown) {
        this.phase = 'racing';
        this.phaseT = 0;
        this.emit({ type: 'go' });
      }
    }
    if (this.phase === 'racing' || this.phase === 'finished') this.time += dt;

    const racing = this.phase === 'racing';
    const track = this.track;

    // --- Move karts ---
    for (const r of this.racers) {
      if (r.external) {
        // Position comes from the network; keep derived track data fresh.
        r.kart.trackS = track.closestS(r.kart.x, r.kart.z, r.kart.trackS);
        continue;
      }
      let input = { throttle: 0, steer: 0, drift: false };
      if (racing && !r.finished) {
        input = r.isBot ? this.botInput(r, dt) : (inputs[r.id] || input);
      }
      stepKart(r.kart, input, dt, track);
    }

    // --- Kart vs kart collisions (never move external racers) ---
    for (let i = 0; i < this.racers.length; i++) {
      for (let j = i + 1; j < this.racers.length; j++) {
        const a = this.racers[i], b = this.racers[j];
        if (a.external && b.external) continue;
        const ax = a.kart.x, az = a.kart.z, bx = b.kart.x, bz = b.kart.z;
        if (resolveKartCollision(a.kart, b.kart, PHYS.kartRadius)) {
          if (a.external) { b.kart.x += a.kart.x - ax + (bx - b.kart.x) * 0; a.kart.x = ax; a.kart.z = az; }
          if (b.external) { b.kart.x = bx; b.kart.z = bz; }
        }
      }
    }

    // --- Progress, laps, finish ---
    if (racing) {
      for (const r of this.racers) {
        if (r.finished) continue;
        const s = r.kart.trackS;
        const d = wrapDelta(s - r.lastS, track.length);
        if (Math.abs(d) < 60) r.dist += d;
        r.lastS = s;
        const lap = Math.max(1, Math.floor(r.dist / track.length) + 1);
        if (lap > r.lap) {
          r.lap = lap;
          if (lap <= this.laps) this.emit({ type: 'lap', id: r.id, lap });
        }
        if (r.dist >= this.laps * track.length) {
          r.finished = true;
          r.finishTime = this.time;
          this.finishedCount++;
          if (this.firstFinishTime === null) this.firstFinishTime = this.time;
          this.emit({ type: 'finish', id: r.id, time: r.finishTime, place: this.finishedCount });
        }
      }

      // Ranks: finishers by finish time, the rest by distance.
      const order = [...this.racers].sort((a, b) => {
        if (a.finished && b.finished) return a.finishTime - b.finishTime;
        if (a.finished) return -1;
        if (b.finished) return 1;
        return b.dist - a.dist;
      });
      order.forEach((r, i) => { r.rank = i + 1; });

      // Race end: everyone finished, or timeout after the winner.
      const allDone = this.finishedCount >= this.racers.length;
      const timedOut = this.firstFinishTime !== null &&
        this.time - this.firstFinishTime > RACE.finishTimeout;
      if (allDone || timedOut) {
        for (const r of this.racers) {
          if (!r.finished) {
            r.finished = true;
            r.finishTime = null; // DNF
            this.finishedCount++;
          }
        }
        this.phase = 'finished';
        this.phaseT = 0;
        this.emit({ type: 'raceOver' });
      }
    }

    // --- Item boxes ---
    for (const box of this.boxes) {
      if (!box.active) {
        box.respawnT -= dt;
        if (box.respawnT <= 0) {
          box.active = true;
          this.emit({ type: 'boxRespawn', boxId: box.id });
        }
        continue;
      }
      if (!racing) continue;
      for (const r of this.racers) {
        if (r.finished || r.item || r.rouletteT > 0) continue;
        const dx = r.kart.x - box.x, dz = r.kart.z - box.z;
        if (dx * dx + dz * dz < 2.6 * 2.6) {
          box.active = false;
          box.respawnT = RACE.boxRespawn;
          r.rouletteT = RACE.rouletteTime;
          this.emit({ type: 'boxTaken', boxId: box.id, id: r.id });
          break;
        }
      }
    }
    for (const r of this.racers) {
      if (r.rouletteT > 0) {
        r.rouletteT -= dt;
        if (r.rouletteT <= 0) {
          r.item = this.rollItem(r);
          this.emit({ type: 'item', id: r.id, item: r.item });
        }
      }
    }

    // --- Projectiles ---
    this.stepProjectiles(dt);

    // --- Hazards ---
    for (let i = this.hazards.length - 1; i >= 0; i--) {
      const h = this.hazards[i];
      h.age += dt; h.life -= dt;
      let dead = h.life <= 0;
      if (!dead && racing) {
        for (const r of this.racers) {
          if (r.finished || r.kart.spinT > 0) continue;
          if (r.id === h.owner && h.age < 1.2) continue;
          const dx = r.kart.x - h.x, dz = r.kart.z - h.z;
          if (dx * dx + dz * dz < PROJ.barrelRadius * PROJ.barrelRadius) {
            this.hitRacer(r, h.owner, 'barrel');
            this.emit({ type: 'hazardGone', id: h.id });
            dead = true;
            break;
          }
        }
      }
      if (dead) this.hazards.splice(i, 1);
    }
  }

  stepProjectiles(dt) {
    const track = this.track;
    for (let i = this.projectiles.length - 1; i >= 0; i--) {
      const p = this.projectiles[i];
      p.age += dt; p.life -= dt;
      let dead = p.life <= 0;

      if (!dead && p.type === 'bolt') {
        p.x += p.dx * p.speed * dt;
        p.z += p.dz * p.speed * dt;
        p.trackS = track.closestS(p.x, p.z, p.trackS);
        const lat = track.lateralOffset(p.x, p.z, p.trackS);
        const maxLat = track.halfWidth + 1.2;
        if (Math.abs(lat) > maxLat) {
          // Reflect off the barrier.
          const c = track.sample(p.trackS);
          const nx = c.nx * Math.sign(lat), nz = c.nz * Math.sign(lat);
          const dot = p.dx * nx + p.dz * nz;
          p.dx -= 2 * dot * nx; p.dz -= 2 * dot * nz;
          const clamped = Math.sign(lat) * maxLat;
          p.x = c.x + c.nx * clamped; p.z = c.z + c.nz * clamped;
          if (--p.bounces < 0) dead = true;
          else this.emit({ type: 'bounce', x: p.x, z: p.z });
        }
      } else if (!dead && p.type === 'seeker') {
        const target = p.target ? this.getRacer(p.target) : null;
        if (target && !target.finished) {
          // Advance along the track, homing laterally near the target.
          p.s = (p.s + PROJ.seekerSpeed * dt) % track.length;
          const gap = wrapDelta(target.kart.trackS - p.s, track.length);
          const targetLat = track.lateralOffset(target.kart.x, target.kart.z, target.kart.trackS);
          const blend = Math.max(0, 1 - Math.abs(gap) / 40);
          p.lat += (targetLat * blend - p.lat) * Math.min(1, 3 * dt);
          const c = track.sample(p.s);
          p.x = c.x + c.nx * p.lat;
          p.z = c.z + c.nz * p.lat;
        } else {
          // No target (leader fired it): fly along the track and expire.
          p.s = (p.s + PROJ.seekerSpeed * dt) % track.length;
          const c = track.sample(p.s);
          p.x = c.x + c.nx * p.lat;
          p.z = c.z + c.nz * p.lat;
        }
      }

      // Hit detection against all racers.
      if (!dead) {
        for (const r of this.racers) {
          if (r.finished || r.kart.spinT > 0) continue;
          if (r.id === p.owner && p.age < 0.5) continue;
          const dx = r.kart.x - p.x, dz = r.kart.z - p.z;
          if (dx * dx + dz * dz < PROJ.hitRadius * PROJ.hitRadius) {
            this.hitRacer(r, p.owner, p.type);
            this.emit({ type: 'projGone', id: p.id, x: p.x, z: p.z });
            dead = true;
            break;
          }
        }
      }
      if (dead) this.projectiles.splice(i, 1);
    }
  }

  // ------------------------------------------------------------------ bots

  botInput(r, dt) {
    const b = r.bot, k = r.kart, track = this.track;

    // Wander between racing lines now and then.
    b.latTimer -= dt;
    if (b.latTimer <= 0) {
      b.latTimer = 2.5 + this.rng() * 3.5;
      b.targetLat = (this.rng() * 2 - 1) * 4.5;
    }

    const lookAhead = 13 + Math.max(0, k.speed) * 0.5;
    const c = track.sample(k.trackS + lookAhead);
    const tx = c.x + c.nx * b.targetLat * 0.6;
    const tz = c.z + c.nz * b.targetLat * 0.6;
    const desired = Math.atan2(tx - k.x, tz - k.z);
    let diff = desired - k.heading;
    while (diff > Math.PI) diff -= 2 * Math.PI;
    while (diff < -Math.PI) diff += 2 * Math.PI;

    const steer = Math.max(-1, Math.min(1, diff * 2.4));
    let throttle = 1;
    if (Math.abs(diff) > 0.75 && k.speed > 22) throttle = 0.35;

    // Rubberbanding: keep the pack near the humans so races stay exciting.
    const humans = this.racers.filter((o) => !o.isBot && !o.finished);
    if (humans.length) {
      const bestHuman = Math.max(...humans.map((h) => h.dist));
      const gap = r.dist - bestHuman;
      k.topMul = r.char.top * (gap < -120 ? 1.09 : gap > 160 ? 0.93 : b.skill * 0.12 + 0.9);
    } else {
      k.topMul = r.char.top;
    }

    // Sustained corners: drift for the mini-turbo (better bots only).
    const drift = b.skill > 0.92 && Math.abs(diff) > 0.3 && k.speed > 22;

    // Fire items after a short human-like delay.
    if (r.item) {
      b.itemTimer -= dt;
      if (b.itemTimer <= 0) {
        b.itemTimer = 1 + this.rng() * 2.5;
        this.useItem(r.id);
      }
    }
    return { throttle, steer, drift };
  }
}
