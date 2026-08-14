// Game orchestration: Three.js scene, camera, render loop, and the two run
// modes – solo (local RaceSim with bots) and online (server-authoritative,
// local kart simulated client-side for zero input latency).

import * as THREE from 'three';
import { buildTrack } from '../shared/track.js';
import { RaceSim } from '../shared/racesim.js';
import { RACE, NET, PROJ, ITEMS, CHARACTERS, characterById } from '../shared/config.js';
import { makeKartState, stepKart } from '../shared/kartphysics.js';
import { buildWorld } from './world.js';
import { KartView } from './kartview.js';
import { Effects } from './effects.js';
import { Hud } from './hud.js';
import { Input } from './input.js';
import { SoundKit } from './audio.js';

const FIXED_DT = 1 / 60;

export class Game {
  constructor({ canvas, mode, playerName, charId, net, onRaceOver }) {
    this.mode = mode;               // 'solo' | 'online'
    this.playerName = playerName;
    this.charId = charId;
    this.net = net || null;
    this.onRaceOver = onRaceOver || (() => {});
    this.disposed = false;

    this.track = buildTrack();

    // --- Renderer / scene ---
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(66, 1, 0.5, 3000);
    this.resize = () => {
      this.renderer.setSize(window.innerWidth, window.innerHeight);
      this.camera.aspect = window.innerWidth / window.innerHeight;
      this.camera.updateProjectionMatrix();
    };
    this.resize();
    window.addEventListener('resize', this.resize);

    this.world = buildWorld(this.scene, this.track);
    this.effects = new Effects(this.scene);
    this.hud = new Hud(this.track);
    this.input = new Input();
    this.sound = new SoundKit();
    this.sound.init();

    this.views = new Map();      // id -> { view, disp } (disp = displayed kart state)
    this.projMeshes = new Map(); // projectile id -> mesh
    this.hazMeshes = new Map();  // hazard id -> mesh
    this.time = 0;
    this.stateAcc = 0;
    this.accumulator = 0;
    this.lastFrame = null;
    this.countdownShown = null;
    this.raceOverSent = false;
    this.serverPhase = 'countdown';
    this.serverPhaseT = 0;
    this.snapshotRacers = new Map();

    if (mode === 'solo') this.setupSolo();
    // online mode waits for setupOnline(msg) from main.js

    this.hud.show();
    this.rafId = requestAnimationFrame((t) => this.frame(t));
  }

  // ------------------------------------------------------------------ setup

  setupSolo() {
    this.playerId = 'player';
    this.sim = new RaceSim(this.track, { laps: RACE.laps });
    this.sim.addRacer({ id: 'player', name: this.playerName, charId: this.charId });
    const others = CHARACTERS.filter((c) => c.id !== this.charId);
    for (let i = 0; i < RACE.maxRacers - 1; i++) {
      const ch = others[i % others.length];
      this.sim.addRacer({ id: 'bot-' + i, name: ch.name, charId: ch.id, isBot: true });
    }
    for (const r of this.sim.racers) this.addView(r.id, r.char, r.kart, r.id === 'player', r.name);
    this.sim.start();
    this.sound.startEngine();
  }

  // msg = raceSetup from server
  setupOnline(msg) {
    this.playerId = this.net.id;
    this.laps = msg.laps;
    this.onlineRacers = new Map(); // id -> latest snapshot entry
    for (const r of msg.racers) {
      const char = characterById(r.charId);
      const kart = makeKartState(r.x, r.z, r.heading, char);
      kart.trackS = this.track.closestS(r.x, r.z, null);
      this.addView(r.id, char, kart, r.id === this.playerId, r.name);
      if (r.id === this.playerId) this.localKart = kart;
      this.onlineRacers.set(r.id, { name: r.name, isBot: r.isBot, char, rank: 1, lap: 1, fin: 0 });
    }
    this.myItem = null;
    this.myRoulette = false;
    this.sound.startEngine();
  }

  addView(id, char, kartState, isPlayer, label = null) {
    const view = new KartView(this.scene, char, { showName: !isPlayer, label });
    this.views.set(id, { view, disp: kartState, char, isPlayer });
  }

  // --------------------------------------------------------------- events

  handleEvent(ev) {
    const me = this.playerId;
    const viewOf = (id) => this.views.get(id);
    switch (ev.type) {
      case 'go':
        this.hud.centerText('LOS!', 1);
        this.sound.countdownBeep(true);
        break;
      case 'boxTaken': {
        const m = this.world.itemBoxMeshes[ev.boxId];
        if (m) {
          m.visible = false;
          this.effects.pickup(m.position.x, m.position.z);
        }
        if (ev.id === me) this.sound.pickup();
        break;
      }
      case 'boxRespawn': {
        const m = this.world.itemBoxMeshes[ev.boxId];
        if (m) m.visible = true;
        break;
      }
      case 'item':
        if (ev.id === me) {
          this.sound.itemReady();
          this.hud.message(ITEMS[ev.item]?.name + '!', 1.4);
        }
        break;
      case 'fire': {
        if (ev.id === me) {
          this.sound.fire();
          if (ev.item === 'turbo' && this.localKart) {
            this.localKart.boostT = Math.max(this.localKart.boostT, 1.6);
            this.sound.boost();
          }
          if (ev.item === 'shield' && this.localKart) {
            this.localKart.shieldT = PROJ.shieldTime;
          }
        }
        break;
      }
      case 'hit': {
        const v = viewOf(ev.id);
        if (v) this.effects.explosion(v.disp.x, v.disp.z);
        if (ev.id === me) {
          this.sound.explosion();
          this.sound.spin();
          this.hud.message('Getroffen: ' + (ITEMS[ev.item]?.name || 'Autsch!'), 1.6);
          if (this.localKart && this.localKart.spinT <= 0) {
            this.localKart.spinT = 1.5;
            this.localKart.speed *= 0.25;
          }
        } else if (ev.by === me) {
          this.hud.message('Treffer! 🎯', 1.4);
          this.sound.explosion();
        }
        break;
      }
      case 'shieldBreak': {
        const v = viewOf(ev.id);
        if (v) this.effects.shieldPop(v.disp.x, v.disp.z);
        if (ev.id === me) {
          this.sound.shieldPop();
          if (this.localKart) this.localKart.shieldT = 0;
          this.hud.message('Schild hat gehalten!', 1.4);
        }
        break;
      }
      case 'bounce':
        this.effects.spawn(6, { x: ev.x, z: ev.z, y: 0.6, color: { r: 0.5, g: 1, b: 0.5 }, vy: 2, spread: 3, life: 0.3 });
        break;
      case 'projGone':
        this.effects.explosion(ev.x, ev.z);
        break;
      case 'lap':
        if (ev.id === me) {
          this.sound.lap();
          const laps = this.laps || RACE.laps;
          this.hud.message(ev.lap >= laps ? '🏁 Letzte Runde!' : `Runde ${ev.lap}/${laps}`, 1.8);
        }
        break;
      case 'finish':
        if (ev.id === me) {
          const v = viewOf(me);
          if (v) this.effects.confetti(v.disp.x, v.disp.z);
          this.sound.fanfare(ev.place === 1);
          this.hud.centerText(ev.place <= 3 ? ['🥇', '🥈', '🥉'][ev.place - 1] : `${ev.place}.`, 3);
          this.finished = true;
        }
        break;
      case 'raceOver':
        this.finishRace();
        break;
    }
  }

  finishRace() {
    if (this.raceOverSent) return;
    this.raceOverSent = true;
    setTimeout(() => {
      if (this.disposed) return;
      let rows;
      if (this.mode === 'solo') {
        rows = [...this.sim.racers].sort((a, b) => a.rank - b.rank)
          .map((r) => ({ id: r.id, name: r.name, isBot: r.isBot, finishTime: r.finishTime }));
      } else {
        rows = [...this.snapshotRacers.entries()]
          .map(([id, s]) => ({ id, name: this.onlineRacers.get(id)?.name || id, isBot: !!this.onlineRacers.get(id)?.isBot, rank: s.rank, finishTime: s.ft }))
          .sort((a, b) => a.rank - b.rank);
      }
      this.hud.showResults(rows, this.playerId, this.mode === 'online');
      this.onRaceOver(rows);
    }, 1200);
  }

  // --------------------------------------------------------------- online

  applySnapshot(msg) {
    this.serverPhase = msg.phase;
    if (msg.pt !== undefined) this.serverPhaseT = msg.pt;
    this.serverTime = msg.time;
    for (const r of msg.racers) {
      this.snapshotRacers.set(r.id, r);
      if (r.id === this.playerId) {
        this.myItem = r.item;
        this.myRoulette = !!r.roul;
        this.myRank = r.rank;
        this.myLap = r.lap;
        if (this.localKart) this.localKart.shieldT = r.shield;
      }
    }
    // Sync projectiles / hazards / boxes.
    this.syncPool(this.projMeshes, msg.proj, (p) => this.makeProjMesh(p.type));
    this.syncPool(this.hazMeshes, msg.haz, () => this.makeBarrelMesh());
    msg.boxes.forEach((active, i) => {
      const m = this.world.itemBoxMeshes[i];
      if (m) m.visible = !!active;
    });
    for (const ev of msg.events || []) this.handleEvent(ev);
  }

  syncPool(pool, list, factory) {
    const seen = new Set();
    for (const item of list) {
      seen.add(item.id);
      let m = pool.get(item.id);
      if (!m) {
        m = factory(item);
        this.scene.add(m);
        pool.set(item.id, m);
        m.position.set(item.x, m.userData.y ?? 0.8, item.z);
      }
      m.userData.tx = item.x;
      m.userData.tz = item.z;
    }
    for (const [id, m] of pool) {
      if (!seen.has(id)) { this.scene.remove(m); pool.delete(id); }
    }
  }

  makeProjMesh(type) {
    let m;
    if (type === 'seeker') {
      m = new THREE.Mesh(
        new THREE.SphereGeometry(0.75, 12, 10),
        new THREE.MeshStandardMaterial({ color: 0xd82c2c, emissive: 0x701010, metalness: 0.4, roughness: 0.3 }),
      );
      m.userData.y = 1.0;
    } else {
      m = new THREE.Mesh(
        new THREE.SphereGeometry(0.65, 12, 10),
        new THREE.MeshStandardMaterial({ color: 0x35d83c, emissive: 0x0c5010, metalness: 0.4, roughness: 0.3 }),
      );
      m.userData.y = 0.8;
    }
    m.castShadow = true;
    return m;
  }

  makeBarrelMesh() {
    const g = new THREE.Group();
    const body = new THREE.Mesh(
      new THREE.CylinderGeometry(1.0, 1.0, 1.6, 12),
      new THREE.MeshStandardMaterial({ color: 0x6b4a2a, roughness: 0.7 }),
    );
    body.position.y = 0.8;
    body.castShadow = true;
    const ring = new THREE.Mesh(
      new THREE.CylinderGeometry(1.04, 1.04, 0.22, 12),
      new THREE.MeshStandardMaterial({ color: 0x33261a, roughness: 0.6 }),
    );
    ring.position.y = 0.8;
    g.add(body, ring);
    g.userData.y = 0;
    return g;
  }

  // ---------------------------------------------------------------- frame

  frame(tMs) {
    if (this.disposed) return;
    this.rafId = requestAnimationFrame((t) => this.frame(t));
    if (this.lastFrame === null) this.lastFrame = tMs;
    let dt = Math.min(0.1, (tMs - this.lastFrame) / 1000);
    this.lastFrame = tMs;
    this.time += dt;

    this.accumulator += dt;
    while (this.accumulator >= FIXED_DT) {
      this.fixedStep(FIXED_DT);
      this.accumulator -= FIXED_DT;
    }

    this.updateViews(dt);
    this.updateCamera(dt);
    this.world.update(dt, this.time);
    this.effects.update(dt);
    this.updateHud(dt);
    this.renderer.render(this.scene, this.camera);
  }

  fixedStep(dt) {
    const input = this.input.read();

    if (this.mode === 'solo') {
      if (input.fire) this.sim.useItem('player');
      this.sim.step(dt, { player: { throttle: input.throttle, steer: input.steer, drift: input.drift } });
      for (const ev of this.sim.drainEvents()) this.handleEvent(ev);
      const player = this.sim.getRacer('player');
      this.localKart = player.kart;
    } else if (this.localKart) {
      // Online: local kart physics runs here; server does items/laps/hits.
      const racing = this.serverPhase === 'racing' && !this.finished;
      const inp = racing
        ? { throttle: input.throttle, steer: input.steer, drift: input.drift }
        : { throttle: 0, steer: 0, drift: false };
      const wasDriftCharge = this.localKart.driftCharge;
      stepKart(this.localKart, inp, dt, this.track);
      if (wasDriftCharge > 0 && this.localKart.driftCharge === 0 && this.localKart.boostT > 0) this.sound.boost();
      if (input.fire && racing) this.net.send({ type: 'useItem' });

      this.stateAcc += dt;
      if (this.stateAcc >= 1 / NET.stateHz) {
        this.stateAcc = 0;
        const k = this.localKart;
        this.net.send({
          type: 'state',
          x: Math.round(k.x * 100) / 100,
          z: Math.round(k.z * 100) / 100,
          heading: Math.round(k.heading * 1000) / 1000,
          speed: Math.round(k.speed * 10) / 10,
          drifting: k.drifting,
          boost: k.boostT > 0,
        });
      }
    }

    // Drift feedback for the local kart (both modes).
    const k = this.localKart;
    if (k) {
      if (k.drifting) {
        const level = k.driftCharge >= 2 ? 2 : k.driftCharge >= 0.9 ? 1 : 0;
        this.effects.driftSparks(k.x, k.z, k.heading, k.driftDir, level);
      }
      if (k.boostT > 0) this.effects.boostTrail(k.x, k.z, k.heading);
      if (k.offroad && Math.abs(k.speed) > 6) this.effects.dust(k.x, k.z);
    }
  }

  updateViews(dt) {
    for (const [id, v] of this.views) {
      let state;
      if (this.mode === 'solo') {
        state = this.sim.getRacer(id).kart;
      } else if (id === this.playerId) {
        state = this.localKart;
      } else {
        // Remote kart: ease displayed state toward the latest snapshot.
        const snap = this.snapshotRacers.get(id);
        const d = v.disp;
        if (snap) {
          const a = Math.min(1, 12 * dt);
          d.x += (snap.x - d.x) * a;
          d.z += (snap.z - d.z) * a;
          let dh = snap.h - d.heading;
          while (dh > Math.PI) dh -= 2 * Math.PI;
          while (dh < -Math.PI) dh += 2 * Math.PI;
          d.heading += dh * a;
          d.speed = snap.v;
          d.drifting = !!snap.drift;
          d.driftDir = d.drifting ? Math.sign(dh) || 1 : 0;
          d.boostT = snap.boost ? 0.2 : 0;
          d.shieldT = snap.shield;
          if (snap.spin > 0) { d.spinT = snap.spin; d.spinPhase += dt * 10; } else { d.spinT = 0; d.spinPhase = 0; }
        }
        state = d;
      }
      if (state) {
        v.disp = state === v.disp ? v.disp : state;
        v.view.update(dt, state, this.time);
      }
    }

    // Ease projectile/hazard meshes toward their network targets.
    for (const pool of [this.projMeshes, this.hazMeshes]) {
      for (const m of pool.values()) {
        if (m.userData.tx !== undefined) {
          const a = Math.min(1, 14 * dt);
          m.position.x += (m.userData.tx - m.position.x) * a;
          m.position.z += (m.userData.tz - m.position.z) * a;
          m.rotation.y += dt * 6;
        }
      }
    }

    // Solo mode: projectiles/hazards/boxes read straight from the sim.
    if (this.mode === 'solo') {
      this.syncPool(this.projMeshes, this.sim.projectiles, (p) => this.makeProjMesh(p.type));
      this.syncPool(this.hazMeshes, this.sim.hazards, () => this.makeBarrelMesh());
      for (const box of this.sim.boxes) {
        const m = this.world.itemBoxMeshes[box.id];
        if (m) m.visible = box.active;
      }
    }
  }

  updateCamera(dt) {
    const k = this.localKart;
    if (!k) return;
    const dirX = Math.sin(k.heading), dirZ = Math.cos(k.heading);
    const speedFrac = Math.min(1, Math.abs(k.speed) / 40);
    const back = 9.5 + speedFrac * 2.5;
    const target = new THREE.Vector3(k.x - dirX * back, 4.6 + speedFrac, k.z - dirZ * back);
    const a = 1 - Math.pow(0.001, dt);
    this.camera.position.lerp(target, a);
    this.camera.lookAt(k.x + dirX * 7, 1.6, k.z + dirZ * 7);
    const wantFov = 66 + (k.boostT > 0 ? 12 : speedFrac * 5);
    this.camera.fov += (wantFov - this.camera.fov) * Math.min(1, 6 * dt);
    this.camera.updateProjectionMatrix();
  }

  updateHud(dt) {
    const k = this.localKart;
    if (!k) return;

    // Countdown display.
    let phase, phaseT;
    if (this.mode === 'solo') { phase = this.sim.phase; phaseT = this.sim.phaseT; }
    else { phase = this.serverPhase; phaseT = this.serverPhaseT += dt; }
    if (phase === 'countdown') {
      const n = Math.ceil(RACE.countdown - phaseT - 0.2);
      if (n !== this.countdownShown && n > 0) {
        this.countdownShown = n;
        this.hud.centerText(String(n), 1);
        this.sound.countdownBeep(false);
      }
    }

    // Wrong-way check.
    let wrongWay = false;
    if (Math.abs(k.speed) > 6) {
      const p = this.track.sample(k.trackS);
      const dot = Math.sin(k.heading) * p.tx + Math.cos(k.heading) * p.tz;
      wrongWay = dot * Math.sign(k.speed) < -0.4;
    }

    let state;
    if (this.mode === 'solo') {
      const r = this.sim.getRacer('player');
      state = {
        rank: r.rank, total: this.sim.racers.length,
        lap: r.lap, laps: this.sim.laps,
        time: this.sim.time, item: r.item, rouletteT: r.rouletteT,
        speed: k.speed, wrongWay,
        racers: this.sim.racers.map((o) => ({
          x: o.kart.x, z: o.kart.z, isPlayer: o.id === 'player',
          color: '#' + o.char.color.toString(16).padStart(6, '0'),
        })),
      };
    } else {
      state = {
        rank: this.myRank || 1, total: this.snapshotRacers.size || 1,
        lap: this.myLap || 1, laps: this.laps || RACE.laps,
        time: this.serverTime, item: this.myItem, rouletteT: this.myRoulette ? 1 : 0,
        speed: k.speed, wrongWay,
        racers: [...this.snapshotRacers.entries()].map(([id, s]) => ({
          x: id === this.playerId ? k.x : s.x,
          z: id === this.playerId ? k.z : s.z,
          isPlayer: id === this.playerId,
          color: '#' + (this.onlineRacers.get(id)?.char.color || 0xffffff).toString(16).padStart(6, '0'),
        })),
      };
    }
    this.hud.update(dt, state);
    this.sound.setEngine(Math.min(1, Math.abs(k.speed) / 45), k.boostT > 0);
  }

  dispose() {
    this.disposed = true;
    cancelAnimationFrame(this.rafId);
    window.removeEventListener('resize', this.resize);
    this.sound.stopEngine();
    this.hud.hide();
    this.hud.hideResults();
    this.renderer.dispose();
  }
}
