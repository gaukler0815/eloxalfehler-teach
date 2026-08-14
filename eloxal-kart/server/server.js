// Eloxal Kart game server.
//
// One process does both jobs:
//   1. Serves the client (public/) over HTTP.
//   2. Runs the authoritative race simulation and talks to the players
//      over WebSocket (same port).
//
// Authority model (prototype level, documented in README):
//   - Each client simulates its own kart and reports position/heading/speed.
//   - The server is authoritative for everything competitive: race phases,
//     laps/ranks, item boxes, item assignment, projectiles, hits and bots.
//
// Start with: npm start   (default port 8420, override with PORT env var)

import http from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { join, extname, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';
import { WebSocketServer } from 'ws';

import { buildTrack } from '../public/shared/track.js';
import { RaceSim } from '../public/shared/racesim.js';
import { RACE, NET, PHYS, CHARACTERS } from '../public/shared/config.js';

const PORT = process.env.PORT || 8420;
const PUBLIC_DIR = join(fileURLToPath(new URL('.', import.meta.url)), '..', 'public');

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
};

const httpServer = http.createServer(async (req, res) => {
  try {
    if (req.url === '/health') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ok: true, players: clients.size, phase: room.sim?.phase || 'lobby' }));
      return;
    }
    let path = decodeURIComponent((req.url || '/').split('?')[0]);
    if (path === '/') path = '/index.html';
    const file = normalize(join(PUBLIC_DIR, path));
    if (!file.startsWith(PUBLIC_DIR)) { res.writeHead(403); res.end(); return; }
    const st = await stat(file).catch(() => null);
    if (!st || !st.isFile()) { res.writeHead(404); res.end('Nicht gefunden'); return; }
    res.writeHead(200, { 'Content-Type': MIME[extname(file)] || 'application/octet-stream' });
    res.end(await readFile(file));
  } catch (err) {
    res.writeHead(500);
    res.end('Serverfehler');
  }
});

// ---------------------------------------------------------------- game room

const track = buildTrack();
const clients = new Map(); // ws -> { id, name, charId, racing, lastState }
let nextClientId = 1;

const room = {
  sim: null,
  lobbyCountdown: null, // seconds until auto start once someone is waiting
  resultsT: 0,
};

function send(ws, msg) {
  if (ws.readyState === ws.OPEN) ws.send(JSON.stringify(msg));
}
function broadcast(msg) {
  const data = JSON.stringify(msg);
  for (const ws of clients.keys()) if (ws.readyState === ws.OPEN) ws.send(data);
}

function lobbyInfo() {
  return {
    type: 'lobby',
    players: [...clients.values()].map((c) => ({ id: c.id, name: c.name, charId: c.charId, racing: c.racing })),
    countdown: room.lobbyCountdown,
    racing: !!room.sim && room.sim.phase !== 'finished',
  };
}

function startRace() {
  room.sim = new RaceSim(track, { laps: RACE.laps });
  room.lobbyCountdown = null;

  const humans = [...clients.values()].filter((c) => c.name);
  const usedChars = new Set();
  for (const c of humans) {
    c.racing = true;
    usedChars.add(c.charId);
    room.sim.addRacer({ id: c.id, name: c.name, charId: c.charId, external: true });
  }
  // Fill the grid with bots on the remaining characters.
  const free = CHARACTERS.filter((ch) => !usedChars.has(ch.id));
  let botN = 0;
  while (room.sim.racers.length < RACE.maxRacers && botN < free.length) {
    const ch = free[botN++];
    room.sim.addRacer({ id: 'bot-' + ch.id, name: ch.name, charId: ch.id, isBot: true });
  }

  broadcast({
    type: 'raceSetup',
    laps: RACE.laps,
    racers: room.sim.racers.map((r) => ({
      id: r.id, name: r.name, charId: r.char.id, isBot: r.isBot,
      x: r.kart.x, z: r.kart.z, heading: r.kart.heading,
    })),
  });
  room.sim.start();
}

// ---------------------------------------------------------------- websocket

const wss = new WebSocketServer({ server: httpServer });

wss.on('connection', (ws) => {
  const client = { id: 'p' + nextClientId++, name: null, charId: 'al', racing: false, lastState: null };
  clients.set(ws, client);
  send(ws, { type: 'welcome', id: client.id, laps: RACE.laps });

  ws.on('message', (raw) => {
    let msg;
    try { msg = JSON.parse(raw); } catch { return; }
    if (typeof msg !== 'object' || !msg) return;

    if (msg.type === 'join') {
      client.name = String(msg.name || 'Fahrer').slice(0, 16);
      client.charId = CHARACTERS.some((c) => c.id === msg.charId) ? msg.charId : 'al';
      // Join running race is not allowed – wait for the next one.
      if (room.sim && room.sim.phase !== 'finished') {
        send(ws, { type: 'wait', reason: 'Rennen läuft – du startest beim nächsten.' });
      } else if (room.lobbyCountdown === null) {
        room.lobbyCountdown = 6; // first joiner arms the start timer
      }
      broadcast(lobbyInfo());
    } else if (msg.type === 'state' && client.racing && room.sim) {
      const r = room.sim.getRacer(client.id);
      if (!r || r.finished) return;
      // Basic sanity clamp – prototype-level cheat protection.
      const maxStep = PHYS.maxSpeed * PHYS.boostSpeedFactor * 0.4;
      const k = r.kart;
      const nx = Number(msg.x), nz = Number(msg.z);
      if (!Number.isFinite(nx) || !Number.isFinite(nz)) return;
      if (client.lastState && Math.hypot(nx - k.x, nz - k.z) > maxStep * 3) {
        // Ignore absurd jumps; keep last known position.
        return;
      }
      k.x = nx; k.z = nz;
      k.heading = Number(msg.heading) || 0;
      k.speed = Math.max(-PHYS.maxReverse, Math.min(PHYS.maxSpeed * PHYS.boostSpeedFactor + 5, Number(msg.speed) || 0));
      k.drifting = !!msg.drifting;
      k.boostT = msg.boost ? 0.2 : 0;
      client.lastState = Date.now();
    } else if (msg.type === 'useItem' && client.racing && room.sim) {
      room.sim.useItem(client.id);
    } else if (msg.type === 'ping') {
      send(ws, { type: 'pong', t: msg.t });
    }
  });

  ws.on('close', () => {
    clients.delete(ws);
    if (room.sim) {
      const r = room.sim.getRacer(client.id);
      if (r && !r.finished) { r.finished = true; r.finishTime = null; room.sim.finishedCount++; }
    }
    broadcast(lobbyInfo());
  });
});

// ---------------------------------------------------------------- main loop

const SIM_DT = 1 / NET.simHz;
let snapshotAcc = 0;

setInterval(() => {
  // Lobby countdown → race start.
  if (!room.sim || room.sim.phase === 'finished') {
    const waiting = [...clients.values()].some((c) => c.name);
    if (room.sim && room.sim.phase === 'finished') {
      room.resultsT += SIM_DT;
      if (room.resultsT > RACE.resultsTime) {
        room.sim = null;
        room.resultsT = 0;
        for (const c of clients.values()) c.racing = false;
        room.lobbyCountdown = waiting ? 6 : null;
        broadcast(lobbyInfo());
      }
    } else if (waiting && room.lobbyCountdown !== null) {
      room.lobbyCountdown -= SIM_DT;
      if (room.lobbyCountdown <= 0) startRace();
    }
    if (room.sim) room.sim.step(SIM_DT); // let finished sim idle (results)
    return;
  }

  room.sim.step(SIM_DT);

  const events = room.sim.drainEvents();
  snapshotAcc += SIM_DT;
  if (snapshotAcc >= 1 / NET.snapshotHz || events.length) {
    snapshotAcc = 0;
    broadcast({
      type: 'snapshot',
      phase: room.sim.phase,
      time: Math.round(room.sim.time * 100) / 100,
      events,
      racers: room.sim.racers.map((r) => ({
        id: r.id,
        x: Math.round(r.kart.x * 100) / 100,
        z: Math.round(r.kart.z * 100) / 100,
        h: Math.round(r.kart.heading * 1000) / 1000,
        v: Math.round(r.kart.speed * 10) / 10,
        drift: r.kart.drifting ? 1 : 0,
        boost: r.kart.boostT > 0 ? 1 : 0,
        spin: Math.round(r.kart.spinT * 100) / 100,
        shield: Math.round(r.kart.shieldT * 10) / 10,
        item: r.item,
        roul: r.rouletteT > 0 ? 1 : 0,
        lap: r.lap,
        rank: r.rank,
        fin: r.finished ? 1 : 0,
        ft: r.finishTime,
      })),
      proj: room.sim.projectiles.map((p) => ({
        id: p.id, type: p.type,
        x: Math.round(p.x * 10) / 10, z: Math.round(p.z * 10) / 10,
      })),
      haz: room.sim.hazards.map((h) => ({ id: h.id, x: h.x, z: h.z })),
      boxes: room.sim.boxes.map((b) => (b.active ? 1 : 0)),
    });
  }
}, SIM_DT * 1000);

httpServer.listen(PORT, () => {
  console.log(`Eloxal Kart Server läuft: http://localhost:${PORT}`);
});
