/* Eloxal Strike — multiplayer lobby/relay server.
 *
 * One file, zero dependencies: plain Node.js (v14+). It serves the game
 * itself over HTTP (the eloxal-strike folder one level up) and speaks
 * WebSocket on the same port for the lobby and match relay.
 *
 *   node server.js            → http://<host>:8081  (game + multiplayer)
 *   node server.js --port 9000
 *
 * Rooms live in memory. The server never simulates gameplay — it keeps the
 * room list, relays state/shot/hit messages between the players of a room
 * and tracks the kill score of a deathmatch.
 */
'use strict';

const http = require('http');
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const PORT = (() => {
  const i = process.argv.indexOf('--port');
  if (i >= 0 && process.argv[i + 1]) { return parseInt(process.argv[i + 1], 10); }
  return parseInt(process.env.PORT || '8081', 10);
})();
const KILL_LIMIT = 15;
const GAME_ROOT = path.join(__dirname, '..');

// ---------------------------------------------------------------------------
// minimal RFC6455 WebSocket implementation (text frames, ping/pong, close)
// ---------------------------------------------------------------------------
const WS_MAGIC = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11';

function wsAccept(key) {
  return crypto.createHash('sha1').update(key + WS_MAGIC).digest('base64');
}

function encodeFrame(str) {
  const payload = Buffer.from(str, 'utf8');
  const len = payload.length;
  let header;
  if (len < 126) {
    header = Buffer.from([0x81, len]);
  } else if (len < 65536) {
    header = Buffer.alloc(4);
    header[0] = 0x81; header[1] = 126;
    header.writeUInt16BE(len, 2);
  } else {
    header = Buffer.alloc(10);
    header[0] = 0x81; header[1] = 127;
    header.writeBigUInt64BE(BigInt(len), 2);
  }
  return Buffer.concat([header, payload]);
}

function encodeControl(opcode, payload) {
  payload = payload || Buffer.alloc(0);
  return Buffer.concat([Buffer.from([0x80 | opcode, payload.length]), payload]);
}

/* Wraps a net.Socket after the upgrade handshake. Emits whole text
 * messages via onMessage, handles ping/pong/close internally. */
class WsConn {
  constructor(socket) {
    this.socket = socket;
    this.buffer = Buffer.alloc(0);
    this.alive = true;
    this.onMessage = null;
    this.onClose = null;
    socket.on('data', (chunk) => this._feed(chunk));
    const bye = () => this._closed();
    socket.on('close', bye);
    socket.on('error', bye);
    socket.on('end', bye);
  }

  _closed() {
    if (!this.alive) { return; }
    this.alive = false;
    if (this.onClose) { this.onClose(); }
  }

  _feed(chunk) {
    this.buffer = Buffer.concat([this.buffer, chunk]);
    while (true) {
      const frame = this._readFrame();
      if (!frame) { break; }
      if (frame.opcode === 0x8) {                 // close
        try { this.socket.end(encodeControl(0x8)); } catch (e) { /* gone */ }
        this._closed();
        return;
      } else if (frame.opcode === 0x9) {          // ping → pong
        this._raw(encodeControl(0xA, frame.payload));
      } else if (frame.opcode === 0x1 && this.onMessage) {
        this.onMessage(frame.payload.toString('utf8'));
      }
    }
  }

  _readFrame() {
    const buf = this.buffer;
    if (buf.length < 2) { return null; }
    const opcode = buf[0] & 0x0f;
    const masked = (buf[1] & 0x80) !== 0;
    let len = buf[1] & 0x7f;
    let off = 2;
    if (len === 126) {
      if (buf.length < 4) { return null; }
      len = buf.readUInt16BE(2);
      off = 4;
    } else if (len === 127) {
      if (buf.length < 10) { return null; }
      const big = buf.readBigUInt64BE(2);
      if (big > 1048576n) { this.socket.destroy(); return null; }
      len = Number(big);
      off = 10;
    }
    if (len > 1048576) { this.socket.destroy(); return null; }
    const maskLen = masked ? 4 : 0;
    if (buf.length < off + maskLen + len) { return null; }
    let payload = buf.slice(off + maskLen, off + maskLen + len);
    if (masked) {
      const mask = buf.slice(off, off + 4);
      payload = Buffer.from(payload);
      for (let i = 0; i < payload.length; i++) { payload[i] ^= mask[i % 4]; }
    }
    this.buffer = buf.slice(off + maskLen + len);
    return { opcode, payload };
  }

  _raw(frame) {
    if (!this.alive || this.socket.destroyed) { return; }
    try { this.socket.write(frame); } catch (e) { this._closed(); }
  }

  send(obj) { this._raw(encodeFrame(JSON.stringify(obj))); }
  ping() { this._raw(encodeControl(0x9)); }
}

// ---------------------------------------------------------------------------
// static file serving (the game itself)
// ---------------------------------------------------------------------------
const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.md': 'text/plain; charset=utf-8'
};

function serveStatic(req, res) {
  let urlPath = decodeURIComponent(req.url.split('?')[0]);
  if (urlPath === '/') { urlPath = '/index.html'; }
  const filePath = path.normalize(path.join(GAME_ROOT, urlPath));
  if (!filePath.startsWith(GAME_ROOT)) {
    res.writeHead(403); res.end('Verboten'); return;
  }
  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end('Nicht gefunden');
      return;
    }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(filePath)] || 'application/octet-stream' });
    res.end(data);
  });
}

// ---------------------------------------------------------------------------
// lobby & rooms
// ---------------------------------------------------------------------------
let nextClientId = 1;
let nextRoomId = 1;
const clients = new Map();   // id → {id, conn, name, colorIdx, roomId}
const rooms = new Map();     // id → {id, name, hostId, started, players:Set, scores: Map id→{kills,deaths}}

function roomSummary(room) {
  return {
    id: room.id, name: room.name,
    host: (clients.get(room.hostId) || {}).name || '?',
    players: room.players.size, started: room.started
  };
}

function roomDetail(room) {
  return {
    id: room.id, name: room.name, hostId: room.hostId, started: room.started,
    players: [...room.players].map((pid) => {
      const c = clients.get(pid);
      const sc = room.scores.get(pid) || { kills: 0, deaths: 0 };
      return { id: pid, name: c ? c.name : '?', colorIdx: c ? c.colorIdx : 0,
        kills: sc.kills, deaths: sc.deaths };
    })
  };
}

function broadcastLobby() {
  const list = { t: 'rooms', rooms: [...rooms.values()].map(roomSummary) };
  for (const c of clients.values()) {
    if (!c.roomId) { c.conn.send(list); }
  }
}

function broadcastRoom(room, msg) {
  for (const pid of room.players) {
    const c = clients.get(pid);
    if (c) { c.conn.send(msg); }
  }
}

function pushRoomState(room) {
  broadcastRoom(room, { t: 'room', room: roomDetail(room) });
}

function leaveRoom(client) {
  const room = rooms.get(client.roomId);
  client.roomId = null;
  if (!room) { return; }
  room.players.delete(client.id);
  room.scores.delete(client.id);
  broadcastRoom(room, { t: 'peerleft', id: client.id, name: client.name });
  if (room.players.size === 0) {
    rooms.delete(room.id);
  } else {
    if (room.hostId === client.id) {
      room.hostId = [...room.players][0];
    }
    pushRoomState(room);
  }
  broadcastLobby();
  client.conn.send({ t: 'rooms', rooms: [...rooms.values()].map(roomSummary) });
}

function handleMessage(client, raw) {
  let msg;
  try { msg = JSON.parse(raw); } catch (e) { return; }
  const room = client.roomId ? rooms.get(client.roomId) : null;

  switch (msg.t) {
    case 'hello': {
      client.name = String(msg.name || 'SPIELER').slice(0, 12).toUpperCase();
      client.conn.send({ t: 'welcome', id: client.id, name: client.name });
      client.conn.send({ t: 'rooms', rooms: [...rooms.values()].map(roomSummary) });
      break;
    }
    case 'list': {
      client.conn.send({ t: 'rooms', rooms: [...rooms.values()].map(roomSummary) });
      break;
    }
    case 'create': {
      if (room) { leaveRoom(client); }
      const r = {
        id: nextRoomId++,
        name: String(msg.name || client.name + 'S HALLE').slice(0, 24).toUpperCase(),
        hostId: client.id, started: false,
        players: new Set([client.id]), scores: new Map([[client.id, { kills: 0, deaths: 0 }]])
      };
      rooms.set(r.id, r);
      client.roomId = r.id;
      pushRoomState(r);
      broadcastLobby();
      break;
    }
    case 'join': {
      const r = rooms.get(msg.id);
      if (!r) { client.conn.send({ t: 'error', msg: 'Raum existiert nicht mehr.' }); break; }
      if (r.players.size >= 8) { client.conn.send({ t: 'error', msg: 'Raum ist voll (max. 8).' }); break; }
      if (room) { leaveRoom(client); }
      r.players.add(client.id);
      r.scores.set(client.id, { kills: 0, deaths: 0 });
      client.roomId = r.id;
      pushRoomState(r);
      broadcastLobby();
      break;
    }
    case 'leave': {
      if (room) { leaveRoom(client); }
      break;
    }
    case 'start': {
      if (room && room.hostId === client.id && !room.started) {
        room.started = true;
        for (const s of room.scores.values()) { s.kills = 0; s.deaths = 0; }
        broadcastRoom(room, { t: 'started' });
        pushRoomState(room);
        broadcastLobby();
      }
      break;
    }
    case 'state':
    case 'shot': {
      if (!room || !room.started) { break; }
      msg.from = client.id;
      for (const pid of room.players) {
        if (pid !== client.id) {
          const c = clients.get(pid);
          if (c) { c.conn.send(msg); }
        }
      }
      break;
    }
    case 'hit': {
      if (!room || !room.started) { break; }
      const target = clients.get(msg.target);
      if (target && target.roomId === room.id) {
        target.conn.send({ t: 'hit', from: client.id, dmg: Number(msg.dmg) || 0, head: !!msg.head });
      }
      break;
    }
    case 'died': {
      if (!room || !room.started) { break; }
      const me = room.scores.get(client.id);
      if (me) { me.deaths++; }
      const killerScore = room.scores.get(msg.killer);
      if (killerScore && msg.killer !== client.id) { killerScore.kills++; }
      broadcastRoom(room, {
        t: 'died', from: client.id,
        killer: msg.killer, killerName: (clients.get(msg.killer) || {}).name || '?'
      });
      pushRoomState(room);
      if (killerScore && killerScore.kills >= KILL_LIMIT) {
        room.started = false;
        broadcastRoom(room, {
          t: 'ended', winner: msg.killer,
          winnerName: (clients.get(msg.killer) || {}).name || '?'
        });
        pushRoomState(room);
        broadcastLobby();
      }
      break;
    }
    default: break;
  }
}

// ---------------------------------------------------------------------------
// wiring
// ---------------------------------------------------------------------------
const server = http.createServer(serveStatic);

server.on('upgrade', (req, socket) => {
  const key = req.headers['sec-websocket-key'];
  if (!key || (req.headers.upgrade || '').toLowerCase() !== 'websocket') {
    socket.destroy();
    return;
  }
  socket.write(
    'HTTP/1.1 101 Switching Protocols\r\n' +
    'Upgrade: websocket\r\n' +
    'Connection: Upgrade\r\n' +
    'Sec-WebSocket-Accept: ' + wsAccept(key) + '\r\n\r\n'
  );
  socket.setNoDelay(true);

  const conn = new WsConn(socket);
  const client = {
    id: nextClientId++, conn, name: 'SPIELER',
    colorIdx: (nextClientId - 2) % 6, roomId: null
  };
  clients.set(client.id, client);
  conn.onMessage = (text) => handleMessage(client, text);
  conn.onClose = () => {
    leaveRoom(client);
    clients.delete(client.id);
  };
});

setInterval(() => {
  for (const c of clients.values()) { c.conn.ping(); }
}, 30000);

server.listen(PORT, () => {
  console.log('Eloxal Strike Server läuft:');
  console.log('  Spiel:       http://localhost:' + PORT + '/');
  console.log('  Multiplayer: ws://localhost:' + PORT + '/');
  console.log('Im Firmennetz erreichbar unter der IP dieses Rechners, Port ' + PORT + '.');
});
