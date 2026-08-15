// DOM head-up display: rank, lap, timer, item slot with roulette, speed,
// minimap, center messages and the results table. All user-facing text is
// German.

import { ITEMS } from '../shared/config.js';

const ITEM_ICONS = { bolt: '⚡', seeker: '🎯', turbo: '🚀', barrel: '🛢️', shield: '🛡️' };
const ITEM_KEYS = Object.keys(ITEM_ICONS);

function fmtTime(t) {
  if (t == null) return '–:–';
  const m = Math.floor(t / 60), s = t - m * 60;
  return `${m}:${s.toFixed(1).padStart(4, '0')}`;
}

export class Hud {
  constructor(track) {
    this.track = track;
    this.el = {
      hud: document.getElementById('hud'),
      rank: document.getElementById('hud-rank'),
      lap: document.getElementById('hud-lap'),
      time: document.getElementById('hud-time'),
      item: document.getElementById('hud-item'),
      speed: document.getElementById('hud-speed'),
      center: document.getElementById('hud-center'),
      msg: document.getElementById('hud-msg'),
      wrongway: document.getElementById('hud-wrongway'),
      minimap: document.getElementById('minimap'),
      results: document.getElementById('results'),
      resultsBody: document.getElementById('results-body'),
      resultsTitle: document.getElementById('results-title'),
    };
    this.msgTimer = 0;
    this.centerTimer = 0;
    this.rouletteFlip = 0;

    // Precompute minimap projection of the track outline.
    const pts = track.points;
    let minX = 1e9, maxX = -1e9, minZ = 1e9, maxZ = -1e9;
    for (const p of pts) {
      minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
      minZ = Math.min(minZ, p.z); maxZ = Math.max(maxZ, p.z);
    }
    // True top view (looking down): world +X = map right, world +Z = map down.
    const pad = 26, size = 340;
    const scale = (size - pad * 2) / Math.max(maxX - minX, maxZ - minZ);
    this.mapProject = (x, z) => [
      pad + (x - minX) * scale + (size - pad * 2 - (maxX - minX) * scale) / 2,
      pad + (z - minZ) * scale + (size - pad * 2 - (maxZ - minZ) * scale) / 2,
    ];
  }

  show() { this.el.hud.classList.remove('hidden'); }
  hide() { this.el.hud.classList.add('hidden'); }

  message(text, seconds = 2.2) {
    this.el.msg.textContent = text;
    this.msgTimer = seconds;
  }

  centerText(text, seconds = 1) {
    this.el.center.textContent = text;
    this.centerTimer = seconds;
  }

  // state: { rank, total, lap, laps, time, item, rouletteT, speed, wrongWay,
  //          racers: [{x, z, isPlayer, color, finished}] }
  update(dt, state) {
    this.msgTimer -= dt;
    if (this.msgTimer <= 0) this.el.msg.textContent = '';
    this.centerTimer -= dt;
    if (this.centerTimer <= 0) this.el.center.textContent = '';

    this.el.rank.innerHTML = `${state.rank}<small>/${state.total}</small>`;
    this.el.lap.textContent = `Runde ${Math.min(state.lap, state.laps)}/${state.laps}`;
    this.el.time.textContent = fmtTime(state.time);
    this.el.speed.innerHTML = `${Math.round(Math.abs(state.speed) * 3.4)} <small>km/h</small>`;
    this.el.wrongway.classList.toggle('hidden', !state.wrongWay);

    if (state.rouletteT > 0) {
      this.rouletteFlip += dt * 14;
      this.el.item.textContent = ITEM_ICONS[ITEM_KEYS[Math.floor(this.rouletteFlip) % ITEM_KEYS.length]];
      this.el.item.style.borderColor = '#fff';
    } else if (state.item) {
      this.el.item.textContent = ITEM_ICONS[state.item] || '?';
      this.el.item.style.borderColor = 'var(--accent)';
      this.el.item.title = ITEMS[state.item]?.name || '';
    } else {
      this.el.item.textContent = '';
      this.el.item.style.borderColor = '#ffffff33';
    }

    this.drawMinimap(state.racers || []);
  }

  drawMinimap(racers) {
    const g = this.el.minimap.getContext('2d');
    g.clearRect(0, 0, 340, 340);
    g.strokeStyle = 'rgba(10,16,30,0.55)';
    g.lineWidth = 16;
    g.lineJoin = 'round';
    g.beginPath();
    const pts = this.track.points;
    for (let i = 0; i <= pts.length; i += 4) {
      const p = pts[i % pts.length];
      const [x, y] = this.mapProject(p.x, p.z);
      if (i === 0) g.moveTo(x, y); else g.lineTo(x, y);
    }
    g.closePath();
    g.stroke();
    g.strokeStyle = '#e8dcc8';
    g.lineWidth = 9;
    g.stroke();
    // start line
    const s0 = this.mapProject(this.track.sample(0).x, this.track.sample(0).z);
    g.fillStyle = '#fff';
    g.fillRect(s0[0] - 4, s0[1] - 4, 8, 8);

    for (const r of racers) {
      const [x, y] = this.mapProject(r.x, r.z);
      g.beginPath();
      g.arc(x, y, r.isPlayer ? 9 : 6, 0, Math.PI * 2);
      g.fillStyle = r.isPlayer ? '#ffd23e' : r.color;
      g.fill();
      g.lineWidth = 2;
      g.strokeStyle = '#101828';
      g.stroke();
    }
  }

  showResults(rows, playerId, isOnline) {
    const tb = this.el.resultsBody;
    tb.innerHTML = '';
    const medals = ['🥇', '🥈', '🥉'];
    rows.forEach((r, i) => {
      const tr = document.createElement('tr');
      if (r.id === playerId) tr.classList.add('me');
      const place = medals[i] || `${i + 1}.`;
      tr.innerHTML = `<td>${place}</td><td>${r.name}${r.isBot ? ' 🤖' : ''}</td><td>${r.finishTime != null ? fmtTime(r.finishTime) : 'DNF'}</td>`;
      tb.appendChild(tr);
    });
    const me = rows.findIndex((r) => r.id === playerId);
    this.el.resultsTitle.textContent =
      me === 0 ? '🏆 Sieg!' : me >= 0 && me < 3 ? 'Podium!' : 'Zieleinlauf';
    document.getElementById('results-status').textContent =
      isOnline ? 'Nächstes Rennen startet automatisch…' : '';
    this.el.results.classList.remove('hidden');
  }

  hideResults() { this.el.results.classList.add('hidden'); }
}
