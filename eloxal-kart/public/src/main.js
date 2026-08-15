// Entry point: main menu, character select, solo start and the online
// lobby flow. Owns the Net connection and creates/disposes Game instances.

import { CHARACTERS } from '../shared/config.js';
import { Game } from './game.js';
import { Net } from './net.js';
import { LOGO_SVG } from './branding.js';

// Jacobi Eloxal logo in every panel.
for (const el of document.querySelectorAll('.brand')) el.innerHTML = LOGO_SVG;

const canvas = document.getElementById('game-canvas');
const menu = document.getElementById('menu');
const lobby = document.getElementById('lobby');
const nameInput = document.getElementById('name');
const netStatus = document.getElementById('netstatus');

let selectedChar = CHARACTERS[0].id;
let game = null;
let net = null;

// ------------------------------------------------------------ menu setup

// localStorage can throw in sandboxed embeds – never let that kill the menu.
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* ignore */ } },
};

nameInput.value = store.get('ek-name') || '';

// Phones: touch controls appear in game; landscape plays much better.
if ('ontouchstart' in window) {
  const hint = document.createElement('div');
  hint.className = 'netstatus';
  hint.textContent = '📱 Tipp: Handy quer halten – gefahren wird mit den Bildschirm-Tasten.';
  document.querySelector('#menu .panel')?.appendChild(hint);
}

const charsEl = document.getElementById('chars');
for (const c of CHARACTERS) {
  const div = document.createElement('div');
  div.className = 'char' + (c.id === selectedChar ? ' sel' : '');
  div.dataset.id = c.id;
  div.innerHTML = `<div class="swatch" style="background:#${c.color.toString(16).padStart(6, '0')}"></div><div class="cname">${c.name}</div>`;
  div.addEventListener('click', () => {
    selectedChar = c.id;
    charsEl.querySelectorAll('.char').forEach((el) => el.classList.toggle('sel', el.dataset.id === c.id));
  });
  charsEl.appendChild(div);
}

function playerName() {
  const n = nameInput.value.trim() || 'Fahrer';
  store.set('ek-name', n);
  return n;
}

function endGame() {
  if (game) { game.dispose(); game = null; }
}

// ------------------------------------------------------------- solo mode

function startSolo() {
  endGame();
  menu.classList.add('hidden');
  lobby.classList.add('hidden');
  game = new Game({
    canvas, mode: 'solo', playerName: playerName(), charId: selectedChar,
    onRaceOver: () => {},
  });
}

document.getElementById('btn-solo').addEventListener('click', startSolo);

// ----------------------------------------------------------- online mode

async function startOnline() {
  const btn = document.getElementById('btn-online');
  btn.disabled = true;
  netStatus.textContent = 'Verbinde mit Server…';
  net = new Net();
  try {
    await net.connect();
  } catch {
    netStatus.textContent = 'Kein Server erreichbar – auf dieser Seite läuft nur der Solo-Modus. ' +
      'Für Online-Rennen den Node-Server starten (siehe README).';
    btn.disabled = false;
    net = null;
    return;
  }
  btn.disabled = false;
  netStatus.textContent = '';
  menu.classList.add('hidden');
  lobby.classList.remove('hidden');
  document.getElementById('lobby-status').textContent = '';

  net.on('lobby', (msg) => {
    const list = document.getElementById('lobby-list');
    list.innerHTML = '';
    for (const p of msg.players.filter((p) => p.name)) {
      const li = document.createElement('li');
      li.textContent = `🏎️ ${p.name}` + (p.racing ? ' (im Rennen)' : '');
      list.appendChild(li);
    }
    const info = document.getElementById('lobby-info');
    if (msg.racing) info.textContent = 'Ein Rennen läuft – du startest beim nächsten.';
    else if (msg.countdown != null) info.textContent = `Start in ${Math.max(0, Math.ceil(msg.countdown))} s – freie Plätze werden mit Bots gefüllt.`;
    else info.textContent = 'Warte auf Fahrer…';
  });

  net.on('wait', (msg) => {
    document.getElementById('lobby-info').textContent = msg.reason;
  });

  net.on('raceSetup', (msg) => {
    endGame();
    lobby.classList.add('hidden');
    game = new Game({
      canvas, mode: 'online', playerName: playerName(), charId: selectedChar, net,
      onRaceOver: () => {},
    });
    game.setupOnline(msg);
  });

  net.on('snapshot', (msg) => {
    if (game && game.mode === 'online') game.applySnapshot(msg);
  });

  net.on('close', () => {
    endGame();
    lobby.classList.add('hidden');
    menu.classList.remove('hidden');
    netStatus.textContent = 'Verbindung zum Server verloren.';
    net = null;
  });

  net.send({ type: 'join', name: playerName(), charId: selectedChar });
}

document.getElementById('btn-online').addEventListener('click', startOnline);

// ------------------------------------------------------------- results UI

document.getElementById('btn-again').addEventListener('click', () => {
  document.getElementById('results').classList.add('hidden');
  if (game && game.mode === 'solo') startSolo();
  else if (net) { lobby.classList.remove('hidden'); endGame(); }
});

document.getElementById('btn-menu').addEventListener('click', () => {
  document.getElementById('results').classList.add('hidden');
  endGame();
  if (net) { net.close(); net = null; }
  lobby.classList.add('hidden');
  menu.classList.remove('hidden');
  netStatus.textContent = '';
});
