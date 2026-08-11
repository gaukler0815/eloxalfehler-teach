/*
 * main.js — bootstrap and glue. Creates the game and renderer, runs the loop,
 * translates pointer input into slingshot/ability actions, and drives the DOM
 * overlays (menu, level-end protocol, leaderboard). All visible text is German.
 */
(function () {
  'use strict';
  const ER = window.ER;
  const canvas = document.getElementById('stage');
  const game = ER.game.create();
  const render = ER.render.create(canvas, game);

  // Small debug hooks (used by the headless browser check and handy in devtools).
  window.__game = game;
  window.__anchor = game.anchor;
  window.__render = render;

  // --- juice wiring: game events -> sound + particles ---------------------
  const sound = ER.sound, particles = ER.particles;
  game.on('launch', () => sound.launch());
  game.on('impact', (e) => {
    sound.impact(e.rv);
    if (e.rv > 5) particles.dust(e.x, e.y, e.rv);
  });
  game.on('blockdestroyed', (e) => {
    particles.shards(e.x, e.y, e.color, e.material === 'kartonage' ? 7 : 12);
    if (e.material === 'fehlcharge') sound.shatter();
    else if (e.material === 'kartonage') sound.thudSoft();
    else sound.crack();
  });
  game.on('enemykill', (e) => {
    particles.burst(e.x, e.y, '#E33A2C', 14, 8);
    particles.pop(e.x, e.y - 40, 'ZACK!', '#F5A81C');
    sound.pop();
  });
  game.on('boom', (e) => {
    particles.burst(e.x, e.y, '#F5A81C', 20, 11);
    particles.dust(e.x, e.y, 20);
    sound.boom();
  });
  game.on('arc', (e) => {
    if (e.segs) particles.sparks(e.segs);
    sound.zap();
  });
  game.on('enemyhurt', (e) => {
    particles.burst(e.x, e.y, '#FFFFFF', 10, 6);
    particles.pop(e.x, e.y - 70, e.hp + ' TREFFER NOCH', '#8FD3FF');
    sound.crack();
  });

  const PROGRESS_KEY = 'eloxal-rebels.progress.v1';
  const $ = (id) => document.getElementById(id);

  // --- progress (best µm per level) -------------------------------------
  function loadProgress() {
    try { return JSON.parse(localStorage.getItem(PROGRESS_KEY)) || { best: {} }; }
    catch (e) { return { best: {} }; }
  }
  function saveProgress(p) {
    try { localStorage.setItem(PROGRESS_KEY, JSON.stringify(p)); } catch (e) { /* best effort */ }
  }
  let progress = loadProgress();
  function totalUm() {
    return Object.values(progress.best).reduce((a, b) => a + b, 0);
  }

  // --- level flow --------------------------------------------------------
  let currentId = null;
  function startLevel(id) {
    currentId = id;
    particles.clear();
    game.loadLevel(ER.levels.get(id));
    window.__anchor = game.anchor;
    hide('menu'); hide('levelend'); hide('board');
    syncShots(true);
  }

  game.on('levelend', (e) => {
    if (e.won) {
      const prev = progress.best[currentId] || 0;
      if (e.um > prev) { progress.best[currentId] = e.um; saveProgress(progress); }
      particles.confetti(960, 200);
      particles.pop(960, 420, '+' + e.um + ' µm', '#F5A81C');
      sound.win();
    } else {
      sound.lose();
    }
    // let the confetti fall before the protocol slides in
    setTimeout(() => showLevelEnd(e), e.won ? 900 : 500);
  });

  // --- render / update loop ---------------------------------------------
  let frameNo = 0;
  function frame() {
    frameNo++;
    if (game.state === 'aim' || game.state === 'flying') game.update();
    particles.update();
    render.draw(frameNo);
    syncShots(false);
    $('umTotal').querySelector('b').textContent = totalUm();
    requestAnimationFrame(frame);
  }

  // --- HUD ---------------------------------------------------------------
  let lastSig = '';
  function syncShots(force) {
    const info = $('levelInfo');
    if (game.level) {
      info.querySelector('b').textContent = game.level.name;
      info.querySelector('small').textContent = game.level.subtitle || '';
    }
    const remaining = (game.currentType ? [game.currentType] : []).concat(game.queue);
    const total = game.shotsTotal;
    const sig = remaining.join(',') + '/' + total;
    if (!force && sig === lastSig) return;
    lastSig = sig;
    const box = $('shots');
    box.innerHTML = '';
    const P = ER.config.PROJECTILES;
    for (let i = 0; i < total; i++) {
      const type = remaining[i - (total - remaining.length)];
      if (i < total - remaining.length) {
        const chip = document.createElement('div');
        chip.className = 'chip spent';
        chip.textContent = '·';
        box.appendChild(chip);
      } else {
        // tiny character portrait instead of a colored dot
        const def = P[type] || P.ali;
        const c = document.createElement('canvas');
        c.className = 'chip';
        c.width = 68; c.height = 68;
        c.title = def.label + ' — ' + def.hint;
        ER.characters.drawStatic(c.getContext('2d'), type, 34, 38, 17, 40);
        box.appendChild(c);
      }
    }
  }

  // --- level end screen --------------------------------------------------
  let lastEntry = null;
  function showLevelEnd(e) {
    $('endTitle').textContent = e.won ? 'GESCHAFFT' : 'GESPERRT';
    $('endSub').textContent = e.won
      ? (game.level.name + ' — ' + e.um + ' µm Schichtdicke (' + e.leftover + ' Geschosse übrig).')
      : (game.level.name + ' — die Charge ist gesperrt. Noch ein Versuch?');
    $('umScale').querySelectorAll('div').forEach((d) => {
      d.classList.toggle('on', e.won && Number(d.dataset.v) === e.um);
    });
    const next = ER.levels.nextId(currentId);
    const btnNext = $('btnNext');
    if (e.won && next) { btnNext.textContent = 'Nächstes Level'; btnNext.onclick = () => startLevel(next); }
    else if (e.won) { btnNext.textContent = 'Weltmenü'; btnNext.onclick = openMenu; }
    else { btnNext.textContent = 'Weltmenü'; btnNext.onclick = openMenu; }
    $('btnAgain').onclick = () => startLevel(currentId);
    $('protoUm').textContent = totalUm();
    $('nameErr').textContent = '';
    lastEntry = null;
    renderBoardInto($('endBoard'), 'total');
    show('levelend');
  }

  // --- leaderboard rendering --------------------------------------------
  function renderBoardInto(el, scope) {
    const v = ER.leaderboard.view(scope, lastEntry, 10);
    let html = '<table><thead><tr><th>#</th><th>Name</th><th>µm</th></tr></thead><tbody>';
    const seen = new Set();
    v.top.forEach((r) => {
      const own = lastEntry && r.ts === lastEntry.ts;
      if (own) seen.add(r.ts);
      html += '<tr class="' + (own ? 'own' : '') + '"><td>' + r.rank + '</td><td>' + escapeHtml(r.name) + '</td><td>' + r.um + '</td></tr>';
    });
    if (v.own && !seen.has(v.own.ts)) {
      html += '<tr><td colspan="3" style="opacity:.4;text-align:center">···</td></tr>';
      html += '<tr class="own"><td>' + v.own.rank + '</td><td>' + escapeHtml(v.own.name) + '</td><td>' + v.own.um + '</td></tr>';
    }
    if (!v.top.length) html += '<tr><td colspan="3" style="opacity:.6">Noch keine Einträge.</td></tr>';
    html += '</tbody></table>';
    el.innerHTML = html;
  }
  function escapeHtml(s) { return String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c])); }

  $('btnSubmit').onclick = () => {
    const res = ER.leaderboard.submit($('nameInput').value, totalUm());
    if (res.error) { $('nameErr').textContent = res.error; return; }
    lastEntry = res.entry;
    $('nameErr').textContent = '';
    $('nameInput').value = res.entry.name;
    renderBoardInto($('endBoard'), 'total');
  };

  // --- full board overlay ------------------------------------------------
  let boardScope = 'total';
  function openBoard() {
    boardScope = 'total';
    document.querySelectorAll('#board .tabs button').forEach((b) => b.classList.toggle('active', b.dataset.scope === 'total'));
    renderBoardInto($('boardBody'), boardScope);
    show('board');
  }
  document.querySelectorAll('#board .tabs button').forEach((b) => {
    b.onclick = () => {
      boardScope = b.dataset.scope;
      document.querySelectorAll('#board .tabs button').forEach((x) => x.classList.toggle('active', x === b));
      renderBoardInto($('boardBody'), boardScope);
    };
  });
  $('btnBoard').onclick = openBoard;
  $('btnBoardClose').onclick = () => { hide('board'); };

  // --- menu --------------------------------------------------------------
  function buildMenu() {
    const list = $('levelList');
    list.innerHTML = '';
    ER.levels.order.forEach((id, i) => {
      const lv = ER.levels.get(id);
      const div = document.createElement('div');
      const best = progress.best[id];
      // each won level unlocks the next one
      const prevId = i > 0 ? ER.levels.order[i - 1] : null;
      const unlocked = i === 0 || (progress.best[prevId] || 0) > 0;
      div.className = 'lvl' + (unlocked ? '' : ' locked');
      div.innerHTML = '<span class="w">Welt ' + lv.world + ' · Level ' + (i + 1) + '</span><b>' + lv.name + '</b>' +
        '<span class="badge">' + (best ? best + ' µm' : (unlocked ? '—' : '')) + '</span>' +
        (unlocked ? '' : '<span class="lock">🔒</span>');
      if (unlocked) div.onclick = () => startLevel(id);
      list.appendChild(div);
    });
  }
  function openMenu() { buildMenu(); hide('levelend'); hide('board'); show('menu'); }

  $('btnMenu').onclick = openMenu;
  $('btnRetry').onclick = () => { if (currentId) startLevel(currentId); };
  $('btnSound').onclick = () => {
    sound.unlock();
    sound.setMuted(!sound.isMuted());
    $('btnSound').textContent = sound.isMuted() ? '🔇' : '🔊';
  };

  // --- savegame file export / import -------------------------------------
  // Progress and leaderboard already persist automatically in localStorage;
  // the file round-trip exists to move a save to another device (e.g. the
  // trade-show machine) or to keep a backup.
  function saveMsg(text, isError) {
    const el = $('saveMsg');
    el.textContent = text;
    el.style.color = isError ? '' : '#7ED957';
  }
  $('btnSaveExport').onclick = () => {
    const data = {
      game: 'eloxal-rebels', version: 1, savedAt: new Date().toISOString(),
      progress, scores: ER.leaderboard.exportData()
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'eloxal-rebels-spielstand.json';
    a.click();
    URL.revokeObjectURL(a.href);
    saveMsg('Spielstand als Datei gesichert (' + totalUm() + ' µm).', false);
  };
  $('btnSaveImport').onclick = () => $('saveFile').click();
  $('saveFile').addEventListener('change', (e) => {
    const file = e.target.files && e.target.files[0];
    e.target.value = '';
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const res = applySaveData(reader.result);
      saveMsg(res.error || ('Spielstand geladen: ' + totalUm() + ' µm.'), !!res.error);
    };
    reader.readAsText(file);
  });
  // Parse + validate a savegame string and apply it. Returns {error} on failure.
  function applySaveData(text) {
    let data;
    try { data = JSON.parse(text); } catch (err) { return { error: 'Datei ist kein gültiger Spielstand.' }; }
    if (!data || data.game !== 'eloxal-rebels' || typeof data.progress !== 'object' || !data.progress) {
      return { error: 'Datei ist kein gültiger Spielstand.' };
    }
    const best = {};
    Object.entries(data.progress.best || {}).forEach(([id, um]) => {
      if (typeof um === 'number' && um > 0) best[id] = Math.min(um, ER.config.SCORING.maxPerLevel);
    });
    progress = { best };
    saveProgress(progress);
    if (Array.isArray(data.scores)) ER.leaderboard.importData(data.scores);
    buildMenu();
    return {};
  }
  window.__applySaveData = applySaveData; // debug/test hook

  // --- overlay helpers ---------------------------------------------------
  function show(id) { $(id).classList.add('show'); }
  function hide(id) { $(id).classList.remove('show'); }

  // --- pointer input -----------------------------------------------------
  let grabbing = false;
  function evPos(e) {
    const r = canvas.getBoundingClientRect();
    const cx = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
    const cy = (e.touches ? e.touches[0].clientY : e.clientY) - r.top;
    return render.toWorld(cx, cy);
  }
  canvas.addEventListener('pointerdown', (e) => {
    sound.unlock(); // audio may only start after a user gesture
    if (isOverlayOpen()) return;
    const w = evPos(e);
    if (game.state === 'aim') grabbing = game.beginAim(w.x, w.y);
    else if (game.state === 'flying') game.tap();
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!grabbing) return;
    const w = evPos(e);
    game.moveAim(w.x, w.y);
    const pull = Math.hypot(game.aimPos.x - game.anchor.x, game.aimPos.y - game.anchor.y);
    sound.stretch(pull / ER.config.PHYSICS.maxPull);
  });
  document.addEventListener('pointerdown', () => sound.unlock(), { once: true });
  window.addEventListener('pointerup', () => {
    if (grabbing) { game.releaseAim(); grabbing = false; }
  });
  function isOverlayOpen() {
    return $('menu').classList.contains('show') || $('levelend').classList.contains('show') || $('board').classList.contains('show');
  }

  // keyboard shortcuts
  window.addEventListener('keydown', (e) => {
    if (e.key === 'r' || e.key === 'R') { if (currentId) startLevel(currentId); }
    else if (e.key === 'Escape') openMenu();
    else if (e.key === ' ') { if (game.state === 'flying') game.tap(); }
  });

  // --- go ----------------------------------------------------------------
  buildMenu();
  requestAnimationFrame(frame);
})();
