/*
 * editor.js — the level editor (second page). Place fortress parts and enemies
 * with the mouse, move the slingshot, play-test with the real engine, and
 * export the result as JSON that drops straight into levels/. Same modules as
 * the game, so what you build is exactly what plays.
 */
(function () {
  'use strict';
  const ER = window.ER;
  const canvas = document.getElementById('stage');
  const game = ER.game.create();
  const render = ER.render.create(canvas, game);
  const $ = (id) => document.getElementById(id);

  let doc = blankDoc();
  let mode = 'block';
  let testing = false;
  let nextNum = 1;

  function blankDoc() {
    return {
      id: $('fId') ? $('fId').value : 'neu-01', world: 1, name: 'Neues Level',
      subtitle: '', slingshot: { x: 340, y: 720 },
      projectiles: ['ali', 'ali', 'ali'], blocks: [], enemies: []
    };
  }
  function clone(o) { return JSON.parse(JSON.stringify(o)); }
  function uid(prefix) { return prefix + (nextNum++); }

  // --- populate selects --------------------------------------------------
  function fillSelects() {
    const mat = $('fMat');
    Object.keys(ER.config.MATERIALS).forEach((k) => {
      const o = document.createElement('option');
      o.value = k; o.textContent = ER.config.MATERIALS[k].label; mat.appendChild(o);
    });
    const en = $('fEnemy');
    Object.keys(ER.config.ENEMIES).forEach((k) => {
      const o = document.createElement('option');
      o.value = k; o.textContent = ER.config.ENEMIES[k].label; en.appendChild(o);
    });
    const load = $('loadSel');
    const blank = document.createElement('option');
    blank.value = ''; blank.textContent = '— leeres Level —'; load.appendChild(blank);
    ER.levels.order.forEach((id) => {
      const o = document.createElement('option');
      o.value = id; o.textContent = id; load.appendChild(o);
    });
  }

  // --- rebuild the visual world from doc (edit view, no simulation) ------
  function rebuild() {
    game.loadLevel(clone(doc));
    game.state = 'idle'; // freeze: the editor loop only draws
    render.resize();
  }

  function syncFieldsToDoc() {
    doc.name = $('fName').value || 'Neues Level';
    doc.world = parseInt($('fWorld').value, 10) || 1;
    doc.id = $('fId').value || 'neu-01';
    doc.projectiles = $('fProj').value.split(',').map((s) => s.trim()).filter(Boolean);
  }
  function syncDocToFields() {
    $('fName').value = doc.name;
    $('fWorld').value = doc.world;
    $('fId').value = doc.id;
    $('fProj').value = doc.projectiles.join(',');
  }

  // --- placing / erasing -------------------------------------------------
  function placeAt(w) {
    if (mode === 'block') {
      doc.blocks.push({
        id: uid('b'), material: $('fMat').value,
        x: Math.round(w.x), y: Math.round(w.y),
        w: parseInt($('fW').value, 10) || 100, h: parseInt($('fH').value, 10) || 80
      });
    } else if (mode === 'enemy') {
      doc.enemies.push({ id: uid('e'), type: $('fEnemy').value, x: Math.round(w.x), y: Math.round(w.y) });
    } else if (mode === 'sling') {
      doc.slingshot = { x: Math.round(w.x), y: Math.round(w.y) };
    } else if (mode === 'erase') {
      eraseAt(w);
    }
    syncFieldsToDoc();
    rebuild();
    exportJson();
  }
  function eraseAt(w) {
    const hit = (o, hw, hh) => Math.abs(o.x - w.x) <= hw && Math.abs(o.y - w.y) <= hh;
    let idx = doc.blocks.findIndex((b) => hit(b, b.w / 2 + 4, b.h / 2 + 4));
    if (idx >= 0) { doc.blocks.splice(idx, 1); return; }
    idx = doc.enemies.findIndex((e) => hit(e, 32, 32));
    if (idx >= 0) doc.enemies.splice(idx, 1);
  }

  // --- export / import ---------------------------------------------------
  function exportJson() {
    syncFieldsToDoc();
    $('json').value = JSON.stringify(doc, null, 2);
  }
  function importJson() {
    try {
      doc = JSON.parse($('json').value);
      syncDocToFields();
      rebuild();
    } catch (e) { alert('Ungültiges JSON: ' + e.message); }
  }
  function download() {
    exportJson();
    const blob = new Blob([$('json').value], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = (doc.id || 'level') + '.json';
    a.click();
    URL.revokeObjectURL(a.href);
  }

  // --- test mode ---------------------------------------------------------
  function setTesting(on) {
    testing = on;
    $('btnTest').textContent = on ? '■ Stop' : '▶ Testen';
    $('btnTest').classList.toggle('sel', on);
    if (on) { syncFieldsToDoc(); game.loadLevel(clone(doc)); }
    else rebuild();
  }

  // --- input -------------------------------------------------------------
  let grabbing = false;
  function evPos(e) {
    const r = canvas.getBoundingClientRect();
    return render.toWorld(e.clientX - r.left, e.clientY - r.top);
  }
  canvas.addEventListener('pointerdown', (e) => {
    const w = evPos(e);
    if (testing) {
      if (game.state === 'aim') grabbing = game.beginAim(w.x, w.y);
      else if (game.state === 'flying') game.tap();
    } else {
      placeAt(w);
    }
  });
  canvas.addEventListener('pointermove', (e) => {
    if (testing && grabbing) game.moveAim(evPos(e).x, evPos(e).y);
  });
  window.addEventListener('pointerup', () => {
    if (testing && grabbing) { game.releaseAim(); grabbing = false; }
  });

  // --- controls ----------------------------------------------------------
  document.querySelectorAll('.modes button').forEach((b) => {
    b.onclick = () => {
      mode = b.dataset.mode;
      document.querySelectorAll('.modes button').forEach((x) => x.classList.toggle('sel', x === b));
      $('blockOpts').style.display = mode === 'block' ? '' : 'none';
      $('enemyOpts').style.display = mode === 'enemy' ? '' : 'none';
    };
  });
  $('btnTest').onclick = () => setTesting(!testing);
  $('btnExport').onclick = exportJson;
  $('btnDownload').onclick = download;
  $('btnImport').onclick = importJson;
  ['fName', 'fWorld', 'fId', 'fProj'].forEach((id) => $(id).addEventListener('change', () => { syncFieldsToDoc(); exportJson(); }));
  $('loadSel').onchange = (e) => {
    if (e.target.value) { doc = ER.levels.get(e.target.value); }
    else { doc = blankDoc(); }
    syncDocToFields(); rebuild(); exportJson();
  };

  // --- loop --------------------------------------------------------------
  let frameNo = 0;
  function frame() {
    frameNo++;
    if (testing && (game.state === 'aim' || game.state === 'flying')) game.update();
    if (ER.particles) ER.particles.update();
    render.draw(frameNo);
    requestAnimationFrame(frame);
  }

  // --- go ----------------------------------------------------------------
  fillSelects();
  syncDocToFields();
  rebuild();
  exportJson();
  requestAnimationFrame(frame);
})();
