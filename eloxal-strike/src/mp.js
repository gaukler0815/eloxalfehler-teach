/* Eloxal Strike — deathmatch multiplayer: lobby UI, remote player avatars
 * with interpolation, shooter-side hit claims, kill feed, scoreboard and
 * respawn flow. Transport in net.js; main.js provides hooks via init().
 */
(function () {
  'use strict';

  var net = window.ES.net;
  var sound = window.ES.sound;
  var H = null;              // hooks from main.js
  var $ = function (id) { return document.getElementById(id); };

  var myId = 0;
  var myName = 'SPIELER';
  var room = null;           // latest room detail from the server
  var inMatch = false;
  var dead = false;
  var deadT = 0;
  var lastAttacker = 0;
  var sendT = 0;
  var avatars = {};          // peerId → avatar record
  var COLORS = [0xE8A33D, 0x58c7f0, 0x9fe348, 0xff5c6c, 0xb18cff, 0x4dd0c4];

  function lin(hex) { return new THREE.Color(hex).convertSRGBToLinear(); }

  // ------------------------------------------------------------------ avatars
  function nameTexture(name, color) {
    var c = document.createElement('canvas');
    c.width = 256; c.height = 64;
    var g = c.getContext('2d');
    g.font = '700 34px system-ui, sans-serif';
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    g.lineWidth = 6;
    g.strokeStyle = 'rgba(10,10,14,0.9)';
    g.strokeText(name, 128, 32);
    g.fillStyle = '#' + new THREE.Color(color).getHexString();
    g.fillText(name, 128, 32);
    var tex = new THREE.CanvasTexture(c);
    tex.encoding = THREE.sRGBEncoding;
    return tex;
  }

  /* Little worker-soldier: overalls, chest plate in the player color,
   * helmet with visor, gun. Body + head are the hitboxes. */
  function buildAvatar(peer) {
    var color = COLORS[peer.colorIdx % COLORS.length];
    var g = new THREE.Group();
    var suit = new THREE.MeshStandardMaterial({ color: lin(0x3a4048), roughness: 0.8, metalness: 0.2 });
    var tint = new THREE.MeshStandardMaterial({ color: lin(color), roughness: 0.5, metalness: 0.5 });
    var skin = new THREE.MeshStandardMaterial({ color: lin(0xc9a184), roughness: 0.8 });
    var dark = new THREE.MeshStandardMaterial({ color: lin(0x22262e), roughness: 0.5, metalness: 0.7 });

    var legs = [];
    [-1, 1].forEach(function (side) {
      var leg = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.75, 0.2), suit);
      leg.position.set(side * 0.12, 0.375, 0);
      leg.castShadow = true;
      g.add(leg);
      legs.push(leg);
    });

    var body = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.62, 0.3), suit);
    body.position.y = 1.06;
    body.castShadow = true;
    g.add(body);
    var chest = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.34, 0.06), tint);
    chest.position.set(0, 1.12, -0.17);
    g.add(chest);
    var pack = new THREE.Mesh(new THREE.BoxGeometry(0.36, 0.4, 0.16), dark);
    pack.position.set(0, 1.1, 0.22);
    g.add(pack);

    // aim pivot carries head + gun so the whole upper body follows pitch
    var aim = new THREE.Group();
    aim.position.y = 1.5;
    g.add(aim);
    var head = new THREE.Mesh(new THREE.SphereGeometry(0.17, 10, 8), skin);
    head.position.y = 0.12;
    head.castShadow = true;
    aim.add(head);
    var helmet = new THREE.Mesh(new THREE.SphereGeometry(0.2, 10, 8, 0, Math.PI * 2, 0, Math.PI * 0.55), tint);
    helmet.position.y = 0.15;
    aim.add(helmet);
    var visor = new THREE.Mesh(new THREE.BoxGeometry(0.26, 0.08, 0.05), dark);
    visor.position.set(0, 0.12, -0.16);
    aim.add(visor);
    var gun = new THREE.Group();
    var gunBody = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.1, 0.5), dark);
    gun.add(gunBody);
    var gunTip = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.2, 8), dark);
    gunTip.rotation.x = Math.PI / 2;
    gunTip.position.set(0, 0.02, -0.32);
    gun.add(gunTip);
    gun.position.set(0.2, -0.18, -0.3);
    aim.add(gun);
    var arm = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.12, 0.4), suit);
    arm.position.set(0.2, -0.15, -0.05);
    aim.add(arm);

    var tag = new THREE.Sprite(new THREE.SpriteMaterial({
      map: nameTexture(peer.name, color), transparent: true, depthWrite: false
    }));
    tag.scale.set(1.5, 0.375, 1);
    tag.position.y = 2.05;
    g.add(tag);

    H.scene.add(g);
    var av = {
      id: peer.id, name: peer.name, color: color,
      group: g, body: body, head: head, aim: aim, legs: legs, gun: gun,
      buf: [],               // interpolation buffer [{t, x,y,z, yaw, pitch}]
      walkPhase: 0, lastPos: new THREE.Vector3(), visible: true
    };
    body.userData = { mpAvatar: av, isHead: false };
    head.userData = { mpAvatar: av, isHead: true };
    return av;
  }

  function removeAvatar(id) {
    var av = avatars[id];
    if (!av) { return; }
    H.scene.remove(av.group);
    delete avatars[id];
  }

  function syncAvatars() {
    if (!room) { return; }
    var seen = {};
    room.players.forEach(function (p) {
      seen[p.id] = true;
      if (p.id !== myId && !avatars[p.id] && inMatch) {
        avatars[p.id] = buildAvatar(p);
      }
    });
    Object.keys(avatars).forEach(function (id) {
      if (!seen[id]) { removeAvatar(id); }
    });
  }

  function clearAvatars() {
    Object.keys(avatars).forEach(removeAvatar);
  }

  // ------------------------------------------------------------------ UI
  function setMpStatus(text, isError) {
    var el = $('mp-status');
    el.textContent = text;
    el.classList.toggle('err', !!isError);
  }

  function renderRooms(list) {
    var box = $('mp-roomlist');
    box.innerHTML = '';
    if (!list.length) {
      box.innerHTML = '<div class="mp-empty">Noch keine Welt offen — mach die erste auf!</div>';
      return;
    }
    list.forEach(function (r) {
      var row = document.createElement('div');
      row.className = 'mp-room';
      row.innerHTML = '<div><b>' + r.name + '</b><span class="mp-sub">Host: ' + r.host +
        ' · ' + r.players + '/8 Spieler' + (r.started ? ' · läuft' : '') + '</span></div>';
      var btn = document.createElement('button');
      btn.className = 'btn small';
      btn.textContent = r.started ? 'Läuft…' : '▶ Beitreten';
      btn.disabled = !!r.started;
      btn.addEventListener('click', function () { net.send({ t: 'join', id: r.id }); });
      row.appendChild(btn);
      box.appendChild(row);
    });
  }

  function renderRoom() {
    var lobby = $('mp-lobbyview');
    var roomview = $('mp-roomview');
    if (!room) {
      lobby.classList.remove('hidden');
      roomview.classList.add('hidden');
      return;
    }
    lobby.classList.add('hidden');
    roomview.classList.remove('hidden');
    $('mp-roomname').textContent = room.name;
    var plist = $('mp-players');
    plist.innerHTML = '';
    room.players.forEach(function (p) {
      var row = document.createElement('div');
      row.className = 'mp-player';
      var col = '#' + new THREE.Color(COLORS[p.colorIdx % COLORS.length]).getHexString();
      row.innerHTML = '<span class="dot" style="background:' + col + '"></span>' +
        '<b>' + p.name + '</b>' + (p.id === room.hostId ? ' 👑' : '') +
        (p.id === myId ? ' (du)' : '') +
        '<span class="mp-sub" style="margin-left:auto">' + p.kills + ' Abschüsse</span>';
      plist.appendChild(row);
    });
    var isHost = room.hostId === myId;
    $('mp-start').classList.toggle('hidden', !isHost);
    $('mp-waithost').classList.toggle('hidden', isHost);
  }

  function updateScoreboard() {
    if (!room) { return; }
    var sb = $('scorepanel');
    var rows = room.players.slice().sort(function (a, b) { return b.kills - a.kills; });
    sb.innerHTML = '<div class="sb-title">Deathmatch — erster bei 15</div>' +
      rows.map(function (p) {
        var col = '#' + new THREE.Color(COLORS[p.colorIdx % COLORS.length]).getHexString();
        return '<div class="sb-row' + (p.id === myId ? ' me' : '') + '">' +
          '<span class="dot" style="background:' + col + '"></span>' +
          '<span class="sb-name">' + p.name + '</span>' +
          '<span class="sb-score">' + p.kills + ' / ' + p.deaths + '</span></div>';
      }).join('');
  }

  var feedTimers = [];
  function killFeed(text) {
    var feed = $('killfeed');
    var line = document.createElement('div');
    line.className = 'kf-line';
    line.innerHTML = text;
    feed.appendChild(line);
    while (feed.children.length > 5) { feed.removeChild(feed.firstChild); }
    feedTimers.push(setTimeout(function () {
      if (line.parentNode) { line.parentNode.removeChild(line); }
    }, 6000));
  }

  // ------------------------------------------------------------------ match
  function randomSpawn() {
    var sp = H.world.spawnPoints[Math.floor(Math.random() * H.world.spawnPoints.length)];
    return { x: sp[0] * 0.85, z: sp[1] * 0.85 };
  }

  function startMatch() {
    inMatch = true;
    dead = false;
    lastAttacker = 0;
    clearAvatars();
    ES.enemies.clear(H.scene);
    ES.weapons.reset();
    ES.weapons.setAim(false);
    var sp = randomSpawn();
    H.player.pos.set(sp.x, 1.7, sp.z);
    H.player.yaw = Math.atan2(sp.x, sp.z);   // face the hall center
    H.player.pitch = 0;
    H.player.hp = 100;
    H.player.lastHurt = -99;
    syncAvatars();
    updateScoreboard();
    $('killfeed').innerHTML = '';
    $('scorepanel').classList.remove('hidden');
    H.setState('mp');
    sound.unlock();
    sound.startDrone();
    H.canvas.requestPointerLock();
    H.showBanner('Deathmatch!', 'Erster mit 15 Abschüssen gewinnt — Feuer frei!', 3);
    sound.waveStart();
  }

  function endMatch(winnerName) {
    if (!inMatch) { return; }
    inMatch = false;
    dead = false;
    $('respawn').classList.add('hidden');
    $('scorepanel').classList.add('hidden');
    clearAvatars();
    sound.stopDrone();
    sound.waveClear();
    if (document.pointerLockElement === H.canvas) { document.exitPointerLock(); }
    H.setState('mplobby');
    setMpStatus(winnerName ? '🏆 ' + winnerName + ' gewinnt die Runde!' : 'Runde beendet.');
  }

  function localDied() {
    if (dead) { return; }
    dead = true;
    deadT = 3.5;
    net.send({ t: 'died', killer: lastAttacker });
    sound.gameOver();
    $('respawn').classList.remove('hidden');
  }

  function respawnNow() {
    dead = false;
    var sp = randomSpawn();
    H.player.pos.set(sp.x, 1.7, sp.z);
    H.player.hp = 100;
    H.player.lastHurt = -99;
    ES.weapons.reset();
    $('respawn').classList.add('hidden');
  }

  // ------------------------------------------------------------------ net in
  function wireNet() {
    net.on('open', function () { setMpStatus('Verbunden als ' + myName + '.'); });
    net.on('close', function () {
      setMpStatus('Verbindung getrennt.', true);
      if (inMatch) { endMatch(null); }
      room = null;
      renderRoom();
    });
    net.on('error', function (msg) { setMpStatus(msg, true); });
    net.on('welcome', function (msg) { myId = msg.id; myName = msg.name; });
    net.on('rooms', function (msg) { renderRooms(msg.rooms); });
    net.on('room', function (msg) {
      room = msg.room;
      renderRoom();
      updateScoreboard();
      if (inMatch) { syncAvatars(); }
    });
    net.on('peerleft', function (msg) {
      removeAvatar(msg.id);
      if (inMatch) { killFeed('<b>' + msg.name + '</b> hat die Halle verlassen'); }
    });
    net.on('started', function () { startMatch(); });
    net.on('ended', function (msg) {
      H.showBanner('🏆 ' + msg.winnerName + ' gewinnt!', 'Zurück zur Lobby…', 4);
      setTimeout(function () { endMatch(msg.winnerName); }, 3500);
    });

    net.on('state', function (msg) {
      var av = avatars[msg.from];
      if (!av) { return; }
      av.buf.push({ t: performance.now(), x: msg.p[0], y: msg.p[1], z: msg.p[2],
        yaw: msg.yaw, pitch: msg.pitch });
      if (av.buf.length > 20) { av.buf.shift(); }
      av.visible = !msg.dead;
      av.group.visible = av.visible;
    });

    net.on('shot', function (msg) {
      var av = avatars[msg.from];
      if (!av || !av.visible) { return; }
      var origin = new THREE.Vector3().setFromMatrixPosition(av.gun.children[1].matrixWorld);
      var dir = new THREE.Vector3(msg.d[0], msg.d[1], msg.d[2]);
      var ray = new THREE.Raycaster(origin, dir, 0.1, 120);
      var hits = ray.intersectObjects(H.world.solids, false);
      var end = hits.length ? hits[0].point
        : origin.clone().addScaledVector(dir, 60);
      H.tracer(origin, end, 0xffd166);
      var dist = origin.distanceTo(H.player.pos);
      if (dist < 45) { sound.shoot(msg.w || 'anodisierer'); }
    });

    net.on('hit', function (msg) {
      lastAttacker = msg.from;
      H.hurtPlayer(msg.dmg);
    });

    net.on('died', function (msg) {
      var who = msg.from === myId ? 'DU' : ((avatars[msg.from] || {}).name || '?');
      var killer = msg.killer === myId ? 'DIR' : (msg.killerName || '?');
      killFeed('<b>' + killer + '</b> ⚡ ' + who);
      if (msg.killer === myId && msg.from !== myId) {
        sound.kill();
        H.showHitmarker(true);
      }
      var av = avatars[msg.from];
      if (av) {
        H.burst(av.group.position.clone().setY(1.2), av.color, 18, 5, 8);
        av.group.visible = false;
        av.visible = false;
      }
    });
  }

  // ------------------------------------------------------------------ API
  var api = {
    init: function (hooks) {
      H = hooks;
      wireNet();

      var nameInput = $('mp-name');
      var serverInput = $('mp-server');
      nameInput.value = localStorage.getItem('eloxal-strike-name') || '';
      serverInput.value = localStorage.getItem('eloxal-strike-server') || net.defaultUrl();

      $('btn-mp').addEventListener('click', function () {
        H.setState('mplobby');
        if (!net.connected()) { api.connect(); }
      });
      $('mp-connect').addEventListener('click', api.connect);
      $('mp-back').addEventListener('click', function () {
        net.send({ t: 'leave' });
        net.disconnect();
        room = null;
        renderRoom();
        H.setState('menu');
      });
      $('mp-create').addEventListener('click', function () {
        var rn = ($('mp-worldname').value || myName + 'S HALLE');
        net.send({ t: 'create', name: rn });
      });
      $('mp-leave').addEventListener('click', function () {
        net.send({ t: 'leave' });
        room = null;
        renderRoom();
      });
      $('mp-start').addEventListener('click', function () { net.send({ t: 'start' }); });
    },

    connect: function () {
      var name = ($('mp-name').value || 'SPIELER').toUpperCase().replace(/[^A-Z0-9ÄÖÜ ]/g, '').slice(0, 12) || 'SPIELER';
      var url = $('mp-server').value.trim() || net.defaultUrl();
      myName = name;
      localStorage.setItem('eloxal-strike-name', name);
      localStorage.setItem('eloxal-strike-server', url);
      setMpStatus('Verbinde mit ' + url + ' …');
      net.connect(url, name);
    },

    active: function () { return inMatch; },
    blocked: function () { return dead; },

    hittables: function () {
      var out = [];
      Object.keys(avatars).forEach(function (id) {
        var av = avatars[id];
        if (av.visible) { out.push(av.body, av.head); }
      });
      return out;
    },

    /* Shooter-side hit claim: our ray hit a remote avatar. */
    onLocalHit: function (av, dmg, isHead) {
      net.send({ t: 'hit', target: av.id, dmg: Math.round(dmg), head: isHead });
    },

    onLocalShot: function (dir, weaponId) {
      net.send({ t: 'shot', d: [dir.x, dir.y, dir.z], w: weaponId });
    },

    localDied: localDied,

    leaveToMenu: function () {
      if (inMatch) { endMatch(null); }
      net.send({ t: 'leave' });
      room = null;
      renderRoom();
    },

    update: function (dt) {
      if (!inMatch) { return; }

      // dead: count down, then respawn
      if (dead) {
        deadT -= dt;
        $('respawn-count').textContent = Math.max(1, Math.ceil(deadT));
        if (deadT <= 0) { respawnNow(); }
      }

      // 15 Hz state broadcast
      sendT -= dt;
      if (sendT <= 0) {
        sendT = 1 / 15;
        net.send({
          t: 'state',
          p: [+H.player.pos.x.toFixed(2), +H.player.pos.y.toFixed(2), +H.player.pos.z.toFixed(2)],
          yaw: +H.player.yaw.toFixed(3), pitch: +H.player.pitch.toFixed(3),
          dead: dead
        });
      }

      // interpolate remote avatars 120 ms in the past
      var renderT = performance.now() - 120;
      Object.keys(avatars).forEach(function (id) {
        var av = avatars[id];
        var buf = av.buf;
        if (buf.length === 0) { return; }
        var a = buf[0], b = buf[buf.length - 1];
        for (var i = buf.length - 1; i >= 0; i--) {
          if (buf[i].t <= renderT) { a = buf[i]; b = buf[Math.min(i + 1, buf.length - 1)]; break; }
        }
        var span = Math.max(1, b.t - a.t);
        var f = Math.max(0, Math.min(1, (renderT - a.t) / span));
        var x = a.x + (b.x - a.x) * f;
        var y = a.y + (b.y - a.y) * f;
        var z = a.z + (b.z - a.z) * f;
        var speed = av.lastPos.distanceTo(new THREE.Vector3(x, y, z)) / Math.max(dt, 0.001);
        av.lastPos.set(x, y, z);
        av.group.position.set(x, y - 1.7, z);   // state carries eye height
        av.group.rotation.y = a.yaw + (b.yaw - a.yaw) * f;
        av.aim.rotation.x = a.pitch + (b.pitch - a.pitch) * f;
        // walk cycle
        av.walkPhase += dt * Math.min(14, 2 + speed * 2.2);
        var swing = speed > 0.5 ? Math.sin(av.walkPhase) * 0.5 : 0;
        av.legs[0].rotation.x = swing;
        av.legs[1].rotation.x = -swing;
      });
    }
  };

  window.ES = window.ES || {};
  window.ES.mp = api;
})();
