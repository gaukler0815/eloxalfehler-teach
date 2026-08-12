/* Eloxal Strike — game orchestration: renderer, player controller, waves,
 * hit resolution, particles, pickups, HUD and the menu/pause/gameover flow.
 * Loads last; everything else hangs off window.ES.
 */
(function () {
  'use strict';

  var cfg = ES.config;
  var sound = ES.sound;

  // --- DOM ------------------------------------------------------------------
  function $(id) { return document.getElementById(id); }
  var canvas = $('game');
  var ui = {
    menu: $('menu'), hud: $('hud'), pause: $('pause'), gameover: $('gameover'),
    hpFill: $('hp-fill'), hpNum: $('hp-num'),
    ammoMag: $('ammo-mag'), ammoReserve: $('ammo-reserve'), weaponName: $('weapon-name'),
    waveNum: $('wave-num'), scoreNum: $('score-num'), enemiesLeft: $('enemies-left'),
    banner: $('banner'), bannerTitle: $('banner-title'), bannerSub: $('banner-sub'),
    hitmarker: $('hitmarker'), damageOverlay: $('damage-overlay'),
    crosshair: $('crosshair'),
    goScore: $('go-score'), goWave: $('go-wave'), goKills: $('go-kills'), goBest: $('go-best'),
    diffLabel: $('hud-diff'), bestLabel: $('menu-best')
  };

  // --- three.js core --------------------------------------------------------
  var renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;

  var scene = new THREE.Scene();
  var camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.05, 200);
  camera.rotation.order = 'YXZ';
  scene.add(camera);

  var world = ES.world.build(scene);
  ES.weapons.init(camera);

  // cheap environment map: a tiny scene of light strips, prefiltered via
  // PMREM — gives all standard materials proper metallic reflections
  (function () {
    var envScene = new THREE.Scene();
    envScene.background = new THREE.Color(0x0a0c12);
    function strip(color, w, h, x, y, z, ry) {
      var m = new THREE.Mesh(
        new THREE.PlaneGeometry(w, h),
        new THREE.MeshBasicMaterial({ color: color, side: THREE.DoubleSide })
      );
      m.position.set(x, y, z);
      m.rotation.y = ry || 0;
      m.rotation.x = (y > 0) ? Math.PI / 2 : 0;
      envScene.add(m);
    }
    strip(0xfff2dd, 6, 1.4, -4, 8, 0);       // warm lamp row
    strip(0xfff2dd, 6, 1.4, 4, 8, 0);
    strip(0x9fb8e8, 3, 14, 0, 9, 0);          // cool skylight
    strip(0x86d32f, 10, 4, 0, -0.01, 0);      // acid glow from below
    strip(0xE8A33D, 4, 2, 0, 3, -8);          // gold signage
    var pmrem = new THREE.PMREMGenerator(renderer);
    scene.environment = pmrem.fromScene(envScene, 0.06).texture;
    pmrem.dispose();
  })();

  // post-processing: render -> bloom (bright lights/emissives glow) -> FXAA
  var fxEnabled = localStorage.getItem('eloxal-strike-fx') !== '0';
  var composer = null, bloomPass = null, fxaaPass = null;
  function setupComposer() {
    var w = window.innerWidth, h = window.innerHeight;
    composer = new THREE.EffectComposer(renderer);
    composer.addPass(new THREE.RenderPass(scene, camera));
    bloomPass = new THREE.UnrealBloomPass(new THREE.Vector2(w, h), 0.45, 0.55, 0.82);
    composer.addPass(bloomPass);
    // render targets are linear in r134 — convert back to sRGB at the end
    composer.addPass(new THREE.ShaderPass(THREE.GammaCorrectionShader));
    fxaaPass = new THREE.ShaderPass(THREE.FXAAShader);
    composer.addPass(fxaaPass);
    updateFxSizes();
  }
  function updateFxSizes() {
    if (!composer) { return; }
    var w = window.innerWidth, h = window.innerHeight;
    composer.setSize(w, h);
    var pr = renderer.getPixelRatio();
    fxaaPass.material.uniforms.resolution.value.set(1 / (w * pr), 1 / (h * pr));
  }
  setupComposer();

  window.addEventListener('resize', function () {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    updateFxSizes();
  });

  // --- particles ------------------------------------------------------------
  var particles = [];
  var particleGeo = new THREE.BoxGeometry(0.07, 0.07, 0.07);
  function burst(pos, color, count, speed, gravity) {
    for (var i = 0; i < count; i++) {
      var m = new THREE.Mesh(particleGeo, new THREE.MeshBasicMaterial({ color: color }));
      m.position.copy(pos);
      var v = new THREE.Vector3(
        (Math.random() - 0.5) * 2, Math.random() * 1.2, (Math.random() - 0.5) * 2
      ).normalize().multiplyScalar(speed * (0.4 + Math.random() * 0.8));
      particles.push({ mesh: m, vel: v, life: 0.4 + Math.random() * 0.5, gravity: gravity });
      scene.add(m);
    }
  }
  function updateParticles(dt) {
    for (var i = particles.length - 1; i >= 0; i--) {
      var p = particles[i];
      p.life -= dt;
      p.vel.y -= p.gravity * dt;
      p.mesh.position.addScaledVector(p.vel, dt);
      p.mesh.scale.multiplyScalar(Math.max(0.01, 1 - dt * 2.5));
      if (p.life <= 0 || p.mesh.position.y < 0) {
        scene.remove(p.mesh);
        p.mesh.material.dispose();
        particles.splice(i, 1);
      }
    }
  }

  // --- tracers --------------------------------------------------------------
  var tracers = [];
  function tracer(from, to, color) {
    var geo = new THREE.BufferGeometry().setFromPoints([from, to]);
    var mat = new THREE.LineBasicMaterial({ color: color, transparent: true, opacity: 0.8 });
    var line = new THREE.Line(geo, mat);
    scene.add(line);
    tracers.push({ line: line, life: 0.08 });
  }
  function updateTracers(dt) {
    for (var i = tracers.length - 1; i >= 0; i--) {
      var t = tracers[i];
      t.life -= dt;
      t.line.material.opacity = Math.max(0, t.life / 0.08);
      if (t.life <= 0) {
        scene.remove(t.line);
        t.line.geometry.dispose();
        t.line.material.dispose();
        tracers.splice(i, 1);
      }
    }
  }

  // --- pickups --------------------------------------------------------------
  var pickups = [];
  function spawnPickup(kind, x, z) {
    var color = kind === 'health' ? 0xff5c6c : 0xffd166;
    var box = new THREE.Mesh(
      new THREE.BoxGeometry(0.55, 0.55, 0.55),
      new THREE.MeshStandardMaterial({
        color: color, emissive: color, emissiveIntensity: 0.6, roughness: 0.4
      })
    );
    box.position.set(x, 0.8, z);
    scene.add(box);
    pickups.push({ kind: kind, mesh: box, t: Math.random() * 6 });
  }
  function clearPickups() {
    pickups.forEach(function (p) { scene.remove(p.mesh); });
    pickups = [];
  }
  function updatePickups(dt) {
    for (var i = pickups.length - 1; i >= 0; i--) {
      var p = pickups[i];
      p.t += dt;
      p.mesh.rotation.y += dt * 2;
      p.mesh.position.y = 0.8 + Math.sin(p.t * 2.4) * 0.15;
      var dx = p.mesh.position.x - player.pos.x;
      var dz = p.mesh.position.z - player.pos.z;
      if (dx * dx + dz * dz < 1.6) {
        if (p.kind === 'health') {
          player.hp = Math.min(cfg.player.hp, player.hp + cfg.pickups.health.amount);
        } else {
          var refill = cfg.pickups.ammo.reserveRefill;
          Object.keys(refill).forEach(function (id) { ES.weapons.addReserve(id, refill[id]); });
        }
        sound.pickup();
        scene.remove(p.mesh);
        pickups.splice(i, 1);
      }
    }
  }

  // --- game state -----------------------------------------------------------
  var state = 'menu';   // menu | playing | intermission | paused | gameover
  var diff = cfg.difficulties[1];
  var player = {
    pos: new THREE.Vector3(8, cfg.player.eyeHeight, 30),
    velY: 0, onGround: true,
    hp: cfg.player.hp,
    yaw: 0.26, pitch: 0,
    lastHurt: -99, shake: 0,
    roll: 0, bobT: 0
  };
  var wave = 0;
  var score = 0;
  var kills = 0;
  var spawnQueue = [];
  var spawnTimer = 0;
  var intermissionT = 0;
  var clockT = 0;

  var keys = {};
  var mouseDown = false;

  function bestKey() { return 'eloxal-strike-best-' + diff.id; }
  function loadBest() { return parseInt(localStorage.getItem(bestKey()) || '0', 10); }

  // --- input ----------------------------------------------------------------
  document.addEventListener('keydown', function (e) {
    keys[e.code] = true;
    if (state === 'playing') {
      if (e.code === 'KeyR') { ES.weapons.startReload(); }
      if (e.code === 'Digit1') { ES.weapons.switchTo(0); }
      if (e.code === 'Digit2') { ES.weapons.switchTo(1); }
      if (e.code === 'Digit3') { ES.weapons.switchTo(2); }
    }
  });
  document.addEventListener('keyup', function (e) { keys[e.code] = false; });

  document.addEventListener('mousemove', function (e) {
    if (document.pointerLockElement !== canvas) { return; }
    var sens = 0.0022;
    player.yaw -= e.movementX * sens;
    player.pitch -= e.movementY * sens;
    player.pitch = Math.max(-1.45, Math.min(1.45, player.pitch));
  });

  canvas.addEventListener('mousedown', function (e) {
    if (state !== 'playing' && state !== 'intermission') { return; }
    if (document.pointerLockElement !== canvas) {
      canvas.requestPointerLock();
      return;
    }
    if (e.button === 0) {
      mouseDown = true;
      attemptShot();
    } else if (e.button === 2) {
      ES.weapons.setAim(true);
    }
  });
  document.addEventListener('mouseup', function (e) {
    if (e.button === 0) { mouseDown = false; }
    if (e.button === 2) { ES.weapons.setAim(false); }
  });
  document.addEventListener('contextmenu', function (e) {
    if (state === 'playing' || state === 'intermission') { e.preventDefault(); }
  });
  document.addEventListener('wheel', function (e) {
    if (state === 'playing' && document.pointerLockElement === canvas) {
      ES.weapons.cycle(e.deltaY > 0 ? 1 : -1);
    }
  });

  var pausedFrom = 'playing';
  document.addEventListener('pointerlockchange', function () {
    if (document.pointerLockElement !== canvas &&
        (state === 'playing' || state === 'intermission')) {
      pausedFrom = state;
      setState('paused');
    }
  });

  // --- shooting -------------------------------------------------------------
  var raycaster = new THREE.Raycaster();

  /* Brass casing kicked out of the breech on every shot. */
  function ejectCasing() {
    var m = new THREE.Mesh(
      new THREE.BoxGeometry(0.032, 0.016, 0.016),
      new THREE.MeshBasicMaterial({ color: 0xC9A227 })
    );
    m.position.copy(camera.localToWorld(new THREE.Vector3(0.28, -0.14, -0.42)));
    m.rotation.set(Math.random() * 3, Math.random() * 3, 0);
    var right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion);
    var v = right.multiplyScalar(1.4 + Math.random())
      .add(new THREE.Vector3(0, 1.9 + Math.random() * 0.6, 0));
    particles.push({ mesh: m, vel: v, life: 0.9, gravity: 9 });
    scene.add(m);
  }

  function attemptShot() {
    var w = ES.weapons.tryFire();
    if (!w) { return; }
    player.pitch += (w.pellets > 1 ? 0.03 : 0.012);
    player.shake = Math.min(0.5, player.shake + (w.pellets > 1 ? 0.25 : 0.08));

    var origin = new THREE.Vector3();
    camera.getWorldPosition(origin);
    var baseDir = new THREE.Vector3();
    camera.getWorldDirection(baseDir);
    var muzzle = ES.weapons.muzzleWorld();
    ejectCasing();

    var targets = world.solids.concat(ES.enemies.hittables());
    var anyHit = false, anyKill = false, anyHead = false;

    for (var p = 0; p < w.pellets; p++) {
      var spread = THREE.MathUtils.degToRad(w.spreadDeg) * ES.weapons.spreadMult();
      var dir = baseDir.clone();
      dir.x += (Math.random() - 0.5) * spread * 2;
      dir.y += (Math.random() - 0.5) * spread * 2;
      dir.z += (Math.random() - 0.5) * spread * 2;
      dir.normalize();

      raycaster.set(origin, dir);
      raycaster.far = w.range;
      var hits = raycaster.intersectObjects(targets, false);
      var end = origin.clone().addScaledVector(dir, w.range);

      if (hits.length > 0) {
        var hit = hits[0];
        end = hit.point;
        var ud = hit.object.userData;
        if (ud && ud.enemy) {
          var e = ud.enemy;
          var dmg = cfg.falloff(w.damage, hit.distance, w.range);
          if (ud.isHead) { dmg *= w.headshotMult; anyHead = true; }
          anyHit = true;
          burst(hit.point, 0xc9552a, 4, 3, 6);
          if (ES.enemies.damage(e, dmg)) {
            anyKill = true;
            kills++;
            var pts = Math.round(e.cfg.score * diff.scoreMult) +
              (ud.isHead ? cfg.scoring.headshotBonus : 0);
            score += pts;
            burst(hit.point, 0xa8502a, 14, 5, 8);
            burst(hit.point, e.cfg.eyeColor, 6, 4, 5);
            ES.enemies.remove(scene, e);
          }
        } else {
          burst(hit.point, 0xffe9a8, 3, 2.5, 7);
        }
      }
      tracer(muzzle, end, w.color);
    }

    if (anyKill) { sound.kill(); showHitmarker(true); }
    else if (anyHit) { if (anyHead) { sound.headshot(); } else { sound.hit(); } showHitmarker(false); }
  }

  var hitmarkerT = null;
  function showHitmarker(kill) {
    ui.hitmarker.classList.remove('show', 'kill');
    void ui.hitmarker.offsetWidth; // restart the CSS animation
    ui.hitmarker.classList.add('show');
    if (kill) { ui.hitmarker.classList.add('kill'); }
    clearTimeout(hitmarkerT);
    hitmarkerT = setTimeout(function () {
      ui.hitmarker.classList.remove('show', 'kill');
    }, 220);
  }

  // --- waves ----------------------------------------------------------------
  function startWave(n) {
    wave = n;
    var comp = cfg.waveFor(n, diff.spawnMult);
    spawnQueue = [];
    Object.keys(comp).forEach(function (type) {
      for (var i = 0; i < comp[type]; i++) { spawnQueue.push(type); }
    });
    // shuffle, but push bosses to the end for drama
    spawnQueue.sort(function (a, b) {
      var ab = (a === 'korrosius') ? 1 : Math.random() - 0.5;
      var bb = (b === 'korrosius') ? 1 : Math.random() - 0.5;
      return ab - bb;
    });
    spawnTimer = 0.5;
    setState('playing');
    var boss = comp.korrosius > 0;
    showBanner('Welle ' + n, boss ? '⚠ Baron Korrosius nähert sich!' : cfg.totalEnemies(comp) + ' Gegner', 2.2);
    sound.waveStart();
  }

  function spawnFromQueue(dt) {
    if (spawnQueue.length === 0) { return; }
    spawnTimer -= dt;
    if (spawnTimer > 0) { return; }
    spawnTimer = 0.7;
    var type = spawnQueue.shift();
    // prefer a spawn point far away from the player
    var best = null, bestD = -1;
    world.spawnPoints.forEach(function (sp) {
      var dx = sp[0] - player.pos.x, dz = sp[1] - player.pos.z;
      var d = dx * dx + dz * dz + Math.random() * 300;
      if (d > bestD) { bestD = d; best = sp; }
    });
    burst(new THREE.Vector3(best[0], 1.0, best[1]), 0x9fe348, 12, 4, 3);
    ES.enemies.spawn(scene, type, best[0], best[1], diff.enemyHp, diff.enemySpeed);
  }

  function checkWaveEnd() {
    if (spawnQueue.length === 0 && ES.enemies.aliveCount() === 0) {
      score += cfg.scoring.waveClearBonus;
      sound.waveClear();
      setState('intermission');
      intermissionT = cfg.intermissionSec;
      showBanner('Welle ' + wave + ' geschafft!', '+' + cfg.scoring.waveClearBonus + ' Punkte — Verschnaufpause', 2.5);
      // supplies for the break
      var spots = world.pickupPoints.slice().sort(function () { return Math.random() - 0.5; });
      spawnPickup('health', spots[0][0], spots[0][1]);
      spawnPickup('ammo', spots[1][0], spots[1][1]);
      if (wave % 3 === 0) { spawnPickup('health', spots[2][0], spots[2][1]); }
    }
  }

  // --- banner ---------------------------------------------------------------
  var bannerTimer = null;
  function showBanner(title, sub, dur) {
    ui.bannerTitle.textContent = title;
    ui.bannerSub.textContent = sub || '';
    ui.banner.classList.add('show');
    clearTimeout(bannerTimer);
    bannerTimer = setTimeout(function () {
      ui.banner.classList.remove('show');
    }, (dur || 2) * 1000);
  }

  // --- player ---------------------------------------------------------------
  function hurtPlayer(dmg) {
    if (state !== 'playing' && state !== 'intermission') { return; }
    player.hp -= dmg;
    player.lastHurt = clockT;
    player.shake = Math.min(0.7, player.shake + 0.3);
    sound.hurt();
    if (player.hp <= 0) {
      player.hp = 0;
      gameOver();
    }
  }

  function updatePlayer(dt) {
    // movement input in camera space
    var f = (keys.KeyW ? 1 : 0) - (keys.KeyS ? 1 : 0);
    var r = (keys.KeyD ? 1 : 0) - (keys.KeyA ? 1 : 0);
    var sprinting = keys.ShiftLeft || keys.ShiftRight;
    var aimF = ES.weapons.aimFactor();
    var speed = cfg.player.speed * (sprinting ? cfg.player.sprintMult : 1) * (1 - 0.25 * aimF);
    var moving = (f !== 0 || r !== 0);

    var sin = Math.sin(player.yaw), cos = Math.cos(player.yaw);
    var vx = (-sin * f + cos * r) * speed;
    var vz = (-cos * f - sin * r) * speed;
    if (moving && f !== 0 && r !== 0) { vx *= 0.7071; vz *= 0.7071; }

    player.pos.x += vx * dt;
    player.pos.z += vz * dt;
    ES.world.collide(player.pos, cfg.player.radius, world.colliders, world.bounds);

    // jump & gravity
    if (keys.Space && player.onGround) {
      player.velY = cfg.player.jumpVel;
      player.onGround = false;
    }
    if (!player.onGround) {
      player.velY -= cfg.player.gravity * dt;
      player.pos.y += player.velY * dt;
      if (player.pos.y <= cfg.player.eyeHeight) {
        player.pos.y = cfg.player.eyeHeight;
        player.velY = 0;
        player.onGround = true;
      }
    }

    // regen
    if (diff.playerRegen && player.hp > 0 && player.hp < cfg.player.hp &&
        clockT - player.lastHurt > cfg.player.regenDelaySec) {
      player.hp = Math.min(cfg.player.hp, player.hp + cfg.player.regenPerSec * dt);
    }

    // camera: shake, head bob while running, slight roll when strafing
    player.shake = Math.max(0, player.shake - dt * 2.2);
    var shakeX = (Math.random() - 0.5) * player.shake * 0.05;
    var shakeY = (Math.random() - 0.5) * player.shake * 0.05;
    if (moving && player.onGround) { player.bobT += dt * (sprinting ? 11 : 8); }
    var headBob = Math.sin(player.bobT) * 0.028 * (moving && player.onGround ? 1 : 0) * (1 - 0.7 * aimF);
    var targetRoll = -r * 0.02 * (1 - 0.5 * aimF);
    player.roll += (targetRoll - player.roll) * Math.min(1, dt * 8);
    camera.position.set(player.pos.x + shakeX, player.pos.y + shakeY + headBob, player.pos.z);
    camera.rotation.set(player.pitch, player.yaw, player.roll);

    // FOV: sprint widens, aiming zooms in
    var baseFov = (sprinting && moving) ? 82 : 75;
    var targetFov = baseFov * (1 - aimF) + 58 * aimF;
    camera.fov += (targetFov - camera.fov) * Math.min(1, dt * 10);
    camera.updateProjectionMatrix();

    ES.weapons.update(dt, moving ? (sprinting ? 1 : 0.6) : 0);

    // auto weapons keep firing while the button is held
    if (mouseDown && ES.weapons.currentWeapon().auto) { attemptShot(); }
  }

  // --- HUD ------------------------------------------------------------------
  function updateHud() {
    var hpFrac = player.hp / cfg.player.hp;
    ui.hpFill.style.width = (hpFrac * 100).toFixed(1) + '%';
    ui.hpFill.classList.toggle('low', hpFrac < 0.35);
    ui.hpNum.textContent = Math.ceil(player.hp);

    var w = ES.weapons.currentWeapon();
    var s = ES.weapons.currentState();
    ui.weaponName.textContent = w.name + (s.reloading ? ' — lädt…' : '');
    ui.ammoMag.textContent = s.mag;
    ui.ammoReserve.textContent = (s.reserve === Infinity) ? '∞' : s.reserve;

    ui.crosshair.style.opacity = (1 - ES.weapons.aimFactor()).toFixed(2);

    ui.waveNum.textContent = wave;
    ui.scoreNum.textContent = score;
    ui.enemiesLeft.textContent = ES.enemies.aliveCount() + spawnQueue.length;

    // damage vignette: recent hit flash + permanent low-hp pulse
    var sinceHurt = clockT - player.lastHurt;
    var flash = Math.max(0, 1 - sinceHurt * 2.2) * 0.75;
    var lowHp = hpFrac < 0.3 ? (0.3 - hpFrac) * (1.4 + Math.sin(clockT * 5) * 0.5) : 0;
    ui.damageOverlay.style.opacity = Math.min(0.85, flash + lowHp).toFixed(3);
  }

  // --- state machine --------------------------------------------------------
  function setState(s) {
    state = s;
    ui.menu.classList.toggle('hidden', s !== 'menu');
    ui.hud.classList.toggle('hidden', s === 'menu' || s === 'gameover');
    ui.pause.classList.toggle('hidden', s !== 'paused');
    ui.gameover.classList.toggle('hidden', s !== 'gameover');
    document.body.classList.toggle('ingame', s === 'playing' || s === 'intermission');
  }

  function startGame() {
    ES.weapons.setAim(false);
    ES.enemies.clear(scene);
    clearPickups();
    ES.weapons.reset();
    player.pos.set(8, cfg.player.eyeHeight, 30);
    player.hp = cfg.player.hp;
    player.yaw = 0.26; player.pitch = 0;
    player.velY = 0; player.onGround = true;
    player.lastHurt = -99;
    score = 0; kills = 0;
    ui.diffLabel.textContent = diff.name;
    sound.unlock();
    sound.startDrone();
    canvas.requestPointerLock();
    startWave(1);
  }

  function gameOver() {
    setState('gameover');
    mouseDown = false;
    ES.weapons.setAim(false);
    sound.gameOver();
    sound.stopDrone();
    if (document.pointerLockElement === canvas) { document.exitPointerLock(); }
    ui.goScore.textContent = score;
    ui.goWave.textContent = wave;
    ui.goKills.textContent = kills;
    var best = loadBest();
    if (score > best) {
      best = score;
      localStorage.setItem(bestKey(), String(best));
      ui.goBest.textContent = 'Neuer Rekord: ' + best + ' Punkte (' + diff.name + ')';
    } else {
      ui.goBest.textContent = 'Rekord: ' + best + ' Punkte (' + diff.name + ')';
    }
  }

  // --- menu wiring ----------------------------------------------------------
  function refreshBestLabel() {
    var best = loadBest();
    ui.bestLabel.textContent = best > 0
      ? 'Rekord (' + diff.name + '): ' + best + ' Punkte'
      : 'Noch kein Rekord auf „' + diff.name + '“ — zeig, was du kannst!';
  }

  var cards = document.querySelectorAll('.diffcard');
  cards.forEach(function (card) {
    card.addEventListener('click', function () {
      cards.forEach(function (c) { c.classList.remove('selected'); });
      card.classList.add('selected');
      var id = card.getAttribute('data-diff');
      diff = cfg.difficulties.filter(function (d) { return d.id === id; })[0];
      refreshBestLabel();
      sound.unlock();
      sound.pickup();
    });
  });
  refreshBestLabel();

  $('btn-start').addEventListener('click', startGame);
  $('btn-resume').addEventListener('click', function () {
    setState(pausedFrom);
    canvas.requestPointerLock();
  });
  $('btn-pause-menu').addEventListener('click', function () {
    sound.stopDrone();
    setState('menu');
    refreshBestLabel();
  });
  $('btn-retry').addEventListener('click', startGame);
  $('btn-go-menu').addEventListener('click', function () {
    setState('menu');
    refreshBestLabel();
  });

  var muteBtn = $('btn-mute');
  function applyMute(m) {
    sound.setMuted(m);
    muteBtn.textContent = m ? '🔇 Ton aus' : '🔊 Ton an';
    localStorage.setItem('eloxal-strike-muted', m ? '1' : '0');
  }
  muteBtn.addEventListener('click', function () { applyMute(!sound.isMuted()); });
  if (localStorage.getItem('eloxal-strike-muted') === '1') { applyMute(true); }

  var fxBtn = $('btn-fx');
  function applyFx(on) {
    fxEnabled = on;
    fxBtn.textContent = on ? '\u2728 Effekte: An' : '\u2728 Effekte: Aus';
    localStorage.setItem('eloxal-strike-fx', on ? '1' : '0');
  }
  fxBtn.addEventListener('click', function () { applyFx(!fxEnabled); });
  applyFx(fxEnabled);

  // --- main loop ------------------------------------------------------------
  var last = performance.now();
  var menuAngle = 0;

  function frame(now) {
    requestAnimationFrame(frame);
    var dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    clockT += dt;

    world.update(dt, clockT);

    // crane trolley drifts along its bridge, in menu and in game
    var trolleyX = Math.sin(clockT * 0.15) * 26;
    world.crane.trolley.position.x = trolleyX;
    world.crane.cable.position.x = trolleyX;
    world.crane.load.position.x = trolleyX;

    if (state === 'menu') {
      // slow cinematic orbit around the hall behind the menu
      menuAngle += dt * 0.08;
      camera.position.set(Math.sin(menuAngle) * 24, 6.5, Math.cos(menuAngle) * 24);
      camera.lookAt(0, 2, 0);
      camera.fov = 65;
      camera.updateProjectionMatrix();
    } else if (state === 'playing' || state === 'intermission') {
      updatePlayer(dt);
      ES.enemies.update(dt, {
        scene: scene,
        playerPos: player.pos,
        colliders: world.colliders,
        bounds: world.bounds,
        dmgMult: diff.enemyDmg,
        onPlayerDamage: hurtPlayer,
        onSpit: sound.spit
      });
      updatePickups(dt);

      if (state === 'playing') {
        spawnFromQueue(dt);
        checkWaveEnd();
      } else {
        intermissionT -= dt;
        if (intermissionT <= 0) { startWave(wave + 1); }
        else if (intermissionT < 3.2) {
          ui.bannerTitle.textContent = 'Nächste Welle';
          ui.bannerSub.textContent = 'in ' + Math.ceil(intermissionT) + '…';
          ui.banner.classList.add('show');
        }
      }
      updateHud();
    }

    updateParticles(dt);
    updateTracers(dt);
    if (fxEnabled && composer) { composer.render(); }
    else { renderer.render(scene, camera); }
  }

  setState('menu');
  requestAnimationFrame(frame);
})();
