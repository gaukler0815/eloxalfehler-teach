/* Eloxal Strike — weapon handling: detailed viewmodels with visible arms,
 * ammo/reload state, recoil, sway, aim-down-sights and muzzle flash.
 * Ballistics numbers come from ES.config.weapons; the hit raycast lives in
 * main.js.
 */
(function () {
  'use strict';

  var cfg = window.ES.config;
  var sound = window.ES.sound;

  var camera = null;
  var rig = null;            // group attached to the camera
  var models = [];           // one viewmodel entry per weapon
  var flashLight = null;

  var current = 0;
  var state = [];            // { mag, reserve, cd, reloading, reloadT }
  var recoil = 0;
  var swayT = 0;
  var switchAnim = 0;
  var aiming = false;
  var aimF = 0;              // smoothed 0..1 aim factor

  var HIP_POS = new THREE.Vector3(0.3, -0.26, -0.6);

  function lin(hex) { return new THREE.Color(hex).convertSRGBToLinear(); }

  function mats(w) {
    return {
      metal: new THREE.MeshStandardMaterial({ color: lin(0x1e2126), roughness: 0.35, metalness: 0.9 }),
      metal2: new THREE.MeshStandardMaterial({ color: lin(0x363b42), roughness: 0.45, metalness: 0.85 }),
      polymer: new THREE.MeshStandardMaterial({ color: lin(0x24262a), roughness: 0.75, metalness: 0.15 }),
      grip: new THREE.MeshStandardMaterial({ color: lin(0x2a2622), roughness: 0.9, metalness: 0.05 }),
      glove: new THREE.MeshStandardMaterial({ color: lin(0x2c2824), roughness: 0.95, metalness: 0 }),
      sleeve: new THREE.MeshStandardMaterial({ color: lin(0x2b2f36), roughness: 1, metalness: 0 }),
      band: new THREE.MeshStandardMaterial({ color: lin(0x6a7d5a), roughness: 0.8, metalness: 0.1 }),
      accent: new THREE.MeshStandardMaterial({
        color: lin(w.color), emissive: new THREE.Color(w.color), emissiveIntensity: 0.3, roughness: 0.4
      })
    };
  }

  function box(g, mat, w, h, d, x, y, z, rx, ry, rz) {
    var m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
    m.position.set(x, y, z);
    if (rx) { m.rotation.x = rx; }
    if (ry) { m.rotation.y = ry; }
    if (rz) { m.rotation.z = rz; }
    g.add(m);
    return m;
  }

  function cyl(g, mat, r1, r2, len, x, y, z, rx, ry, rz) {
    var m = new THREE.Mesh(new THREE.CylinderGeometry(r1, r2, len, 12), mat);
    m.position.set(x, y, z);
    m.rotation.x = (rx === undefined) ? Math.PI / 2 : rx;
    if (ry) { m.rotation.y = ry; }
    if (rz) { m.rotation.z = rz; }
    g.add(m);
    return m;
  }

  /* Gloved right hand on the pistol grip + forearm sleeve coming up from the
   * lower screen edge, wristwatch included (every operator needs one). */
  function rightArm(g, m, gx, gy, gz, gripTilt) {
    box(g, m.glove, 0.075, 0.1, 0.09, gx + 0.012, gy, gz, gripTilt);           // palm
    for (var f = 0; f < 3; f++) {                                              // fingers
      box(g, m.glove, 0.07, 0.022, 0.024, gx - 0.002, gy + 0.028 - f * 0.034,
        gz - 0.052, gripTilt);
    }
    box(g, m.glove, 0.026, 0.05, 0.03, gx - 0.038, gy + 0.03, gz - 0.02, gripTilt); // thumb
    var arm = cyl(g, m.sleeve, 0.046, 0.056, 0.4, gx + 0.06, gy - 0.21, gz + 0.15, -1.3, 0, -0.2);
    cyl(g, m.glove, 0.045, 0.048, 0.07, gx + 0.028, gy - 0.07, gz + 0.05, -1.3, 0, -0.2); // wrist
    cyl(g, m.band, 0.05, 0.05, 0.025, gx + 0.04, gy - 0.115, gz + 0.085, -1.3, 0, -0.2);    // watch
    return arm;
  }

  /* Gloved left hand wrapped over a foregrip/pump + sleeve from lower left. */
  function leftArm(g, m, gx, gy, gz) {
    box(g, m.glove, 0.085, 0.055, 0.12, gx, gy, gz);                           // palm over grip
    for (var f = 0; f < 3; f++) {                                              // fingers curling right
      box(g, m.glove, 0.024, 0.05, 0.026, gx + 0.038, gy - 0.02, gz - 0.045 + f * 0.042);
    }
    cyl(g, m.sleeve, 0.045, 0.054, 0.38, gx - 0.09, gy - 0.22, gz + 0.1, -1.4, 0, 0.45);
    cyl(g, m.glove, 0.042, 0.045, 0.06, gx - 0.03, gy - 0.06, gz + 0.03, -1.4, 0, 0.45);
  }

  function muzzleFlash(g, z, size) {
    var flashMat = new THREE.MeshBasicMaterial({
      color: 0xffe9a8, transparent: true, opacity: 0, side: THREE.DoubleSide,
      depthWrite: false, blending: THREE.AdditiveBlending
    });
    var flash = new THREE.Group();
    var f1 = new THREE.Mesh(new THREE.PlaneGeometry(size, size), flashMat);
    var f2 = f1.clone();
    f2.rotation.z = Math.PI / 4;
    var f3 = new THREE.Mesh(new THREE.PlaneGeometry(size, size), flashMat);
    f3.rotation.y = Math.PI / 2;
    flash.add(f1); flash.add(f2); flash.add(f3);
    flash.position.set(0, 0.03, z);
    g.add(flash);
    return { flash: flash, flashMat: flashMat };
  }

  /* --- the three guns ----------------------------------------------------- */

  function buildPistol(w) {
    var g = new THREE.Group();
    var m = mats(w);

    box(g, m.polymer, 0.07, 0.09, 0.3, 0, -0.01, -0.02);                 // frame
    box(g, m.metal, 0.075, 0.07, 0.36, 0, 0.065, -0.05);                 // slide
    for (var i = 0; i < 4; i++) {                                        // serrations
      box(g, m.metal2, 0.078, 0.05, 0.008, 0, 0.065, 0.07 + i * 0.016);
    }
    cyl(g, m.metal2, 0.02, 0.02, 0.06, 0, 0.068, -0.26);                 // barrel tip
    box(g, m.metal, 0.02, 0.008, 0.08, 0, -0.062, -0.1);                 // trigger guard bottom
    box(g, m.metal, 0.02, 0.05, 0.008, 0, -0.038, -0.135);               // trigger guard front
    box(g, m.metal2, 0.008, 0.018, 0.012, 0, -0.04, -0.075, 0.3);        // trigger
    box(g, m.metal2, 0.01, 0.02, 0.01, -0.014, 0.11, 0.115);             // rear sight L
    box(g, m.metal2, 0.01, 0.02, 0.01, 0.014, 0.11, 0.115);              // rear sight R
    box(g, m.metal2, 0.008, 0.022, 0.01, 0, 0.111, -0.21);               // front sight
    box(g, m.accent, 0.05, 0.035, 0.09, 0, 0.0, -0.13);                  // glowing cell
    box(g, m.grip, 0.065, 0.17, 0.095, 0, -0.115, 0.1, 0.3);             // grip
    box(g, m.metal2, 0.02, 0.03, 0.02, 0, 0.075, 0.13);                  // hammer

    rightArm(g, m, 0, -0.12, 0.1, 0.3);
    // support hand wrapped from the left
    box(g, m.glove, 0.03, 0.09, 0.09, -0.05, -0.13, 0.075, 0.3);
    var fl = muzzleFlash(g, -0.34, 0.2);
    fl.flash.position.y = 0.068;
    return { group: g, flash: fl.flash, flashMat: fl.flashMat,
      adsPos: new THREE.Vector3(0, -0.074, -0.42) };
  }

  function buildShotgun(w) {
    var g = new THREE.Group();
    var m = mats(w);

    box(g, m.metal, 0.08, 0.11, 0.34, 0, 0, 0.02);                        // receiver
    cyl(g, m.metal, 0.031, 0.031, 0.58, -0.024, 0.035, -0.4);             // barrel L
    cyl(g, m.metal, 0.031, 0.031, 0.58, 0.024, 0.035, -0.4);              // barrel R
    box(g, m.metal2, 0.012, 0.01, 0.5, 0, 0.075, -0.38);                  // top rib
    box(g, m.metal2, 0.01, 0.014, 0.014, 0, 0.086, -0.62);                // bead sight
    box(g, m.polymer, 0.075, 0.06, 0.17, 0, -0.05, -0.3);                 // pump
    for (var i = 0; i < 3; i++) {
      box(g, m.grip, 0.08, 0.008, 0.02, 0, -0.028, -0.36 + i * 0.055);
    }
    box(g, m.grip, 0.062, 0.1, 0.24, 0, -0.045, 0.26, 0.14);              // stock
    box(g, m.grip, 0.065, 0.13, 0.09, 0, -0.1, 0.13, 0.35);               // pistol grip
    cyl(g, m.accent, 0.045, 0.045, 0.16, 0, -0.085, -0.08);               // acid tank
    cyl(g, m.metal2, 0.016, 0.016, 0.06, 0.048, 0.03, 0.1);               // spare shell 1
    cyl(g, m.accent, 0.016, 0.016, 0.06, 0.048, -0.005, 0.1);             // spare shell 2

    rightArm(g, m, 0, -0.105, 0.13, 0.35);
    leftArm(g, m, 0, -0.075, -0.3);
    var fl = muzzleFlash(g, -0.72, 0.3);
    fl.flash.position.y = 0.035;
    return { group: g, flash: fl.flash, flashMat: fl.flashMat,
      adsPos: new THREE.Vector3(0, -0.058, -0.4) };
  }

  function buildLMG(w) {
    var g = new THREE.Group();
    var m = mats(w);

    box(g, m.metal, 0.085, 0.12, 0.52, 0, 0.01, -0.05);                   // receiver
    box(g, m.metal2, 0.06, 0.018, 0.44, 0, 0.088, -0.06);                 // top rail
    for (var i = 0; i < 6; i++) {                                         // rail notches
      box(g, m.metal, 0.062, 0.02, 0.012, 0, 0.088, -0.24 + i * 0.07);
    }
    // ring red-dot sight
    var ring = new THREE.Mesh(new THREE.TorusGeometry(0.034, 0.008, 8, 18), m.metal2);
    ring.position.set(0, 0.128, 0.05);
    g.add(ring);
    var dot = new THREE.Mesh(new THREE.SphereGeometry(0.006, 6, 6),
      new THREE.MeshBasicMaterial({ color: 0xff3b30 }));
    dot.position.set(0, 0.128, 0.05);
    g.add(dot);
    box(g, m.metal2, 0.02, 0.03, 0.03, 0, 0.104, 0.05);                   // sight base
    cyl(g, m.metal, 0.024, 0.028, 0.5, 0, 0.03, -0.56);                   // barrel
    box(g, m.metal2, 0.05, 0.05, 0.09, 0, 0.03, -0.83);                   // muzzle brake
    for (var c = 0; c < 3; c++) {                                         // arc coils
      var coil = new THREE.Mesh(new THREE.TorusGeometry(0.042, 0.009, 8, 14), m.accent);
      coil.position.set(0, 0.03, -0.46 - c * 0.09);
      g.add(coil);
    }
    cyl(g, m.polymer, 0.075, 0.075, 0.09, 0, -0.1, -0.02, Math.PI / 2, 0, Math.PI / 2); // drum
    box(g, m.accent, 0.094, 0.02, 0.02, 0, -0.1, -0.02);                  // drum charge strip
    cyl(g, m.metal2, 0.028, 0.028, 0.18, 0, 0.02, 0.26, 0);               // buffer tube
    box(g, m.polymer, 0.055, 0.14, 0.07, 0, -0.005, 0.37);                // stock plate
    box(g, m.grip, 0.06, 0.14, 0.08, 0, -0.12, 0.12, 0.25);               // pistol grip
    box(g, m.polymer, 0.05, 0.11, 0.06, 0, -0.09, -0.36, -0.15);          // front grip

    rightArm(g, m, 0, -0.125, 0.12, 0.25);
    leftArm(g, m, 0, -0.1, -0.36);
    var fl = muzzleFlash(g, -0.9, 0.24);
    return { group: g, flash: fl.flash, flashMat: fl.flashMat,
      adsPos: new THREE.Vector3(0, -0.092, -0.4) };
  }

  function buildViewmodel(w) {
    if (w.id === 'streuer') { return buildShotgun(w); }
    if (w.id === 'lichtbogen') { return buildLMG(w); }
    return buildPistol(w);
  }

  var api = {
    init: function (cam) {
      camera = cam;
      rig = new THREE.Group();
      rig.position.copy(HIP_POS);
      rig.scale.set(0.72, 0.72, 0.72);
      camera.add(rig);

      var fill = new THREE.PointLight(0xbfd0e2, 0.32, 3);
      fill.position.set(-0.3, 0.4, 0.2);
      camera.add(fill);

      flashLight = new THREE.PointLight(0xffd166, 0, 8, 2);
      flashLight.position.set(0.3, -0.1, -1);
      camera.add(flashLight);

      models = [];
      state = [];
      cfg.weapons.forEach(function (w) {
        var vm = buildViewmodel(w);
        vm.group.visible = false;
        rig.add(vm.group);
        models.push(vm);
        state.push({ mag: w.magSize, reserve: w.reserve, cd: 0, reloading: false, reloadT: 0 });
      });
      current = 0;
      models[0].group.visible = true;
      recoil = 0;
      aiming = false;
      aimF = 0;
      switchAnim = 0.25;
    },

    reset: function () {
      cfg.weapons.forEach(function (w, i) {
        state[i].mag = w.magSize;
        state[i].reserve = w.reserve;
        state[i].cd = 0;
        state[i].reloading = false;
      });
      aiming = false;
      aimF = 0;
      api.switchTo(0);
    },

    currentWeapon: function () { return cfg.weapons[current]; },
    currentState: function () { return state[current]; },

    setAim: function (a) { aiming = a; },
    aimFactor: function () { return aimF; },
    spreadMult: function () { return 1 - 0.65 * aimF; },

    /* World position of the current muzzle (for tracers). */
    muzzleWorld: function (target) {
      return models[current].flash.getWorldPosition(target || new THREE.Vector3());
    },

    switchTo: function (i) {
      if (i < 0 || i >= cfg.weapons.length || i === current) { return; }
      models[current].group.visible = false;
      current = i;
      models[current].group.visible = true;
      state[current].reloading = false;
      state[current].cd = Math.max(state[current].cd, 0.18);
      switchAnim = 0.25;
    },

    cycle: function (dir) {
      api.switchTo((current + dir + cfg.weapons.length) % cfg.weapons.length);
    },

    startReload: function () {
      var w = cfg.weapons[current];
      var s = state[current];
      if (s.reloading || s.mag >= w.magSize || s.reserve <= 0) { return; }
      s.reloading = true;
      s.reloadT = w.reloadSec;
      sound.reload();
    },

    /* Attempt to fire. Returns the weapon config when a shot goes out,
     * null otherwise. Ammo, cooldown, recoil and flash handled here. */
    tryFire: function () {
      var w = cfg.weapons[current];
      var s = state[current];
      if (s.reloading || s.cd > 0) { return null; }
      if (s.mag <= 0) {
        sound.empty();
        api.startReload();
        s.cd = 0.3;
        return null;
      }
      s.mag--;
      s.cd = w.fireDelay;
      recoil = Math.min(1, recoil + (w.pellets > 1 ? 0.9 : 0.45));
      var vm = models[current];
      vm.flashMat.opacity = 1;
      vm.flash.rotation.z = Math.random() * Math.PI;
      vm.flash.scale.setScalar(0.8 + Math.random() * 0.5);
      flashLight.intensity = 2.2;
      sound.shoot(w.id);
      return w;
    },

    addReserve: function (weaponId, amount) {
      for (var i = 0; i < cfg.weapons.length; i++) {
        if (cfg.weapons[i].id === weaponId && state[i].reserve !== Infinity) {
          state[i].reserve += amount;
        }
      }
    },

    update: function (dt, moveAmount) {
      var w = cfg.weapons[current];
      var s = state[current];
      if (s.cd > 0) { s.cd -= dt; }
      if (s.reloading) {
        s.reloadT -= dt;
        if (s.reloadT <= 0) {
          var want = w.magSize - s.mag;
          var take = (s.reserve === Infinity) ? want : Math.min(want, s.reserve);
          if (s.reserve !== Infinity) { s.reserve -= take; }
          s.mag += take;
          s.reloading = false;
        }
      }

      // smooth aim in/out; no aiming while reloading or switching
      var wantAim = (aiming && !s.reloading && switchAnim <= 0) ? 1 : 0;
      aimF += (wantAim - aimF) * Math.min(1, dt * 10);

      // recoil decay + weapon sway/bob (sway calms down when aiming)
      recoil = Math.max(0, recoil - dt * 4);
      if (switchAnim > 0) { switchAnim -= dt; }
      swayT += dt * (2 + moveAmount * 6);
      var swayScale = 1 - 0.85 * aimF;
      var bobX = Math.sin(swayT) * 0.006 * (0.3 + moveAmount) * swayScale;
      var bobY = Math.abs(Math.cos(swayT)) * 0.008 * (0.3 + moveAmount) * swayScale;
      var raise = Math.max(0, switchAnim / 0.25);
      var ads = models[current].adsPos;
      var px = HIP_POS.x * (1 - aimF) + ads.x * aimF;
      var py = HIP_POS.y * (1 - aimF) + ads.y * aimF;
      var pz = HIP_POS.z * (1 - aimF) + ads.z * aimF;
      var recoilZ = recoil * 0.06 * (1 - 0.4 * aimF);
      rig.position.set(px + bobX, py - bobY - raise * 0.25, pz + recoilZ);
      rig.rotation.x = recoil * 0.12 * (1 - 0.4 * aimF) - raise * 0.5;
      rig.rotation.y = -0.05 * (1 - aimF);
      var reloadDip = s.reloading ? Math.sin((1 - s.reloadT / w.reloadSec) * Math.PI) : 0;
      rig.rotation.z = reloadDip * 0.5;
      rig.position.y -= reloadDip * 0.15;

      // fade the muzzle flash
      models.forEach(function (vm) {
        if (vm.flashMat.opacity > 0) {
          vm.flashMat.opacity = Math.max(0, vm.flashMat.opacity - dt * 14);
        }
      });
      if (flashLight.intensity > 0) {
        flashLight.intensity = Math.max(0, flashLight.intensity - dt * 18);
      }
    },

    /* Camera pitch kick for main.js to apply this frame. */
    recoilKick: function () { return recoil; }
  };

  window.ES = window.ES || {};
  window.ES.weapons = api;
})();
