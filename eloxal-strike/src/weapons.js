/* Eloxal Strike — weapon handling: viewmodels built from primitives,
 * ammo/reload state, recoil, sway and muzzle flash. Ballistics numbers come
 * from ES.config.weapons; the actual hit raycast lives in main.js.
 */
(function () {
  'use strict';

  var cfg = window.ES.config;
  var sound = window.ES.sound;

  var camera = null;
  var rig = null;            // group attached to the camera
  var models = [];           // one viewmodel group per weapon
  var flashes = [];          // muzzle flash meshes per weapon
  var flashLight = null;

  var current = 0;
  var state = [];            // { mag, reserve, cd, reloading, reloadT }
  var recoil = 0;
  var swayT = 0;
  var switchAnim = 0;

  function buildViewmodel(w) {
    var g = new THREE.Group();
    var dark = new THREE.MeshStandardMaterial({ color: 0x454c58, roughness: 0.45, metalness: 0.75 });
    var alu = new THREE.MeshStandardMaterial({ color: 0x9aa7b4, roughness: 0.3, metalness: 0.9 });
    var accent = new THREE.MeshStandardMaterial({
      color: w.color, emissive: w.color, emissiveIntensity: 0.9, roughness: 0.4
    });

    var body = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.13, 0.42), dark);
    g.add(body);
    var grip = new THREE.Mesh(new THREE.BoxGeometry(0.07, 0.16, 0.09), dark);
    grip.position.set(0, -0.13, 0.12);
    grip.rotation.x = 0.35;
    g.add(grip);

    if (w.id === 'streuer') {
      var b1 = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.5, 10), alu);
      b1.rotation.x = Math.PI / 2;
      b1.position.set(-0.025, 0.02, -0.38);
      g.add(b1);
      var b2 = b1.clone();
      b2.position.x = 0.025;
      g.add(b2);
      var tank = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 0.18, 10), accent);
      tank.rotation.x = Math.PI / 2;
      tank.position.set(0, -0.06, -0.1);
      g.add(tank);
    } else if (w.id === 'lichtbogen') {
      var barrel = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.04, 0.55, 10), alu);
      barrel.rotation.x = Math.PI / 2;
      barrel.position.set(0, 0.02, -0.42);
      g.add(barrel);
      var coil;
      for (var i = 0; i < 3; i++) {
        coil = new THREE.Mesh(new THREE.TorusGeometry(0.055, 0.012, 8, 14), accent);
        coil.position.set(0, 0.02, -0.2 - i * 0.12);
        g.add(coil);
      }
      var drum = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, 0.08, 12), dark);
      drum.rotation.z = Math.PI / 2;
      drum.position.set(0, -0.1, -0.02);
      g.add(drum);
    } else {
      var slide = new THREE.Mesh(new THREE.BoxGeometry(0.075, 0.06, 0.34), alu);
      slide.position.set(0, 0.075, -0.05);
      g.add(slide);
      var muzzle = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 0.16, 10), alu);
      muzzle.rotation.x = Math.PI / 2;
      muzzle.position.set(0, 0.075, -0.3);
      g.add(muzzle);
      var cell = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.05, 0.1), accent);
      cell.position.set(0, 0.01, -0.16);
      g.add(cell);
    }

    // muzzle flash: two crossed glowing planes at the barrel tip
    var flashMat = new THREE.MeshBasicMaterial({
      color: 0xffe9a8, transparent: true, opacity: 0, side: THREE.DoubleSide,
      depthWrite: false
    });
    var flash = new THREE.Group();
    var f1 = new THREE.Mesh(new THREE.PlaneGeometry(0.22, 0.22), flashMat);
    var f2 = f1.clone();
    f2.rotation.z = Math.PI / 4;
    flash.add(f1); flash.add(f2);
    flash.position.set(0, 0.03, -0.62);
    g.add(flash);

    return { group: g, flash: flash, flashMat: flashMat };
  }

  var api = {
    init: function (cam) {
      camera = cam;
      rig = new THREE.Group();
      rig.position.set(0.3, -0.26, -0.6);
      rig.scale.set(0.72, 0.72, 0.72);
      camera.add(rig);

      var fill = new THREE.PointLight(0xbfd0e2, 0.5, 3);
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
      switchAnim = 0.25;
    },

    reset: function () {
      cfg.weapons.forEach(function (w, i) {
        state[i].mag = w.magSize;
        state[i].reserve = w.reserve;
        state[i].cd = 0;
        state[i].reloading = false;
      });
      api.switchTo(0);
    },

    currentWeapon: function () { return cfg.weapons[current]; },
    currentState: function () { return state[current]; },

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
      models[current].flashMat.opacity = 1;
      models[current].flash.rotation.z = Math.random() * Math.PI;
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

      // recoil decay + weapon sway/bob
      recoil = Math.max(0, recoil - dt * 4);
      if (switchAnim > 0) { switchAnim -= dt; }
      swayT += dt * (2 + moveAmount * 6);
      var bobX = Math.sin(swayT) * 0.006 * (0.3 + moveAmount);
      var bobY = Math.abs(Math.cos(swayT)) * 0.008 * (0.3 + moveAmount);
      var raise = Math.max(0, switchAnim / 0.25);
      rig.position.set(0.3 + bobX, -0.26 - bobY - raise * 0.25, -0.6 + recoil * 0.06);
      rig.rotation.x = recoil * 0.12 - raise * 0.5;
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
