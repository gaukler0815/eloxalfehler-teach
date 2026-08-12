/* Eloxal Strike — the Korrosionsbande. Procedurally built rust creatures,
 * their chase/attack AI and the acid projectiles of the Säure-Sprüher.
 * Rendering-independent numbers all come from ES.config.enemies.
 */
(function () {
  'use strict';

  var cfg = window.ES.config;
  var enemies = [];
  var spits = [];
  var geoCache = {};

  /* Lumpy rust blob: sphere with jittered vertices, flat-shaded. */
  function blobGeometry(radius, seed) {
    var key = radius.toFixed(2) + '_' + seed;
    if (geoCache[key]) { return geoCache[key]; }
    var geo = new THREE.SphereGeometry(radius, 10, 8);
    var p = geo.attributes.position;
    var s = seed;
    function rnd() { s = (s * 16807) % 2147483647; return s / 2147483647; }
    for (var i = 0; i < p.count; i++) {
      var scale = 1 + (rnd() - 0.5) * 0.36;
      p.setXYZ(i, p.getX(i) * scale, p.getY(i) * scale, p.getZ(i) * scale);
    }
    geo.computeVertexNormals();
    geoCache[key] = geo;
    return geo;
  }

  function buildMesh(type) {
    var c = cfg.enemies[type];
    var group = new THREE.Group();
    var lin = function (hex) { return new THREE.Color(hex).convertSRGBToLinear(); };

    var bodyMat = new THREE.MeshStandardMaterial({
      color: lin(c.color), roughness: 0.95, metalness: 0.1, flatShading: true
    });
    var darkMat = new THREE.MeshStandardMaterial({
      color: lin(c.color).multiplyScalar(0.55), roughness: 0.9, metalness: 0.2,
      flatShading: true
    });
    var plateMat = new THREE.MeshStandardMaterial({
      color: lin(0x3d4450), roughness: 0.5, metalness: 0.85, flatShading: true
    });
    var glowMat = new THREE.MeshStandardMaterial({
      color: lin(c.eyeColor), emissive: new THREE.Color(c.eyeColor),
      emissiveIntensity: 0.85, roughness: 0.6
    });

    var bodyR = c.radius;
    var bodyY = c.height * 0.42;

    // main mass + two side lumps so the silhouette isn't a plain ball
    var body = new THREE.Mesh(blobGeometry(bodyR, 7 + bodyR * 100), bodyMat);
    body.position.y = bodyY;
    body.castShadow = true;
    group.add(body);
    [-1, 1].forEach(function (side) {
      var lump = new THREE.Mesh(blobGeometry(bodyR * 0.5, 17 + bodyR * 40), darkMat);
      lump.position.set(side * bodyR * 0.75, bodyY - bodyR * 0.25, -bodyR * 0.25);
      lump.castShadow = true;
      group.add(lump);
    });

    // corroded armor plates grown into the rust
    for (var pl = 0; pl < 3; pl++) {
      var plate = new THREE.Mesh(
        new THREE.BoxGeometry(bodyR * 0.55, bodyR * 0.45, bodyR * 0.1), plateMat);
      var pa = -0.7 + pl * 0.7;
      plate.position.set(Math.sin(pa) * bodyR * 0.85, bodyY + bodyR * 0.35,
        -Math.cos(pa) * bodyR * 0.7);
      plate.rotation.set(0.5, pa, 0.15 * pl);
      group.add(plate);
    }

    // glowing cracks in the crust
    [[0.35, 0.5, -0.75, 0.5], [-0.55, 0.1, -0.7, -0.3]].forEach(function (cr) {
      var crack = new THREE.Mesh(
        new THREE.BoxGeometry(bodyR * 0.5, bodyR * 0.06, bodyR * 0.06), glowMat);
      crack.position.set(cr[0] * bodyR, bodyY + cr[1] * bodyR, cr[2] * bodyR);
      crack.rotation.z = cr[3];
      group.add(crack);
    });

    // stubby feet
    [-1, 1].forEach(function (side) {
      var foot = new THREE.Mesh(blobGeometry(bodyR * 0.32, 23), darkMat);
      foot.position.set(side * bodyR * 0.45, bodyR * 0.18, bodyR * 0.15);
      group.add(foot);
    });

    // head with brow plate, eyes with pupils, jagged mouth
    var headR = bodyR * 0.55;
    var headY = bodyY + bodyR * 0.9 + headR * 0.4;
    var head = new THREE.Mesh(blobGeometry(headR, 13 + bodyR * 50), bodyMat);
    head.position.y = headY;
    head.castShadow = true;
    group.add(head);
    var brow = new THREE.Mesh(
      new THREE.BoxGeometry(headR * 1.5, headR * 0.35, headR * 0.5), plateMat);
    brow.position.set(0, headY + headR * 0.55, headR * 0.35);
    brow.rotation.x = -0.25;
    group.add(brow);

    var eyeMat = new THREE.MeshBasicMaterial({ color: c.eyeColor });
    var pupilMat = new THREE.MeshBasicMaterial({ color: 0x17131F });
    [-1, 1].forEach(function (side) {
      var eye = new THREE.Mesh(new THREE.SphereGeometry(headR * 0.24, 8, 6), eyeMat);
      eye.position.set(side * headR * 0.45, headY + headR * 0.12, headR * 0.78);
      group.add(eye);
      var pupil = new THREE.Mesh(new THREE.SphereGeometry(headR * 0.1, 6, 5), pupilMat);
      pupil.position.set(side * headR * 0.45, headY + headR * 0.12, headR * 0.97);
      group.add(pupil);
    });

    var mouth = new THREE.Mesh(
      new THREE.BoxGeometry(headR * 0.8, headR * 0.16, headR * 0.2), pupilMat);
    mouth.position.set(0, headY - headR * 0.4, headR * 0.72);
    group.add(mouth);
    for (var th = 0; th < 3; th++) {
      var tooth = new THREE.Mesh(
        new THREE.ConeGeometry(headR * 0.06, headR * 0.14, 4), plateMat);
      tooth.position.set((th - 1) * headR * 0.25, headY - headR * 0.32, headR * 0.78);
      tooth.rotation.x = Math.PI;
      group.add(tooth);
    }

    if (c.ranged) {
      // acid sac on the back + spit nozzle
      var sac = new THREE.Mesh(blobGeometry(bodyR * 0.55, 31), glowMat);
      sac.position.set(0, bodyY + bodyR * 0.5, -bodyR * 0.85);
      group.add(sac);
      var nozzle = new THREE.Mesh(
        new THREE.ConeGeometry(headR * 0.18, headR * 0.5, 8), darkMat);
      nozzle.position.set(0, headY - headR * 0.4, headR * 0.95);
      nozzle.rotation.x = Math.PI / 2;
      group.add(nozzle);
    }

    if (type === 'brocken' || c.boss) {
      // shoulder spikes for the heavies
      [-1, 1].forEach(function (side) {
        var spike = new THREE.Mesh(
          new THREE.ConeGeometry(bodyR * 0.18, bodyR * 0.55, 5), plateMat);
        spike.position.set(side * bodyR * 0.85, bodyY + bodyR * 0.75, 0);
        spike.rotation.z = -side * 0.5;
        group.add(spike);
      });
    }

    if (c.boss) {
      var crownMat = new THREE.MeshStandardMaterial({
        color: lin(0xE8A33D), metalness: 0.9, roughness: 0.3
      });
      for (var k = 0; k < 5; k++) {
        var spike2 = new THREE.Mesh(new THREE.ConeGeometry(0.16, 0.7, 5), crownMat);
        var a2 = (k / 5) * Math.PI * 2;
        spike2.position.set(Math.cos(a2) * headR * 0.7,
          headY + headR * 0.95, Math.sin(a2) * headR * 0.7);
        group.add(spike2);
      }
      // smoldering core in the chest
      var core = new THREE.Mesh(new THREE.SphereGeometry(bodyR * 0.28, 8, 6),
        new THREE.MeshBasicMaterial({ color: 0xff2e2e }));
      core.position.set(0, bodyY + bodyR * 0.2, bodyR * 0.82);
      group.add(core);
    }

    // arms: two stubby lumps
    [-1, 1].forEach(function (side) {
      var arm = new THREE.Mesh(blobGeometry(bodyR * 0.35, 29), bodyMat);
      arm.position.set(side * bodyR * 1.05, bodyY, bodyR * 0.2);
      group.add(arm);
    });

    return { group: group, body: body, head: head };
  }

  var api = {
    spawn: function (scene, type, x, z, hpMult, speedMult) {
      var c = cfg.enemies[type];
      var parts = buildMesh(type);
      parts.group.position.set(x, 0, z);
      scene.add(parts.group);
      var e = {
        type: type, cfg: c,
        hp: c.hp * (hpMult || 1),
        maxHp: c.hp * (hpMult || 1),
        speed: c.speed * (speedMult || 1),
        group: parts.group, body: parts.body, head: parts.head,
        attackCd: 0.8 + Math.random() * 0.8,
        hitFlash: 0,
        bobPhase: Math.random() * Math.PI * 2,
        alive: true
      };
      parts.body.userData = { enemy: e, isHead: false };
      parts.head.userData = { enemy: e, isHead: true };
      enemies.push(e);
      return e;
    },

    list: function () { return enemies; },

    aliveCount: function () {
      var n = 0;
      for (var i = 0; i < enemies.length; i++) { if (enemies[i].alive) { n++; } }
      return n;
    },

    hittables: function () {
      var out = [];
      for (var i = 0; i < enemies.length; i++) {
        if (enemies[i].alive) { out.push(enemies[i].body, enemies[i].head); }
      }
      return out;
    },

    /* Apply damage; returns true if the enemy died from it. */
    damage: function (e, amount) {
      if (!e.alive) { return false; }
      e.hp -= amount;
      e.hitFlash = 0.12;
      if (e.hp <= 0) {
        e.alive = false;
        return true;
      }
      return false;
    },

    remove: function (scene, e) {
      scene.remove(e.group);
      var i = enemies.indexOf(e);
      if (i >= 0) { enemies.splice(i, 1); }
    },

    clear: function (scene) {
      enemies.forEach(function (e) { scene.remove(e.group); });
      enemies = [];
      spits.forEach(function (s) { scene.remove(s.mesh); });
      spits = [];
    },

    /* ctx: { scene, playerPos (THREE.Vector3), colliders, bounds,
     *        dmgMult, onPlayerDamage(dmg), onSpit() } */
    update: function (dt, ctx) {
      var pp = ctx.playerPos;

      for (var i = 0; i < enemies.length; i++) {
        var e = enemies[i];
        if (!e.alive) { continue; }
        var g = e.group;
        var dx = pp.x - g.position.x;
        var dz = pp.z - g.position.z;
        var dist = Math.sqrt(dx * dx + dz * dz) || 0.001;

        // face the player (yaw only)
        g.rotation.y = Math.atan2(dx, dz);

        var wantRange = e.cfg.ranged ? e.cfg.attackRange * 0.75 : e.cfg.attackRange * 0.7;
        if (dist > wantRange) {
          var step = e.speed * dt;
          g.position.x += dx / dist * step;
          g.position.z += dz / dist * step;
        }

        // separation from other enemies (cheap pairwise push)
        for (var j = i + 1; j < enemies.length; j++) {
          var o = enemies[j];
          if (!o.alive) { continue; }
          var sx = g.position.x - o.group.position.x;
          var sz = g.position.z - o.group.position.z;
          var minD = e.cfg.radius + o.cfg.radius;
          var sd2 = sx * sx + sz * sz;
          if (sd2 > 0.0001 && sd2 < minD * minD) {
            var sd = Math.sqrt(sd2);
            var push = (minD - sd) * 0.5;
            g.position.x += sx / sd * push;
            g.position.z += sz / sd * push;
            o.group.position.x -= sx / sd * push;
            o.group.position.z -= sz / sd * push;
          }
        }

        ES.world.collide(g.position, e.cfg.radius, ctx.colliders, ctx.bounds);

        // waddle animation
        e.bobPhase += dt * (4 + e.speed);
        var bob = Math.sin(e.bobPhase);
        g.position.y = Math.abs(bob) * 0.12;
        g.rotation.z = bob * 0.07;

        // hit flash + scale punch; the boss smolders even when untouched
        if (e.hitFlash > 0) {
          e.hitFlash -= dt;
          e.body.material.emissive.setHex(0xffffff);
          e.body.material.emissiveIntensity = 0.8;
        } else if (e.cfg.boss) {
          e.body.material.emissive.setHex(0xff2e2e);
          e.body.material.emissiveIntensity = 0.15 + Math.sin(e.bobPhase * 0.7) * 0.1;
        } else if (e.body.material.emissiveIntensity !== 0) {
          e.body.material.emissiveIntensity = 0;
        }
        var punch = 1 + Math.max(0, e.hitFlash) * 1.2;
        g.scale.set(punch, 1 / punch, punch);

        // attacks
        e.attackCd -= dt;
        if (e.attackCd <= 0 && dist <= e.cfg.attackRange) {
          e.attackCd = e.cfg.attackDelay;
          if (e.cfg.ranged) {
            var origin = g.position.clone();
            origin.y = e.cfg.height * 0.75;
            var dir = new THREE.Vector3(pp.x - origin.x, (pp.y - 0.2) - origin.y, pp.z - origin.z).normalize();
            var mesh = new THREE.Mesh(
              new THREE.SphereGeometry(0.18, 8, 6),
              new THREE.MeshBasicMaterial({ color: 0xb7ff4d })
            );
            mesh.position.copy(origin);
            ctx.scene.add(mesh);
            spits.push({
              mesh: mesh,
              vel: dir.multiplyScalar(e.cfg.projectileSpeed),
              dmg: e.cfg.damage * ctx.dmgMult,
              life: 4
            });
            if (ctx.onSpit) { ctx.onSpit(); }
          } else {
            // melee lunge
            ctx.onPlayerDamage(e.cfg.damage * ctx.dmgMult);
            e.bobPhase += Math.PI / 2;
          }
        }
      }

      // acid projectiles
      for (var k = spits.length - 1; k >= 0; k--) {
        var s = spits[k];
        s.vel.y -= 4 * dt; // light gravity arc
        s.mesh.position.addScaledVector(s.vel, dt);
        s.life -= dt;
        var hx = s.mesh.position.x - pp.x;
        var hy = s.mesh.position.y - (pp.y - 0.6);
        var hz = s.mesh.position.z - pp.z;
        var hitPlayer = (hx * hx + hy * hy + hz * hz) < 0.75;
        if (hitPlayer) { ctx.onPlayerDamage(s.dmg); }
        if (hitPlayer || s.mesh.position.y < 0.05 || s.life <= 0) {
          ctx.scene.remove(s.mesh);
          spits.splice(k, 1);
        }
      }
    }
  };

  window.ES = window.ES || {};
  window.ES.enemies = api;
})();
