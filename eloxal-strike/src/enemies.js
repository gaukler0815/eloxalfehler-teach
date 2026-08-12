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

    var bodyMat = new THREE.MeshStandardMaterial({
      color: c.color, roughness: 0.95, metalness: 0.1, flatShading: true
    });
    var bodyR = c.radius;
    var body = new THREE.Mesh(blobGeometry(bodyR, 7 + bodyR * 100), bodyMat);
    body.position.y = c.height * 0.42;
    body.castShadow = true;
    group.add(body);

    var headR = bodyR * 0.55;
    var head = new THREE.Mesh(blobGeometry(headR, 13 + bodyR * 50), bodyMat);
    head.position.y = c.height * 0.42 + bodyR * 0.9 + headR * 0.4;
    head.castShadow = true;
    group.add(head);

    var eyeMat = new THREE.MeshBasicMaterial({ color: c.eyeColor });
    var eyeGeo = new THREE.SphereGeometry(headR * 0.22, 8, 6);
    [-1, 1].forEach(function (side) {
      var eye = new THREE.Mesh(eyeGeo, eyeMat);
      eye.position.set(side * headR * 0.45, head.position.y + headR * 0.1, headR * 0.8);
      group.add(eye);
    });

    if (c.boss) {
      var crownMat = new THREE.MeshStandardMaterial({
        color: 0xE8A33D, metalness: 0.9, roughness: 0.3
      });
      for (var k = 0; k < 5; k++) {
        var spike = new THREE.Mesh(new THREE.ConeGeometry(0.16, 0.7, 5), crownMat);
        var a = (k / 5) * Math.PI * 2;
        spike.position.set(Math.cos(a) * headR * 0.7,
          head.position.y + headR * 0.95, Math.sin(a) * headR * 0.7);
        group.add(spike);
      }
    }

    // arms: two stubby lumps
    [-1, 1].forEach(function (side) {
      var arm = new THREE.Mesh(blobGeometry(bodyR * 0.35, 29), bodyMat);
      arm.position.set(side * bodyR * 1.05, c.height * 0.42, bodyR * 0.2);
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

        // hit flash
        if (e.hitFlash > 0) {
          e.hitFlash -= dt;
          e.body.material.emissive = new THREE.Color(0xffffff);
          e.body.material.emissiveIntensity = 0.8;
        } else if (e.body.material.emissiveIntensity !== 0) {
          e.body.material.emissiveIntensity = 0;
        }

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
