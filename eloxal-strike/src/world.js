/* Eloxal Strike — builds the anodizing hall (the one world of the game):
 * floor, walls, glowing electrolyte basins, racks, crates, crane bridge,
 * hall signage with the JACOBI ELOXAL logo, lights and fog. Also owns the
 * static collision list (axis-aligned boxes) used by player and enemies.
 */
(function () {
  'use strict';

  var HALL = 76;      // hall is HALL x HALL meters, centered on origin
  var WALL_H = 11;

  function makeCanvas(w, h) {
    var c = document.createElement('canvas');
    c.width = w; c.height = h;
    return c;
  }

  /* Brushed-metal floor with anodizing-line markings. */
  function floorTexture() {
    var c = makeCanvas(512, 512);
    var g = c.getContext('2d');
    g.fillStyle = '#23262c';
    g.fillRect(0, 0, 512, 512);
    for (var i = 0; i < 2200; i++) {
      g.fillStyle = 'rgba(255,255,255,' + (Math.random() * 0.03) + ')';
      g.fillRect(Math.random() * 512, Math.random() * 512, Math.random() * 40 + 4, 1);
    }
    g.strokeStyle = 'rgba(0,0,0,0.5)';
    g.lineWidth = 3;
    g.strokeRect(1, 1, 510, 510);
    g.strokeStyle = 'rgba(232,163,61,0.5)';
    g.lineWidth = 6;
    g.beginPath();
    g.moveTo(0, 256); g.lineTo(512, 256);
    g.stroke();
    var tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(10, 10);
    return tex;
  }

  /* Corrugated dark wall panels. */
  function wallTexture() {
    var c = makeCanvas(256, 256);
    var g = c.getContext('2d');
    g.fillStyle = '#1b1e26';
    g.fillRect(0, 0, 256, 256);
    for (var x = 0; x < 256; x += 32) {
      var grad = g.createLinearGradient(x, 0, x + 32, 0);
      grad.addColorStop(0, 'rgba(255,255,255,0.06)');
      grad.addColorStop(0.5, 'rgba(0,0,0,0.25)');
      grad.addColorStop(1, 'rgba(255,255,255,0.03)');
      g.fillStyle = grad;
      g.fillRect(x, 0, 32, 256);
    }
    var tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(12, 2);
    return tex;
  }

  /* JACOBI ELOXAL hall sign: round JX badge + wordmark, same branding as
   * the menu SVG and the eloxal-rebels hall signs. */
  function logoTexture() {
    var c = makeCanvas(1024, 256);
    var g = c.getContext('2d');
    g.fillStyle = '#141019';
    g.fillRect(0, 0, 1024, 256);
    g.strokeStyle = '#E8A33D';
    g.lineWidth = 8;
    g.strokeRect(10, 10, 1004, 236);
    var grad = g.createLinearGradient(60, 40, 220, 210);
    grad.addColorStop(0, '#E8A33D');
    grad.addColorStop(1, '#B8722A');
    g.fillStyle = grad;
    g.beginPath();
    g.arc(140, 128, 88, 0, Math.PI * 2);
    g.fill();
    g.lineWidth = 10;
    g.strokeStyle = '#17131F';
    g.stroke();
    g.fillStyle = '#ffffff';
    g.font = '800 92px system-ui, sans-serif';
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    g.fillText('JX', 140, 134);
    g.textAlign = 'left';
    g.fillStyle = '#C9D2DC';
    g.font = '800 96px system-ui, sans-serif';
    g.fillText('JACOBI', 270, 100);
    g.fillStyle = '#E8A33D';
    g.font = '800 74px system-ui, sans-serif';
    g.fillText('ELOXAL', 272, 192);
    return new THREE.CanvasTexture(c);
  }

  function warnTexture() {
    var c = makeCanvas(128, 128);
    var g = c.getContext('2d');
    g.fillStyle = '#E8A33D';
    g.fillRect(0, 0, 128, 128);
    g.fillStyle = '#17131F';
    for (var i = -128; i < 256; i += 32) {
      g.beginPath();
      g.moveTo(i, 128); g.lineTo(i + 64, 0); g.lineTo(i + 80, 0); g.lineTo(i + 16, 128);
      g.fill();
    }
    var tex = new THREE.CanvasTexture(c);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    return tex;
  }

  function build(scene) {
    var colliders = [];   // {minX,maxX,minZ,maxZ}
    var solids = [];      // meshes that block hitscan shots
    var half = HALL / 2;

    function addCollider(cx, cz, sx, sz) {
      colliders.push({
        minX: cx - sx / 2, maxX: cx + sx / 2,
        minZ: cz - sz / 2, maxZ: cz + sz / 2
      });
    }

    scene.background = new THREE.Color(0x0c0e14);
    scene.fog = new THREE.Fog(0x0c0e14, 24, 95);

    // --- floor / ceiling ---------------------------------------------------
    var floor = new THREE.Mesh(
      new THREE.PlaneGeometry(HALL, HALL),
      new THREE.MeshStandardMaterial({ map: floorTexture(), roughness: 0.85, metalness: 0.35 })
    );
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);
    solids.push(floor);

    var ceil = new THREE.Mesh(
      new THREE.PlaneGeometry(HALL, HALL),
      new THREE.MeshStandardMaterial({ color: 0x11141b, roughness: 1 })
    );
    ceil.rotation.x = Math.PI / 2;
    ceil.position.y = WALL_H;
    scene.add(ceil);

    // --- walls -------------------------------------------------------------
    var wallMat = new THREE.MeshStandardMaterial({ map: wallTexture(), roughness: 0.9, metalness: 0.2 });
    var wallGeo = new THREE.PlaneGeometry(HALL, WALL_H);
    [
      { pos: [0, WALL_H / 2, -half], rot: 0 },
      { pos: [0, WALL_H / 2, half], rot: Math.PI },
      { pos: [-half, WALL_H / 2, 0], rot: Math.PI / 2 },
      { pos: [half, WALL_H / 2, 0], rot: -Math.PI / 2 }
    ].forEach(function (w) {
      var m = new THREE.Mesh(wallGeo, wallMat);
      m.position.set(w.pos[0], w.pos[1], w.pos[2]);
      m.rotation.y = w.rot;
      scene.add(m);
      solids.push(m);
    });
    // wall colliders (thick boxes just outside the visible planes)
    addCollider(0, -half - 0.5, HALL + 4, 1);
    addCollider(0, half + 0.5, HALL + 4, 1);
    addCollider(-half - 0.5, 0, 1, HALL + 4);
    addCollider(half + 0.5, 0, 1, HALL + 4);

    // --- logo signs on two walls ------------------------------------------
    var logoTex = logoTexture();
    var logoMat = new THREE.MeshStandardMaterial({
      map: logoTex, emissive: 0xffffff, emissiveMap: logoTex, emissiveIntensity: 0.55,
      roughness: 0.6
    });
    [[0, 7.4, -half + 0.12, 0], [0, 7.4, half - 0.12, Math.PI]].forEach(function (p) {
      var sign = new THREE.Mesh(new THREE.PlaneGeometry(16, 4), logoMat);
      sign.position.set(p[0], p[1], p[2]);
      sign.rotation.y = p[3];
      scene.add(sign);
    });

    // --- electrolyte basins (glowing hazard pools with warn rims) ----------
    var basinPositions = [[-16, -10], [16, -10], [-16, 12], [16, 12]];
    var warnTex = warnTexture();
    basinPositions.forEach(function (bp) {
      var bx = bp[0], bz = bp[1];
      var rim = new THREE.Mesh(
        new THREE.BoxGeometry(9, 0.9, 6),
        new THREE.MeshStandardMaterial({ map: warnTex, roughness: 0.7, metalness: 0.3 })
      );
      rim.position.set(bx, 0.45, bz);
      scene.add(rim);
      solids.push(rim);
      var pool = new THREE.Mesh(
        new THREE.PlaneGeometry(8.2, 5.2),
        new THREE.MeshStandardMaterial({
          color: 0x9fe348, emissive: 0x86d32f, emissiveIntensity: 0.9, roughness: 0.2
        })
      );
      pool.rotation.x = -Math.PI / 2;
      pool.position.set(bx, 0.92, bz);
      scene.add(pool);
      var glow = new THREE.PointLight(0x9fe348, 0.9, 16, 2);
      glow.position.set(bx, 2.2, bz);
      scene.add(glow);
      addCollider(bx, bz, 9, 6);
    });

    // --- anodizing racks (pillar pairs with hanging aluminum parts) --------
    var alu = new THREE.MeshStandardMaterial({ color: 0xb9c4cf, roughness: 0.35, metalness: 0.85 });
    var steel = new THREE.MeshStandardMaterial({ color: 0x3a4048, roughness: 0.6, metalness: 0.7 });
    var goldAlu = new THREE.MeshStandardMaterial({ color: 0xE8A33D, roughness: 0.35, metalness: 0.8 });
    var barGeo = new THREE.BoxGeometry(0.25, 4.2, 0.25);
    var partGeo = new THREE.BoxGeometry(0.5, 1.1, 0.14);
    var rackSpots = [[-28, 0], [28, 0], [0, -24], [0, 24], [-24, -22], [24, 22]];
    rackSpots.forEach(function (rs, ri) {
      var rx = rs[0], rz = rs[1];
      var group = new THREE.Group();
      for (var side = -1; side <= 1; side += 2) {
        var post = new THREE.Mesh(barGeo, steel);
        post.position.set(side * 2.4, 2.1, 0);
        group.add(post);
      }
      var beam = new THREE.Mesh(new THREE.BoxGeometry(5.4, 0.22, 0.22), steel);
      beam.position.set(0, 4.1, 0);
      group.add(beam);
      for (var i = 0; i < 6; i++) {
        var part = new THREE.Mesh(partGeo, (ri + i) % 3 === 0 ? goldAlu : alu);
        part.position.set(-2 + i * 0.8, 3.3, 0);
        part.rotation.y = Math.random() * 0.3 - 0.15;
        part.castShadow = true;
        group.add(part);
      }
      group.position.set(rx, 0, rz);
      group.rotation.y = (ri % 2) * Math.PI / 2;
      scene.add(group);
      if (ri % 2 === 0) { addCollider(rx, rz, 5.6, 0.9); }
      else { addCollider(rx, rz, 0.9, 5.6); }
    });

    // --- crates & barrels as cover -----------------------------------------
    var crateMat = new THREE.MeshStandardMaterial({ color: 0x6b5a3e, roughness: 0.9 });
    var crateSpots = [
      [-8, -18, 2.2], [9, -20, 1.8], [-10, 20, 2.0], [8, 18, 2.4],
      [-30, 12, 1.8], [30, -12, 2.0], [-6, 2, 1.6], [7, -3, 1.9],
      [22, 8, 1.7], [-22, -8, 1.7]
    ];
    crateSpots.forEach(function (cs) {
      var s = cs[2];
      var crate = new THREE.Mesh(new THREE.BoxGeometry(s, s, s), crateMat);
      crate.position.set(cs[0], s / 2, cs[1]);
      crate.rotation.y = (cs[0] * 13 + cs[1] * 7) % 1;
      crate.castShadow = true;
      crate.receiveShadow = true;
      scene.add(crate);
      solids.push(crate);
      addCollider(cs[0], cs[1], s + 0.3, s + 0.3);
    });

    var barrelGeo = new THREE.CylinderGeometry(0.55, 0.55, 1.3, 14);
    var barrelMat = new THREE.MeshStandardMaterial({ color: 0x2e6b8a, roughness: 0.5, metalness: 0.6 });
    [[-12, -4], [13, 5], [-3, -26], [4, 27], [-26, 20], [27, -19]].forEach(function (bp) {
      var b = new THREE.Mesh(barrelGeo, barrelMat);
      b.position.set(bp[0], 0.65, bp[1]);
      b.castShadow = true;
      scene.add(b);
      solids.push(b);
      addCollider(bp[0], bp[1], 1.3, 1.3);
    });

    // --- crane bridge across the hall --------------------------------------
    var crane = new THREE.Group();
    var craneBeam = new THREE.Mesh(new THREE.BoxGeometry(HALL - 2, 1.1, 2.2), goldAlu);
    craneBeam.position.y = 9.2;
    crane.add(craneBeam);
    var trolley = new THREE.Mesh(new THREE.BoxGeometry(2.4, 1.4, 2.6), steel);
    trolley.position.set(6, 8.4, 0);
    crane.add(trolley);
    var cable = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 3.4), steel);
    cable.position.set(6, 6.2, 0);
    crane.add(cable);
    var hookLoad = new THREE.Mesh(new THREE.BoxGeometry(1.6, 1.0, 0.5), alu);
    hookLoad.position.set(6, 4.4, 0);
    crane.add(hookLoad);
    scene.add(crane);

    // --- lights ------------------------------------------------------------
    scene.add(new THREE.AmbientLight(0x404860, 0.55));
    var hemi = new THREE.HemisphereLight(0x6a7a9a, 0x1a140e, 0.5);
    scene.add(hemi);

    var key = new THREE.DirectionalLight(0xfff2dd, 0.75);
    key.position.set(14, 18, 8);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.left = -40; key.shadow.camera.right = 40;
    key.shadow.camera.top = 40; key.shadow.camera.bottom = -40;
    scene.add(key);

    // hanging hall lamps
    [[-18, 0], [18, 0], [0, -16], [0, 16]].forEach(function (lp) {
      var lamp = new THREE.PointLight(0xffe6b0, 0.6, 30, 2);
      lamp.position.set(lp[0], 8.6, lp[1]);
      scene.add(lamp);
      var bulb = new THREE.Mesh(
        new THREE.SphereGeometry(0.28, 10, 8),
        new THREE.MeshBasicMaterial({ color: 0xffe6b0 })
      );
      bulb.position.copy(lamp.position);
      scene.add(bulb);
    });

    return {
      colliders: colliders,
      solids: solids,
      bounds: { min: -half + 1, max: half - 1 },
      crane: { trolley: trolley, cable: cable, load: hookLoad },
      // enemy spawn points along the hall edges
      spawnPoints: [
        [-half + 4, -half + 4], [half - 4, -half + 4],
        [-half + 4, half - 4], [half - 4, half - 4],
        [0, -half + 3], [0, half - 3],
        [-half + 3, 0], [half - 3, 0]
      ],
      pickupPoints: [[-12, 0], [12, 0], [0, -14], [0, 14], [-24, 16], [24, -16]]
    };
  }

  /* Push a circle (x,z,r) out of all static colliders. Mutates and returns
   * the given position object {x,z}. Shared by player and enemies. */
  function collide(pos, radius, colliders, bounds) {
    if (bounds) {
      pos.x = Math.max(bounds.min + radius, Math.min(bounds.max - radius, pos.x));
      pos.z = Math.max(bounds.min + radius, Math.min(bounds.max - radius, pos.z));
    }
    for (var i = 0; i < colliders.length; i++) {
      var c = colliders[i];
      var cx = Math.max(c.minX, Math.min(c.maxX, pos.x));
      var cz = Math.max(c.minZ, Math.min(c.maxZ, pos.z));
      var dx = pos.x - cx, dz = pos.z - cz;
      var d2 = dx * dx + dz * dz;
      if (d2 < radius * radius) {
        if (d2 > 0.000001) {
          var d = Math.sqrt(d2);
          pos.x = cx + dx / d * radius;
          pos.z = cz + dz / d * radius;
        } else {
          // center is inside the box: push out along the smallest overlap
          var pushLeft = pos.x - c.minX + radius;
          var pushRight = c.maxX - pos.x + radius;
          var pushDown = pos.z - c.minZ + radius;
          var pushUp = c.maxZ - pos.z + radius;
          var m = Math.min(pushLeft, pushRight, pushDown, pushUp);
          if (m === pushLeft) { pos.x = c.minX - radius; }
          else if (m === pushRight) { pos.x = c.maxX + radius; }
          else if (m === pushDown) { pos.z = c.minZ - radius; }
          else { pos.z = c.maxZ + radius; }
        }
      }
    }
    return pos;
  }

  window.ES = window.ES || {};
  window.ES.world = { build: build, collide: collide };
})();
