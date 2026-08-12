/* Eloxal Strike — builds the anodizing hall (the one world of the game):
 * floor, walls, glowing electrolyte basins, racks, crates, crane bridge,
 * ceiling girders, pipes, light cones, steam, dust, rotating warn beacons
 * and the JACOBI ELOXAL hall signage. Also owns the static collision list
 * (axis-aligned boxes) used by player and enemies, and an update() hook for
 * its ambient animations.
 */
(function () {
  'use strict';

  var HALL = 76;      // hall is HALL x HALL meters, centered on origin
  var WALL_H = 11;

  function lin(hex) { return new THREE.Color(hex).convertSRGBToLinear(); }

  function makeCanvas(w, h) {
    var c = document.createElement('canvas');
    c.width = w; c.height = h;
    return c;
  }

  /* Brushed-metal floor with anodizing-line markings and oil stains. */
  function floorTexture() {
    var c = makeCanvas(1024, 1024);
    var g = c.getContext('2d');
    g.fillStyle = '#23262c';
    g.fillRect(0, 0, 1024, 1024);
    // brushed streaks
    for (var i = 0; i < 5200; i++) {
      g.fillStyle = 'rgba(255,255,255,' + (Math.random() * 0.03) + ')';
      g.fillRect(Math.random() * 1024, Math.random() * 1024, Math.random() * 60 + 6, 1);
    }
    // oil / chemical stains
    for (var s = 0; s < 26; s++) {
      var sx = Math.random() * 1024, sy = Math.random() * 1024;
      var sr = 18 + Math.random() * 70;
      var grad = g.createRadialGradient(sx, sy, 2, sx, sy, sr);
      grad.addColorStop(0, 'rgba(8,10,14,' + (0.25 + Math.random() * 0.2) + ')');
      grad.addColorStop(1, 'rgba(8,10,14,0)');
      g.fillStyle = grad;
      g.beginPath();
      g.ellipse(sx, sy, sr, sr * (0.5 + Math.random() * 0.5), Math.random() * 3, 0, Math.PI * 2);
      g.fill();
    }
    // panel seams
    g.strokeStyle = 'rgba(0,0,0,0.5)';
    g.lineWidth = 4;
    g.strokeRect(2, 2, 1020, 1020);
    // painted walkway lines
    g.strokeStyle = 'rgba(232,163,61,0.45)';
    g.lineWidth = 10;
    g.beginPath();
    g.moveTo(0, 512); g.lineTo(1024, 512);
    g.moveTo(512, 0); g.lineTo(512, 1024);
    g.stroke();
    g.setLineDash([40, 30]);
    g.strokeStyle = 'rgba(201,210,220,0.18)';
    g.lineWidth = 6;
    g.beginPath();
    g.moveTo(0, 100); g.lineTo(1024, 100);
    g.moveTo(0, 924); g.lineTo(1024, 924);
    g.stroke();
    g.setLineDash([]);
    var tex = new THREE.CanvasTexture(c);
    tex.encoding = THREE.sRGBEncoding;
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(6, 6);
    return tex;
  }

  /* Corrugated dark wall panels with a grimy lower edge. */
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
    var dirt = g.createLinearGradient(0, 190, 0, 256);
    dirt.addColorStop(0, 'rgba(10,8,6,0)');
    dirt.addColorStop(1, 'rgba(10,8,6,0.5)');
    g.fillStyle = dirt;
    g.fillRect(0, 190, 256, 66);
    var tex = new THREE.CanvasTexture(c);
    tex.encoding = THREE.sRGBEncoding;
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
    var tex = new THREE.CanvasTexture(c);
    tex.encoding = THREE.sRGBEncoding;
    return tex;
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
    tex.encoding = THREE.sRGBEncoding;
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    return tex;
  }

  /* Soft dark blob planted under props as a cheap contact shadow. */
  function aoTexture() {
    var c = makeCanvas(128, 128);
    var g = c.getContext('2d');
    var grad = g.createRadialGradient(64, 64, 8, 64, 64, 62);
    grad.addColorStop(0, 'rgba(0,0,0,0.55)');
    grad.addColorStop(0.7, 'rgba(0,0,0,0.25)');
    grad.addColorStop(1, 'rgba(0,0,0,0)');
    g.fillStyle = grad;
    g.fillRect(0, 0, 128, 128);
    return new THREE.CanvasTexture(c);
  }

  /* Scrolling LED ticker: message drawn twice so texture offset wraps
   * seamlessly at repeat 0.5. */
  function tickerTexture() {
    var msg = 'JACOBI ELOXAL  +++  WELLE FUER WELLE QUALITAET  +++  VORSICHT: KORROSIONSBANDE IN HALLE 1  +++  SCHUTZSCHICHT PRUEFEN  +++  ';
    var c = makeCanvas(2048, 64);
    var g = c.getContext('2d');
    g.fillStyle = '#04070a';
    g.fillRect(0, 0, 2048, 64);
    g.font = '700 40px monospace';
    g.textBaseline = 'middle';
    g.shadowColor = '#9fe348';
    g.shadowBlur = 12;
    g.fillStyle = '#9fe348';
    // squeeze one full message into 1024px, then repeat it for the wrap
    var w = g.measureText(msg).width;
    g.save();
    g.scale(1024 / w, 1);
    g.fillText(msg, 0, 34);
    g.fillText(msg, w, 34);
    g.restore();
    var tex = new THREE.CanvasTexture(c);
    tex.encoding = THREE.sRGBEncoding;
    tex.wrapS = THREE.RepeatWrapping;
    tex.repeat.x = 0.5;
    return tex;
  }

  /* Soft round dot used by steam/dust point clouds. */
  function puffTexture() {
    var c = makeCanvas(64, 64);
    var g = c.getContext('2d');
    var grad = g.createRadialGradient(32, 32, 2, 32, 32, 30);
    grad.addColorStop(0, 'rgba(255,255,255,0.9)');
    grad.addColorStop(0.4, 'rgba(255,255,255,0.35)');
    grad.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = grad;
    g.fillRect(0, 0, 64, 64);
    return new THREE.CanvasTexture(c);
  }

  function build(scene) {
    var colliders = [];   // {minX,maxX,minZ,maxZ}
    var solids = [];      // meshes that block hitscan shots
    var animated = { steams: [], beacons: [], pools: [], dust: null, sparks: null,
      leds: [], fans: [], chains: [], ticker: null, flicker: null };
    var half = HALL / 2;

    function addCollider(cx, cz, sx, sz) {
      colliders.push({
        minX: cx - sx / 2, maxX: cx + sx / 2,
        minZ: cz - sz / 2, maxZ: cz + sz / 2
      });
    }

    scene.background = new THREE.Color(0x0a0c12);
    scene.fog = new THREE.Fog(0x0a0c12, 20, 88);

    var puffTex = puffTexture();

    // --- floor / ceiling ---------------------------------------------------
    var floor = new THREE.Mesh(
      new THREE.PlaneGeometry(HALL, HALL),
      new THREE.MeshStandardMaterial({ map: floorTexture(), roughness: 0.78, metalness: 0.3 })
    );
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);
    solids.push(floor);

    var ceil = new THREE.Mesh(
      new THREE.PlaneGeometry(HALL, HALL),
      new THREE.MeshStandardMaterial({ color: 0x0d1016, roughness: 1 })
    );
    ceil.rotation.x = Math.PI / 2;
    ceil.position.y = WALL_H;
    scene.add(ceil);

    // skylight strips in the ceiling (cool daylight slits)
    var skyMat = new THREE.MeshBasicMaterial({ color: 0x9fb8e8 });
    [-18, 0, 18].forEach(function (sx) {
      var strip = new THREE.Mesh(new THREE.PlaneGeometry(2.6, HALL - 16), skyMat);
      strip.rotation.x = Math.PI / 2;
      strip.position.set(sx, WALL_H - 0.05, 0);
      scene.add(strip);
    });

    // ceiling girders (I-beam look: dark boxes) across both axes
    var girderMat = new THREE.MeshStandardMaterial({ color: lin(0x2a2f38), roughness: 0.7, metalness: 0.6 });
    for (var gx = -30; gx <= 30; gx += 12) {
      var girder = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.8, HALL - 2), girderMat);
      girder.position.set(gx, WALL_H - 0.5, 0);
      scene.add(girder);
    }
    var crossGirder = new THREE.Mesh(new THREE.BoxGeometry(HALL - 2, 0.6, 0.5), girderMat);
    crossGirder.position.set(0, WALL_H - 1.1, -14);
    scene.add(crossGirder);
    var crossGirder2 = crossGirder.clone();
    crossGirder2.position.z = 14;
    scene.add(crossGirder2);

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

    // pipe runs along the side walls
    var pipeMatA = new THREE.MeshStandardMaterial({ color: lin(0x4a5560), roughness: 0.45, metalness: 0.85 });
    var pipeMatB = new THREE.MeshStandardMaterial({ color: lin(0x2e6b8a), roughness: 0.4, metalness: 0.8 });
    [-1, 1].forEach(function (side) {
      [[2.6, pipeMatA, 0.16], [3.15, pipeMatB, 0.11], [5.2, pipeMatA, 0.22]].forEach(function (p) {
        var pipe = new THREE.Mesh(new THREE.CylinderGeometry(p[2], p[2], HALL - 2, 10), p[1]);
        pipe.rotation.x = Math.PI / 2;
        pipe.position.set(side * (half - 0.45), p[0], 0);
        scene.add(pipe);
      });
    });

    // --- logo signs on two walls ------------------------------------------
    var logoTex = logoTexture();
    var logoMat = new THREE.MeshStandardMaterial({
      map: logoTex, emissive: 0xffffff, emissiveMap: logoTex, emissiveIntensity: 0.9,
      roughness: 0.6
    });
    [[0, 7.4, -half + 0.12, 0], [0, 7.4, half - 0.12, Math.PI]].forEach(function (p) {
      var sign = new THREE.Mesh(new THREE.PlaneGeometry(16, 4), logoMat);
      sign.position.set(p[0], p[1], p[2]);
      sign.rotation.y = p[3];
      scene.add(sign);
      // small warm spot washing the sign area
      var wash = new THREE.PointLight(0xE8A33D, 0.5, 14, 2);
      wash.position.set(p[0], p[1] - 1.5, p[2] + (p[3] === 0 ? 3 : -3));
      scene.add(wash);
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
      var poolMat = new THREE.MeshStandardMaterial({
        color: 0x9fe348, emissive: 0x86d32f, emissiveIntensity: 0.9, roughness: 0.15,
        metalness: 0.1
      });
      var pool = new THREE.Mesh(new THREE.PlaneGeometry(8.2, 5.2), poolMat);
      pool.rotation.x = -Math.PI / 2;
      pool.position.set(bx, 0.92, bz);
      scene.add(pool);
      var glow = new THREE.PointLight(0x9fe348, 1.1, 17, 2);
      glow.position.set(bx, 2.4, bz);
      scene.add(glow);
      animated.pools.push({ mat: poolMat, light: glow, phase: bx * 0.7 + bz });
      addCollider(bx, bz, 9, 6);

      // rising steam above the bath
      var steamCount = 26;
      var pos = new Float32Array(steamCount * 3);
      for (var i = 0; i < steamCount; i++) {
        pos[i * 3] = bx + (Math.random() - 0.5) * 7;
        pos[i * 3 + 1] = 1 + Math.random() * 3.2;
        pos[i * 3 + 2] = bz + (Math.random() - 0.5) * 4.4;
      }
      var steamGeo = new THREE.BufferGeometry();
      steamGeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      var steamMat = new THREE.PointsMaterial({
        map: puffTex, color: 0xcfe8b8, size: 1.2, transparent: true, opacity: 0.12,
        depthWrite: false, blending: THREE.AdditiveBlending, sizeAttenuation: true
      });
      var steam = new THREE.Points(steamGeo, steamMat);
      scene.add(steam);
      animated.steams.push({ points: steam, cx: bx, cz: bz });
    });

    // --- anodizing racks (pillar pairs with hanging aluminum parts) --------
    var alu = new THREE.MeshStandardMaterial({ color: lin(0xb9c4cf), roughness: 0.25, metalness: 0.9 });
    var steel = new THREE.MeshStandardMaterial({ color: lin(0x3a4048), roughness: 0.55, metalness: 0.75 });
    var goldAlu = new THREE.MeshStandardMaterial({ color: lin(0xE8A33D), roughness: 0.28, metalness: 0.85 });
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
    var crateMat = new THREE.MeshStandardMaterial({ color: lin(0x6b5a3e), roughness: 0.9 });
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
    var barrelMat = new THREE.MeshStandardMaterial({ color: lin(0x2e6b8a), roughness: 0.35, metalness: 0.75 });
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

    // --- hall props: forklift, control cabinet, fans, chains, ticker -------
    var aoTex = aoTexture();
    function contactShadow(x, z, size) {
      var blob = new THREE.Mesh(new THREE.PlaneGeometry(size, size),
        new THREE.MeshBasicMaterial({ map: aoTex, transparent: true, depthWrite: false }));
      blob.rotation.x = -Math.PI / 2;
      blob.position.set(x, 0.012, z);
      scene.add(blob);
    }
    crateSpots.forEach(function (cs) { contactShadow(cs[0], cs[1], cs[2] * 2.1); });
    [[-12, -4], [13, 5], [-3, -26], [4, 27], [-26, 20], [27, -19]].forEach(function (bp) {
      contactShadow(bp[0], bp[1], 2.2);
    });

    // company forklift, parked
    (function () {
      var fk = new THREE.Group();
      var bodyMat = new THREE.MeshStandardMaterial({ color: lin(0xd97f2a), roughness: 0.5, metalness: 0.6 });
      var darkMat = new THREE.MeshStandardMaterial({ color: lin(0x22262e), roughness: 0.7, metalness: 0.4 });
      var body = new THREE.Mesh(new THREE.BoxGeometry(1.15, 0.85, 1.9), bodyMat);
      body.position.set(0, 0.85, 0.2);
      body.castShadow = true;
      fk.add(body);
      var counter = new THREE.Mesh(new THREE.BoxGeometry(1.15, 0.6, 0.5), bodyMat);
      counter.position.set(0, 0.75, 1.3);
      fk.add(counter);
      var seat = new THREE.Mesh(new THREE.BoxGeometry(0.55, 0.5, 0.45), darkMat);
      seat.position.set(0, 1.5, 0.45);
      fk.add(seat);
      // overhead guard
      [[-0.5, -0.6], [0.5, -0.6], [-0.5, 0.9], [0.5, 0.9]].forEach(function (pp) {
        var post = new THREE.Mesh(new THREE.BoxGeometry(0.07, 1.4, 0.07), darkMat);
        post.position.set(pp[0], 1.95, pp[1]);
        fk.add(post);
      });
      var roof = new THREE.Mesh(new THREE.BoxGeometry(1.15, 0.07, 1.7), darkMat);
      roof.position.set(0, 2.68, 0.15);
      fk.add(roof);
      // mast + forks
      [-0.3, 0.3].forEach(function (mx) {
        var bar = new THREE.Mesh(new THREE.BoxGeometry(0.09, 2.3, 0.09), darkMat);
        bar.position.set(mx, 1.15, -1.15);
        fk.add(bar);
      });
      var cross = new THREE.Mesh(new THREE.BoxGeometry(0.75, 0.09, 0.09), darkMat);
      cross.position.set(0, 1.9, -1.15);
      fk.add(cross);
      [-0.28, 0.28].forEach(function (fx) {
        var fork = new THREE.Mesh(new THREE.BoxGeometry(0.13, 0.05, 1.15), new THREE.MeshStandardMaterial({
          color: lin(0x3a4048), roughness: 0.5, metalness: 0.8
        }));
        fork.position.set(fx, 0.1, -1.75);
        fk.add(fork);
      });
      var wheelGeo = new THREE.CylinderGeometry(0.32, 0.32, 0.22, 14);
      var wheelMat = new THREE.MeshStandardMaterial({ color: lin(0x17131F), roughness: 0.9 });
      [[-0.62, -0.55], [0.62, -0.55], [-0.62, 1.0], [0.62, 1.0]].forEach(function (wp) {
        var wheel = new THREE.Mesh(wheelGeo, wheelMat);
        wheel.rotation.z = Math.PI / 2;
        wheel.position.set(wp[0], 0.32, wp[1]);
        fk.add(wheel);
      });
      fk.position.set(-21, 0, 27);
      fk.rotation.y = 2.5;
      scene.add(fk);
      solids.push(body);
      contactShadow(-21, 27, 4.2);
      addCollider(-21, 27, 3.2, 3.2);
    })();

    // control cabinet with blinking status LEDs
    (function () {
      var cab = new THREE.Mesh(new THREE.BoxGeometry(1.8, 2.1, 0.5),
        new THREE.MeshStandardMaterial({ color: lin(0x515c68), roughness: 0.5, metalness: 0.6 }));
      cab.position.set(12, 1.05, -half + 0.4);
      cab.castShadow = true;
      scene.add(cab);
      solids.push(cab);
      addCollider(12, -half + 0.4, 2.0, 0.8);
      var seam = new THREE.Mesh(new THREE.BoxGeometry(0.02, 1.9, 0.02),
        new THREE.MeshStandardMaterial({ color: lin(0x2a2f38), roughness: 0.8 }));
      seam.position.set(12, 1.05, -half + 0.66);
      scene.add(seam);
      var ledColors = [0xff3b30, 0x9fe348, 0xE8A33D, 0x9fe348, 0x58c7f0, 0xff3b30];
      for (var li2 = 0; li2 < 6; li2++) {
        var ledMat = new THREE.MeshBasicMaterial({ color: ledColors[li2] });
        var led = new THREE.Mesh(new THREE.BoxGeometry(0.07, 0.07, 0.03), ledMat);
        led.position.set(11.4 + (li2 % 3) * 0.6, 2.4 - Math.floor(li2 / 3) * 0.25, -half + 0.67);
        scene.add(led);
        animated.leds.push({
          mat: ledMat, on: new THREE.Color(ledColors[li2]),
          off: new THREE.Color(ledColors[li2]).multiplyScalar(0.15),
          phase: Math.random() * 6, speed: 0.4 + Math.random() * 1.2
        });
      }
    })();

    // slow industrial wall fans
    [[half - 0.45, 7.6, -12, -Math.PI / 2], [-half + 0.45, 7.6, 12, Math.PI / 2]].forEach(function (fp) {
      var fan = new THREE.Group();
      var ring = new THREE.Mesh(new THREE.TorusGeometry(0.95, 0.09, 8, 20), steel);
      fan.add(ring);
      var blades = new THREE.Group();
      for (var b3 = 0; b3 < 3; b3++) {
        var blade = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.85, 0.05), steel);
        blade.position.y = 0.48;
        var holder = new THREE.Group();
        holder.rotation.z = (b3 / 3) * Math.PI * 2;
        holder.add(blade);
        blades.add(holder);
      }
      var hub = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.14, 0.14, 10), steel);
      hub.rotation.x = Math.PI / 2;
      blades.add(hub);
      fan.add(blades);
      fan.position.set(fp[0], fp[1], fp[2]);
      fan.rotation.y = fp[3];
      scene.add(fan);
      animated.fans.push(blades);
    });

    // chains with hooks hanging from the ceiling girders, swaying gently
    [[-6, -8], [-6, -11], [18, 6]].forEach(function (cp, ci) {
      var chain = new THREE.Group();
      var link = new THREE.Mesh(new THREE.CylinderGeometry(0.028, 0.028, 2.6, 8), steel);
      link.position.y = -1.3;
      chain.add(link);
      var hook = new THREE.Mesh(new THREE.TorusGeometry(0.14, 0.035, 8, 14), steel);
      hook.position.y = -2.72;
      chain.add(hook);
      chain.position.set(cp[0], 10.1, cp[1]);
      scene.add(chain);
      animated.chains.push({ group: chain, phase: ci * 2.1 });
    });

    // LED ticker board high on the east wall
    (function () {
      var tex = tickerTexture();
      var board = new THREE.Mesh(new THREE.PlaneGeometry(15, 1.0),
        new THREE.MeshBasicMaterial({ map: tex }));
      board.position.set(half - 0.15, 5.9, 8);
      board.rotation.y = -Math.PI / 2;
      scene.add(board);
      var frame = new THREE.Mesh(new THREE.BoxGeometry(0.08, 1.3, 15.5), steel);
      frame.position.set(half - 0.08, 5.9, 8);
      scene.add(frame);
      animated.ticker = tex;
    })();

    // --- lights ------------------------------------------------------------
    scene.add(new THREE.AmbientLight(0x39415a, 0.4));
    var hemi = new THREE.HemisphereLight(0x5a6a8c, 0x231a10, 0.45);
    scene.add(hemi);

    var key = new THREE.DirectionalLight(0xfff2dd, 0.85);
    key.position.set(14, 18, 8);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.left = -40; key.shadow.camera.right = 40;
    key.shadow.camera.top = 40; key.shadow.camera.bottom = -40;
    scene.add(key);

    // hanging hall lamps with visible light cones
    var coneMat = new THREE.MeshBasicMaterial({
      color: 0xffe6b0, transparent: true, opacity: 0.05, depthWrite: false,
      blending: THREE.AdditiveBlending, side: THREE.DoubleSide
    });
    [[-18, 0], [18, 0], [0, -16], [0, 16]].forEach(function (lp, li) {
      var lamp = new THREE.PointLight(0xffe6b0, 0.75, 32, 2);
      lamp.position.set(lp[0], 8.6, lp[1]);
      scene.add(lamp);
      var shade = new THREE.Mesh(
        new THREE.ConeGeometry(0.55, 0.5, 12, 1, true),
        new THREE.MeshStandardMaterial({ color: lin(0x22262e), roughness: 0.5, metalness: 0.8, side: THREE.DoubleSide })
      );
      shade.position.set(lp[0], 8.85, lp[1]);
      scene.add(shade);
      var bulb = new THREE.Mesh(
        new THREE.SphereGeometry(0.24, 10, 8),
        new THREE.MeshBasicMaterial({ color: 0xffe6b0 })
      );
      bulb.position.copy(lamp.position);
      scene.add(bulb);
      var cone = new THREE.Mesh(new THREE.ConeGeometry(3.6, 8.6, 20, 1, true),
        li === 2 ? coneMat.clone() : coneMat);
      cone.position.set(lp[0], 4.3, lp[1]);
      scene.add(cone);
      if (li === 2) {
        animated.flicker = { light: lamp, bulb: bulb, cone: cone, base: 0.75, t: 5, burst: 0 };
      }
    });

    // rotating orange warn beacons high on the side walls
    [[-half + 0.6, -18], [half - 0.6, 18]].forEach(function (wp) {
      var beacon = new THREE.Group();
      var base = new THREE.Mesh(
        new THREE.CylinderGeometry(0.22, 0.26, 0.3, 10),
        new THREE.MeshStandardMaterial({ color: lin(0x2a2f38), roughness: 0.6 })
      );
      beacon.add(base);
      var dome = new THREE.Mesh(
        new THREE.SphereGeometry(0.2, 10, 8),
        new THREE.MeshBasicMaterial({ color: 0xff7b2e })
      );
      dome.position.y = 0.24;
      beacon.add(dome);
      var ray = new THREE.Mesh(
        new THREE.ConeGeometry(1.4, 5.5, 14, 1, true),
        new THREE.MeshBasicMaterial({
          color: 0xff7b2e, transparent: true, opacity: 0.08, depthWrite: false,
          blending: THREE.AdditiveBlending, side: THREE.DoubleSide
        })
      );
      ray.rotation.z = Math.PI / 2;
      ray.position.set(2.75, 0.24, 0);
      beacon.add(ray);
      var blink = new THREE.PointLight(0xff7b2e, 0.6, 20, 2);
      blink.position.y = 0.3;
      beacon.add(blink);
      beacon.position.set(wp[0], 8.2, wp[1]);
      scene.add(beacon);
      animated.beacons.push({ group: beacon, light: blink });
    });

    // electric sparks: one pooled point cloud, bursting from changing spots
    var SPARKS = 70;
    var sparkPos = new Float32Array(SPARKS * 3);
    var sparkVel = new Float32Array(SPARKS * 3);
    var sparkLife = new Float32Array(SPARKS);
    for (var sp = 0; sp < SPARKS; sp++) { sparkPos[sp * 3 + 1] = -5; }
    var sparkGeo = new THREE.BufferGeometry();
    sparkGeo.setAttribute('position', new THREE.BufferAttribute(sparkPos, 3));
    var sparkPoints = new THREE.Points(sparkGeo, new THREE.PointsMaterial({
      map: puffTex, color: 0xffc36b, size: 0.16, transparent: true, opacity: 0.95,
      depthWrite: false, blending: THREE.AdditiveBlending, sizeAttenuation: true
    }));
    scene.add(sparkPoints);
    var sparkLight = new THREE.PointLight(0xffc36b, 0, 14, 2);
    scene.add(sparkLight);
    animated.sparks = {
      pos: sparkPos, vel: sparkVel, life: sparkLife, geo: sparkGeo,
      light: sparkLight, timer: 2, next: 0,
      spots: [[-28, 4.2, 0], [28, 4.2, 0], [0, 4.2, -24], [0, 4.2, 24]]
    };

    // dust motes drifting through the hall
    var dustCount = 220;
    var dustPos = new Float32Array(dustCount * 3);
    for (var d = 0; d < dustCount; d++) {
      dustPos[d * 3] = (Math.random() - 0.5) * (HALL - 6);
      dustPos[d * 3 + 1] = 0.5 + Math.random() * 9;
      dustPos[d * 3 + 2] = (Math.random() - 0.5) * (HALL - 6);
    }
    var dustGeo = new THREE.BufferGeometry();
    dustGeo.setAttribute('position', new THREE.BufferAttribute(dustPos, 3));
    var dust = new THREE.Points(dustGeo, new THREE.PointsMaterial({
      map: puffTex, color: 0xbfcbd8, size: 0.09, transparent: true, opacity: 0.2,
      depthWrite: false, blending: THREE.AdditiveBlending, sizeAttenuation: true
    }));
    scene.add(dust);
    animated.dust = dust;

    /* Ambient animation: steam rises, pools shimmer, beacons rotate,
     * dust drifts. Called once per frame from main.js. */
    function update(dt, t) {
      for (var i = 0; i < animated.steams.length; i++) {
        var st = animated.steams[i];
        var arr = st.points.geometry.attributes.position;
        for (var j = 0; j < arr.count; j++) {
          var y = arr.getY(j) + dt * (0.5 + (j % 5) * 0.1);
          if (y > 4.6) {
            y = 1;
            arr.setX(j, st.cx + (Math.random() - 0.5) * 7);
            arr.setZ(j, st.cz + (Math.random() - 0.5) * 4.4);
          }
          arr.setY(j, y);
        }
        arr.needsUpdate = true;
      }
      for (var p = 0; p < animated.pools.length; p++) {
        var po = animated.pools[p];
        var pulse = 0.85 + Math.sin(t * 1.7 + po.phase) * 0.18;
        po.mat.emissiveIntensity = pulse;
        po.light.intensity = 0.9 + pulse * 0.35;
      }
      for (var b = 0; b < animated.beacons.length; b++) {
        var be = animated.beacons[b];
        be.group.rotation.y = t * 2.4 + b * Math.PI;
        be.light.intensity = 0.45 + Math.sin(t * 4.8 + b) * 0.25;
      }
      if (animated.dust) {
        animated.dust.rotation.y = t * 0.006;
        animated.dust.material.opacity = 0.16 + Math.sin(t * 0.7) * 0.05;
      }

      // sparks: advance live ones, periodically burst at a new spot
      var sk = animated.sparks;
      if (sk) {
        for (var q = 0; q < sk.life.length; q++) {
          if (sk.life[q] <= 0) { continue; }
          sk.life[q] -= dt;
          sk.vel[q * 3 + 1] -= 12 * dt;
          sk.pos[q * 3] += sk.vel[q * 3] * dt;
          sk.pos[q * 3 + 1] += sk.vel[q * 3 + 1] * dt;
          sk.pos[q * 3 + 2] += sk.vel[q * 3 + 2] * dt;
          if (sk.life[q] <= 0 || sk.pos[q * 3 + 1] < 0.03) {
            sk.life[q] = 0;
            sk.pos[q * 3 + 1] = -5;
          }
        }
        sk.geo.attributes.position.needsUpdate = true;
        sk.light.intensity = Math.max(0, sk.light.intensity - dt * 6);
        sk.timer -= dt;
        if (sk.timer <= 0) {
          sk.timer = 2.5 + Math.random() * 4;
          var spot;
          if (Math.random() < 0.3 && trolley) {
            spot = [trolley.position.x, 8.0, trolley.position.z];
          } else {
            spot = sk.spots[Math.floor(Math.random() * sk.spots.length)];
          }
          var burstN = 10 + Math.floor(Math.random() * 8);
          for (var b2 = 0; b2 < sk.life.length && burstN > 0; b2++) {
            if (sk.life[b2] > 0) { continue; }
            burstN--;
            sk.life[b2] = 0.35 + Math.random() * 0.5;
            sk.pos[b2 * 3] = spot[0];
            sk.pos[b2 * 3 + 1] = spot[1];
            sk.pos[b2 * 3 + 2] = spot[2];
            var a = Math.random() * Math.PI * 2;
            var sv = 1.5 + Math.random() * 3;
            sk.vel[b2 * 3] = Math.cos(a) * sv;
            sk.vel[b2 * 3 + 1] = Math.random() * 2.5;
            sk.vel[b2 * 3 + 2] = Math.sin(a) * sv;
          }
          sk.light.position.set(spot[0], spot[1], spot[2]);
          sk.light.intensity = 2.2;
        }
      }

      // status LEDs blink, fans turn, chains sway, ticker scrolls
      for (var l2 = 0; l2 < animated.leds.length; l2++) {
        var led = animated.leds[l2];
        var on = Math.sin(t * led.speed * Math.PI + led.phase) > -0.2;
        led.mat.color.copy(on ? led.on : led.off);
      }
      for (var f2 = 0; f2 < animated.fans.length; f2++) {
        animated.fans[f2].rotation.z += dt * (2.2 + f2 * 0.5);
      }
      for (var c2 = 0; c2 < animated.chains.length; c2++) {
        var ch = animated.chains[c2];
        ch.group.rotation.x = Math.sin(t * 0.6 + ch.phase) * 0.05;
        ch.group.rotation.z = Math.cos(t * 0.45 + ch.phase) * 0.05;
      }
      if (animated.ticker) {
        animated.ticker.offset.x = (animated.ticker.offset.x + dt * 0.03) % 0.5;
      }

      // one hall lamp has a loose contact and flickers now and then
      var fl2 = animated.flicker;
      if (fl2) {
        fl2.t -= dt;
        if (fl2.t <= 0 && fl2.burst <= 0) {
          fl2.burst = 0.4 + Math.random() * 0.9;
          fl2.t = 7 + Math.random() * 12;
        }
        if (fl2.burst > 0) {
          fl2.burst -= dt;
          var lit = Math.random() > 0.45;
          fl2.light.intensity = lit ? fl2.base : 0.04;
          fl2.bulb.material.color.setHex(lit ? 0xffe6b0 : 0x2c2417);
          fl2.cone.material.opacity = lit ? 0.05 : 0.005;
          if (fl2.burst <= 0) {
            fl2.light.intensity = fl2.base;
            fl2.bulb.material.color.setHex(0xffe6b0);
            fl2.cone.material.opacity = 0.05;
          }
        }
      }
    }

    return {
      colliders: colliders,
      solids: solids,
      bounds: { min: -half + 1, max: half - 1 },
      crane: { trolley: trolley, cable: cable, load: hookLoad },
      update: update,
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
