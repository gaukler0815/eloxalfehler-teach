// Builds the visual world for the Canyon-Kurs: sky, desert terrain, the road
// with curbs and barriers, rock mesas, cacti, sponsor banners, start gate,
// item boxes and clouds. Everything is generated procedurally – no asset
// files, loads instantly.

import * as THREE from 'three';
import { drawLogo, BRAND } from './branding.js';

function seededRng(seed) {
  return () => {
    seed = (seed * 1103515245 + 12345) % 2147483648;
    return seed / 2147483648;
  };
}

function canvasTexture(w, h, draw) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  draw(c.getContext('2d'), w, h);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return tex;
}

export function buildWorld(scene, track) {
  const rng = seededRng(1337);
  const world = { itemBoxMeshes: [], animated: [] };

  // ------------------------------------------------------------------ sky
  scene.background = new THREE.Color(0x87c8ec);
  scene.fog = new THREE.Fog(0xd8ecf5, 420, 1500);

  // Simple equirect environment for metallic (anodized) kart paint.
  {
    const c = document.createElement('canvas');
    c.width = 128; c.height = 64;
    const g = c.getContext('2d');
    const grad = g.createLinearGradient(0, 0, 0, 64);
    grad.addColorStop(0, '#bfe6ff');
    grad.addColorStop(0.5, '#87c8ec');
    grad.addColorStop(0.52, '#e8c890');
    grad.addColorStop(1, '#b8935e');
    g.fillStyle = grad;
    g.fillRect(0, 0, 128, 64);
    g.fillStyle = '#fffbe8';
    g.beginPath(); g.arc(36, 14, 7, 0, Math.PI * 2); g.fill();
    const env = new THREE.CanvasTexture(c);
    env.mapping = THREE.EquirectangularReflectionMapping;
    env.colorSpace = THREE.SRGBColorSpace;
    scene.environment = env;
  }

  const skyGeo = new THREE.SphereGeometry(1400, 24, 12);
  const skyMat = new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false, fog: false,
    uniforms: {
      top: { value: new THREE.Color(0x3f9ede) },
      mid: { value: new THREE.Color(0x9adcf5) },
      bottom: { value: new THREE.Color(0xf2d9a8) },
    },
    vertexShader: `varying vec3 vP; void main(){ vP = position; gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.0); }`,
    fragmentShader: `
      uniform vec3 top, mid, bottom; varying vec3 vP;
      void main(){
        float h = normalize(vP).y;
        vec3 c = h > 0.25 ? mix(mid, top, smoothstep(0.25, 0.9, h))
                          : mix(bottom, mid, smoothstep(-0.05, 0.25, h));
        gl_FragColor = vec4(c, 1.0);
      }`,
  });
  scene.add(new THREE.Mesh(skyGeo, skyMat));

  // Clouds: flat white puffs drifting slowly.
  const cloudMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.88, fog: false });
  const clouds = new THREE.Group();
  for (let i = 0; i < 16; i++) {
    const puff = new THREE.Group();
    const n = 3 + Math.floor(rng() * 3);
    for (let j = 0; j < n; j++) {
      const s = 22 + rng() * 30;
      const m = new THREE.Mesh(new THREE.SphereGeometry(s, 8, 6), cloudMat);
      m.position.set((j - n / 2) * s * 0.9, rng() * 8, rng() * 12);
      m.scale.y = 0.4;
      puff.add(m);
    }
    const ang = rng() * Math.PI * 2;
    const rad = 420 + rng() * 420;
    puff.position.set(Math.cos(ang) * rad, 280 + rng() * 160, Math.sin(ang) * rad);
    clouds.add(puff);
  }
  scene.add(clouds);
  world.animated.push((dt) => { clouds.rotation.y += dt * 0.004; });

  // ------------------------------------------------------------------ light
  scene.add(new THREE.HemisphereLight(0xbfe3ff, 0xd8b078, 0.95));
  const sun = new THREE.DirectionalLight(0xfff2d8, 2.0);
  sun.position.set(220, 320, 140);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  const sc = 260;
  sun.shadow.camera.left = -sc; sun.shadow.camera.right = sc;
  sun.shadow.camera.top = sc; sun.shadow.camera.bottom = -sc;
  sun.shadow.camera.far = 900;
  sun.shadow.bias = -0.0004;
  scene.add(sun);
  world.sun = sun;

  // ---------------------------------------------------------------- terrain
  const groundGeo = new THREE.CircleGeometry(1300, 96);
  groundGeo.rotateX(-Math.PI / 2);
  const pos = groundGeo.attributes.position;
  const colors = [];
  const sand = new THREE.Color(0xdfb571), sandDark = new THREE.Color(0xc99a55), sandRed = new THREE.Color(0xd39a62);
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), z = pos.getZ(i);
    const n = Math.sin(x * 0.013) * Math.cos(z * 0.011) + Math.sin(x * 0.041 + z * 0.037) * 0.5;
    const c = sand.clone().lerp(n > 0.3 ? sandRed : sandDark, Math.abs(n) * 0.55);
    colors.push(c.r, c.g, c.b);
  }
  groundGeo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  const ground = new THREE.Mesh(groundGeo, new THREE.MeshLambertMaterial({ vertexColors: true }));
  ground.position.y = -0.08;
  ground.receiveShadow = true;
  scene.add(ground);

  // ------------------------------------------------------------------ road
  const N = track.points.length;
  const roadW = track.halfWidth;
  const roadTex = canvasTexture(256, 256, (g) => {
    g.fillStyle = '#4a4d55'; g.fillRect(0, 0, 256, 256);
    // asphalt noise
    for (let i = 0; i < 2600; i++) {
      const v = 60 + Math.random() * 40;
      g.fillStyle = `rgba(${v},${v},${v + 6},0.25)`;
      g.fillRect(Math.random() * 256, Math.random() * 256, 2, 2);
    }
    // edge lines (like the reference: yellow inside, white outside)
    g.fillStyle = '#f8f8f2'; g.fillRect(6, 0, 7, 256);
    g.fillStyle = '#e8b93a'; g.fillRect(243, 0, 7, 256);
    // center dashes
    g.fillStyle = '#f0f0e8';
    for (let y = 0; y < 256; y += 64) g.fillRect(124, y, 8, 34);
  });
  roadTex.wrapS = roadTex.wrapT = THREE.RepeatWrapping;

  function ribbon(halfLeft, halfRight, y, material, uvScaleV = 0.05) {
    const verts = [], uvs = [], idx = [];
    for (let i = 0; i <= N; i++) {
      const p = track.points[i % N];
      verts.push(p.x + p.nx * halfLeft, y, p.z + p.nz * halfLeft);
      verts.push(p.x + p.nx * halfRight, y, p.z + p.nz * halfRight);
      const v = p.s * uvScaleV;
      uvs.push(0, v, 1, v);
      if (i < N) {
        const a = i * 2;
        idx.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
    geo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    geo.setIndex(idx);
    geo.computeVertexNormals();
    material.side = THREE.DoubleSide; // ribbon winding varies with curve direction
    const mesh = new THREE.Mesh(geo, material);
    mesh.receiveShadow = true;
    return mesh;
  }

  scene.add(ribbon(roadW, -roadW, 0.02, new THREE.MeshLambertMaterial({ map: roadTex })));

  // Curbs: Jacobi-blue/white stripes just outside the road edges.
  const curbTex = canvasTexture(64, 64, (g) => {
    g.fillStyle = BRAND.blue; g.fillRect(0, 0, 64, 32);
    g.fillStyle = '#f5f0e6'; g.fillRect(0, 32, 64, 32);
  });
  curbTex.wrapS = curbTex.wrapT = THREE.RepeatWrapping;
  const curbMat = new THREE.MeshLambertMaterial({ map: curbTex });
  scene.add(ribbon(roadW + 1.4, roadW, 0.04, curbMat, 0.12));
  scene.add(ribbon(-roadW, -roadW - 1.4, 0.04, curbMat, 0.12));

  // Finish line: checkered strip at s = 0.
  const finTex = canvasTexture(128, 32, (g) => {
    for (let x = 0; x < 8; x++) for (let y = 0; y < 2; y++) {
      g.fillStyle = (x + y) % 2 ? '#111' : '#fafafa';
      g.fillRect(x * 16, y * 16, 16, 16);
    }
  });
  const p0 = track.sample(0);
  const fin = new THREE.Mesh(
    new THREE.PlaneGeometry(roadW * 2, 4),
    new THREE.MeshLambertMaterial({ map: finTex }),
  );
  fin.rotation.x = -Math.PI / 2;
  fin.rotation.z = -Math.atan2(p0.tx, p0.tz);
  fin.position.set(p0.x, 0.05, p0.z);
  scene.add(fin);

  // Barrier posts along both sides (instanced).
  const postGeo = new THREE.CylinderGeometry(0.45, 0.5, 1.6, 8);
  const postMatRed = new THREE.MeshLambertMaterial({ color: 0xd8402c });
  const postMatWhite = new THREE.MeshLambertMaterial({ color: 0xf2ede2 });
  const postStep = 14;
  const nPosts = Math.floor(track.length / postStep);
  const instRed = new THREE.InstancedMesh(postGeo, postMatRed, nPosts);
  const instWhite = new THREE.InstancedMesh(postGeo, postMatWhite, nPosts);
  instRed.castShadow = instWhite.castShadow = true;
  const m4 = new THREE.Matrix4();
  let iR = 0, iW = 0;
  for (let i = 0; i < nPosts; i++) {
    const p = track.sample(i * postStep);
    const side = i % 2 ? 1 : -1;
    const off = track.halfWidth + 6.8;
    m4.setPosition(p.x + p.nx * off * side, 0.8, p.z + p.nz * off * side);
    if (i % 4 < 2) instRed.setMatrixAt(iR++, m4); else instWhite.setMatrixAt(iW++, m4);
  }
  instRed.count = iR; instWhite.count = iW;
  scene.add(instRed, instWhite);

  // ---------------------------------------------------------------- helpers
  function farFromTrack(x, z, minDist) {
    for (let i = 0; i < N; i += 6) {
      const p = track.points[i];
      const dx = x - p.x, dz = z - p.z;
      if (dx * dx + dz * dz < minDist * minDist) return false;
    }
    return true;
  }

  // ------------------------------------------------------------------ mesas
  const mesaMats = [0xb5653c, 0xa85834, 0xc2764a].map((c) => new THREE.MeshLambertMaterial({ color: c }));
  for (let i = 0; i < 26; i++) {
    const ang = rng() * Math.PI * 2;
    const rad = 320 + rng() * 700;
    const x = Math.cos(ang) * rad, z = Math.sin(ang) * rad;
    if (!farFromTrack(x, z, 70)) continue;
    const h = 40 + rng() * 90, r = 30 + rng() * 55;
    const mesa = new THREE.Group();
    const body = new THREE.Mesh(new THREE.CylinderGeometry(r * (0.7 + rng() * 0.2), r, h, 9, 1), mesaMats[i % 3]);
    body.position.y = h / 2 - 2;
    mesa.add(body);
    const cap = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.55, r * 0.75, h * 0.22, 9, 1), mesaMats[(i + 1) % 3]);
    cap.position.y = h + h * 0.1 - 2;
    mesa.add(cap);
    mesa.position.set(x, 0, z);
    mesa.rotation.y = rng() * Math.PI;
    scene.add(mesa);
  }

  // Small rocks near the road.
  const rockGeo = new THREE.DodecahedronGeometry(1.6, 0);
  const rockMat = new THREE.MeshLambertMaterial({ color: 0x9c6b45 });
  const rocks = new THREE.InstancedMesh(rockGeo, rockMat, 90);
  rocks.castShadow = true;
  let rockI = 0;
  for (let i = 0; i < 300 && rockI < 90; i++) {
    const s = rng() * track.length;
    const p = track.sample(s);
    const side = rng() > 0.5 ? 1 : -1;
    const off = track.halfWidth + 10 + rng() * 30;
    const x = p.x + p.nx * off * side, z = p.z + p.nz * off * side;
    const sc2 = 0.5 + rng() * 1.6;
    m4.makeRotationY(rng() * Math.PI);
    m4.scale(new THREE.Vector3(sc2, sc2 * (0.6 + rng() * 0.5), sc2));
    m4.setPosition(x, 0.4 * sc2, z);
    rocks.setMatrixAt(rockI++, m4);
  }
  rocks.count = rockI;
  scene.add(rocks);

  // ------------------------------------------------------------------ cacti
  const cactusMat = new THREE.MeshLambertMaterial({ color: 0x4e8f3a });
  for (let i = 0; i < 40; i++) {
    const s = rng() * track.length;
    const p = track.sample(s);
    const side = rng() > 0.5 ? 1 : -1;
    const off = track.halfWidth + 14 + rng() * 55;
    const x = p.x + p.nx * off * side, z = p.z + p.nz * off * side;
    const g = new THREE.Group();
    const h = 3 + rng() * 3;
    const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.6, h, 7), cactusMat);
    trunk.position.y = h / 2; trunk.castShadow = true;
    g.add(trunk);
    for (let a = 0; a < 2; a++) {
      const arm = new THREE.Mesh(new THREE.CylinderGeometry(0.32, 0.36, h * 0.5, 6), cactusMat);
      const sideA = a ? 1 : -1;
      arm.position.set(sideA * 0.9, h * (0.45 + rng() * 0.2), 0);
      arm.rotation.z = sideA * -0.5;
      arm.castShadow = true;
      g.add(arm);
    }
    g.position.set(x, 0, z);
    g.rotation.y = rng() * Math.PI * 2;
    scene.add(g);
  }

  // ---------------------------------------------------------------- banners
  // A: light cloth with the full Jacobi Eloxal logo (badge + wordmark).
  const bannerTexA = canvasTexture(512, 128, (g) => {
    g.fillStyle = '#f2ede0'; g.fillRect(0, 0, 512, 128);
    g.fillStyle = BRAND.blue; g.fillRect(0, 0, 512, 10); g.fillRect(0, 118, 512, 10);
    drawLogo(g, 130, 64, 1.15, { wordmark: true });
  });
  // B: dark hall cloth with gold event lettering + small JX badge.
  const bannerTexB = canvasTexture(512, 128, (g) => {
    g.fillStyle = BRAND.hallDark; g.fillRect(0, 0, 512, 128);
    g.fillStyle = BRAND.gold; g.fillRect(0, 0, 512, 8); g.fillRect(0, 120, 512, 8);
    drawLogo(g, 62, 64, 0.9, { wordmark: false });
    g.fillStyle = BRAND.gold; g.font = 'bold 58px Trebuchet MS';
    g.textAlign = 'left'; g.textBaseline = 'middle';
    g.fillText('ELOXAL KART', 118, 68);
  });
  const poleMat = new THREE.MeshLambertMaterial({ color: 0x8a8f98 });
  for (let i = 0; i < 8; i++) {
    const s = (i / 8 + 0.06) * track.length;
    const p = track.sample(s);
    const side = i % 2 ? 1 : -1;
    const off = track.halfWidth + 9;
    const g = new THREE.Group();
    for (const px of [-7, 7]) {
      const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.22, 6.4, 6), poleMat);
      pole.position.set(px, 3.2, 0);
      g.add(pole);
    }
    const cloth = new THREE.Mesh(
      new THREE.PlaneGeometry(14, 3.5),
      new THREE.MeshLambertMaterial({ map: i % 2 ? bannerTexA : bannerTexB, side: THREE.DoubleSide }),
    );
    cloth.position.y = 4.6;
    g.add(cloth);
    g.position.set(p.x + p.nx * off * side, 0, p.z + p.nz * off * side);
    g.rotation.y = -Math.atan2(p.tx, p.tz) + Math.PI / 2 + (side > 0 ? Math.PI : 0);
    scene.add(g);
  }

  // -------------------------------------------------------------- start gate
  {
    const p = track.sample(4);
    const gate = new THREE.Group();
    // Anodized-blue pillars with gold caps – Jacobi colors.
    const pillarMat = new THREE.MeshStandardMaterial({ color: 0x1e74dc, metalness: 0.6, roughness: 0.3 });
    const capMat = new THREE.MeshStandardMaterial({ color: 0xf5a81c, metalness: 0.5, roughness: 0.35 });
    for (const side of [-1, 1]) {
      const pillar = new THREE.Mesh(new THREE.BoxGeometry(1.8, 11, 1.8), pillarMat);
      pillar.position.set(side * (roadW + 3), 5.5, 0);
      pillar.castShadow = true;
      gate.add(pillar);
      const cap = new THREE.Mesh(new THREE.BoxGeometry(2.2, 0.5, 2.2), capMat);
      cap.position.set(side * (roadW + 3), 11.2, 0);
      gate.add(cap);
    }
    const bannerTex = canvasTexture(1024, 128, (g) => {
      g.fillStyle = BRAND.hallDark; g.fillRect(0, 0, 1024, 128);
      g.fillStyle = BRAND.gold; g.fillRect(0, 0, 1024, 6); g.fillRect(0, 122, 1024, 6);
      drawLogo(g, 80, 64, 1.05, { wordmark: false });
      g.fillStyle = BRAND.silver; g.font = '800 62px system-ui, sans-serif';
      g.textAlign = 'left'; g.textBaseline = 'middle';
      g.fillText('JACOBI', 150, 62);
      g.fillStyle = BRAND.gold; g.font = '700 44px system-ui, sans-serif';
      g.fillText('E L O X A L   K A R T', 408, 66);
    });
    const beam = new THREE.Mesh(
      new THREE.BoxGeometry(roadW * 2 + 9.5, 3.2, 1.6),
      [new THREE.MeshLambertMaterial({ color: 0x241d33 }), new THREE.MeshLambertMaterial({ color: 0x241d33 }),
       new THREE.MeshLambertMaterial({ color: 0x241d33 }), new THREE.MeshLambertMaterial({ color: 0x241d33 }),
       new THREE.MeshLambertMaterial({ map: bannerTex }), new THREE.MeshLambertMaterial({ map: bannerTex })],
    );
    beam.position.y = 9.6;
    beam.castShadow = true;
    gate.add(beam);
    gate.position.set(p.x, 0, p.z);
    gate.rotation.y = Math.atan2(p.tx, p.tz);
    scene.add(gate);
  }

  // -------------------------------------------------------------- item boxes
  // Anodized-blue crates with the JX question mark – "frisch aus dem Bad".
  const boxTex = canvasTexture(128, 128, (g) => {
    const grad = g.createLinearGradient(0, 0, 128, 128);
    grad.addColorStop(0, BRAND.blueLight); grad.addColorStop(1, BRAND.blue);
    g.fillStyle = grad; g.fillRect(0, 0, 128, 128);
    g.strokeStyle = BRAND.gold; g.lineWidth = 10; g.strokeRect(5, 5, 118, 118);
    g.fillStyle = '#ffffff'; g.font = 'bold 88px Trebuchet MS';
    g.textAlign = 'center'; g.textBaseline = 'middle';
    g.fillText('?', 64, 70);
  });
  const boxMat = new THREE.MeshLambertMaterial({
    map: boxTex, transparent: true, opacity: 0.93, emissive: 0x102a55,
  });
  for (const box of track.itemBoxes) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(2.4, 2.4, 2.4), boxMat);
    m.position.set(box.x, 1.7, box.z);
    m.castShadow = true;
    scene.add(m);
    world.itemBoxMeshes[box.id] = m;
  }
  world.animated.push((dt, t) => {
    for (const m of world.itemBoxMeshes) {
      if (!m.visible) continue;
      m.rotation.y += dt * 1.6;
      m.rotation.x += dt * 0.9;
      m.position.y = 1.7 + Math.sin(t * 2.2 + m.position.x) * 0.25;
    }
  });

  // ----------------------------------------------- Jacobi Eloxal factory set
  // Finds a clear spot beside the track: scans arc positions until the point
  // at `offset` from the centerline keeps `minDist` to every track sample.
  function findSpot(fStart, offset, minDist) {
    for (let f = 0; f < 1; f += 0.015) {
      const s = ((fStart + f) % 1) * track.length;
      const p = track.sample(s);
      for (const side of [1, -1]) {
        const x = p.x + p.nx * offset * side, z = p.z + p.nz * offset * side;
        if (farFromTrack(x, z, minDist)) {
          return { x, z, angle: -Math.atan2(p.tx, p.tz) + (side > 0 ? Math.PI : 0) };
        }
      }
    }
    return null;
  }

  // The anodizing hall: big shed with sawtooth roof and the logo on the wall.
  {
    const spot = findSpot(0.62, 105, 78);
    if (spot) {
      const hall = new THREE.Group();
      const wallMat = new THREE.MeshLambertMaterial({ color: 0x9aa2ae });
      const body = new THREE.Mesh(new THREE.BoxGeometry(120, 26, 55), wallMat);
      body.position.y = 13;
      body.castShadow = true;
      hall.add(body);
      const roofMat = new THREE.MeshLambertMaterial({ color: 0x5a6472 });
      for (let i = 0; i < 5; i++) {
        const seg = new THREE.Mesh(new THREE.CylinderGeometry(6, 6, 55, 3, 1), roofMat);
        seg.rotation.z = Math.PI / 2;
        seg.rotation.y = Math.PI / 2;
        seg.position.set(-48 + i * 24, 28.5, 0);
        hall.add(seg);
      }
      const signTex = canvasTexture(512, 128, (g) => {
        g.fillStyle = BRAND.hallDark; g.fillRect(0, 0, 512, 128);
        g.strokeStyle = BRAND.gold; g.lineWidth = 6; g.strokeRect(3, 3, 506, 122);
        drawLogo(g, 120, 64, 1.3, { wordmark: true });
      });
      const sign = new THREE.Mesh(
        new THREE.PlaneGeometry(64, 16),
        new THREE.MeshLambertMaterial({ map: signTex }),
      );
      sign.position.set(0, 17, 27.7);
      hall.add(sign);
      // Chimney with a soft smoke puff.
      const chimney = new THREE.Mesh(new THREE.CylinderGeometry(2, 2.4, 18, 8), roofMat);
      chimney.position.set(44, 34, -12);
      hall.add(chimney);
      hall.position.set(spot.x, 0, spot.z);
      hall.rotation.y = spot.angle;
      scene.add(hall);
    }
  }

  // A row of anodizing baths with glowing electrolyte – classic Eloxal.
  {
    const spot = findSpot(0.30, 26, 20);
    if (spot) {
      const baths = new THREE.Group();
      const tankMat = new THREE.MeshLambertMaterial({ color: 0x4a5468 });
      const rimMat = new THREE.MeshLambertMaterial({ color: 0xb8a375 });
      const liquids = [];
      const tones = [0x2e8fff, 0x39d7ff, 0x2e8fff];
      for (let i = 0; i < 3; i++) {
        const tank = new THREE.Mesh(new THREE.BoxGeometry(7, 2.6, 5), tankMat);
        tank.position.set(i * 8 - 8, 1.3, 0);
        tank.castShadow = true;
        baths.add(tank);
        const rim = new THREE.Mesh(new THREE.BoxGeometry(7.6, 0.4, 5.6), rimMat);
        rim.position.set(i * 8 - 8, 2.7, 0);
        baths.add(rim);
        const liquid = new THREE.Mesh(
          new THREE.PlaneGeometry(6.4, 4.4),
          new THREE.MeshBasicMaterial({ color: tones[i], transparent: true, opacity: 0.85 }),
        );
        liquid.rotation.x = -Math.PI / 2;
        liquid.position.set(i * 8 - 8, 2.62, 0);
        baths.add(liquid);
        liquids.push(liquid);
      }
      baths.position.set(spot.x, 0, spot.z);
      baths.rotation.y = spot.angle;
      scene.add(baths);
      world.animated.push((dt, t) => {
        liquids.forEach((l, i) => { l.material.opacity = 0.7 + Math.sin(t * 2.4 + i * 1.7) * 0.18; });
      });
    }
  }

  // Stacks of freshly anodized aluminum profiles on pallets.
  {
    const profMats = [
      new THREE.MeshStandardMaterial({ color: 0xc8ccd4, metalness: 0.85, roughness: 0.25 }),
      new THREE.MeshStandardMaterial({ color: 0x4d8fe0, metalness: 0.8, roughness: 0.3 }),
      new THREE.MeshStandardMaterial({ color: 0xd8a860, metalness: 0.8, roughness: 0.3 }),
    ];
    const palletMat = new THREE.MeshLambertMaterial({ color: 0x8a6a42 });
    for (let n = 0; n < 3; n++) {
      const spot = findSpot(0.08 + n * 0.28, 19, 15);
      if (!spot) continue;
      const stack = new THREE.Group();
      const pallet = new THREE.Mesh(new THREE.BoxGeometry(6.5, 0.5, 4.5), palletMat);
      pallet.position.y = 0.25;
      stack.add(pallet);
      for (let layer = 0; layer < 3; layer++) {
        for (let row = 0; row < 4; row++) {
          const prof = new THREE.Mesh(new THREE.BoxGeometry(6, 0.55, 0.8), profMats[n % 3]);
          prof.position.set(0, 0.85 + layer * 0.6, row * 1.05 - 1.6);
          prof.castShadow = true;
          stack.add(prof);
        }
      }
      stack.position.set(spot.x, 0, spot.z);
      stack.rotation.y = spot.angle + 0.3 * n;
      scene.add(stack);
    }
  }

  world.update = (dt, t) => { for (const f of world.animated) f(dt, t); };
  return world;
}
