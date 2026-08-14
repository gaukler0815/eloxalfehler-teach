// Procedural kart + driver meshes and their per-frame animation (wheel spin,
// steering, drift lean, spin-out rotation, boost flame, shield bubble,
// floating name tag).

import * as THREE from 'three';

function nameSprite(text, accent) {
  const c = document.createElement('canvas');
  c.width = 256; c.height = 64;
  const g = c.getContext('2d');
  g.font = 'bold 34px Trebuchet MS';
  g.textAlign = 'center'; g.textBaseline = 'middle';
  g.lineWidth = 7; g.strokeStyle = 'rgba(10,16,30,0.85)';
  g.strokeText(text, 128, 32);
  g.fillStyle = accent;
  g.fillText(text, 128, 32);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  const spr = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false }));
  spr.scale.set(5, 1.25, 1);
  spr.position.y = 3.4;
  return spr;
}

export class KartView {
  constructor(scene, char, { showName = true, label = null } = {}) {
    this.char = char;
    this.group = new THREE.Group();
    this.wheelSpin = 0;

    const color = new THREE.Color(char.color);
    const accent = new THREE.Color(char.accent);
    const bodyMat = new THREE.MeshStandardMaterial({ color, metalness: 0.35, roughness: 0.35 });
    const accentMat = new THREE.MeshStandardMaterial({ color: accent, metalness: 0.2, roughness: 0.5 });
    const darkMat = new THREE.MeshStandardMaterial({ color: 0x22242a, metalness: 0.1, roughness: 0.9 });
    const chromeMat = new THREE.MeshStandardMaterial({ color: 0xd8dce4, metalness: 0.8, roughness: 0.25 });

    // Chassis: a chunky rounded body with a nose.
    const body = new THREE.Mesh(new THREE.BoxGeometry(2.1, 0.7, 3.4), bodyMat);
    body.position.y = 0.75;
    body.castShadow = true;
    this.group.add(body);
    const nose = new THREE.Mesh(new THREE.CylinderGeometry(0.6, 1.0, 1.1, 4), bodyMat);
    nose.rotation.x = Math.PI / 2;
    nose.rotation.y = Math.PI / 4;
    nose.position.set(0, 0.72, 2.1);
    nose.castShadow = true;
    this.group.add(nose);
    const seatBack = new THREE.Mesh(new THREE.BoxGeometry(1.6, 0.9, 0.35), accentMat);
    seatBack.position.set(0, 1.35, -1.15);
    this.group.add(seatBack);
    const bumper = new THREE.Mesh(new THREE.BoxGeometry(2.3, 0.35, 0.4), chromeMat);
    bumper.position.set(0, 0.5, -1.75);
    this.group.add(bumper);

    // Exhausts.
    for (const px of [-0.55, 0.55]) {
      const ex = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.22, 0.8, 8), chromeMat);
      ex.rotation.x = 1.25;
      ex.position.set(px, 1.05, -1.9);
      this.group.add(ex);
    }

    // Steering wheel.
    const wheelCol = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, 0.7, 6), darkMat);
    wheelCol.rotation.x = -0.7;
    wheelCol.position.set(0, 1.35, 0.9);
    this.group.add(wheelCol);
    this.steeringWheel = new THREE.Mesh(new THREE.TorusGeometry(0.32, 0.07, 8, 16), darkMat);
    this.steeringWheel.rotation.x = -0.7 + Math.PI / 2;
    this.steeringWheel.position.set(0, 1.62, 1.12);
    this.group.add(this.steeringWheel);

    // Wheels: front pair steers, all spin.
    this.wheels = [];
    this.frontWheels = [];
    const wheelGeo = new THREE.CylinderGeometry(0.62, 0.62, 0.55, 14);
    wheelGeo.rotateZ(Math.PI / 2);
    const hubGeo = new THREE.CylinderGeometry(0.3, 0.3, 0.57, 10);
    hubGeo.rotateZ(Math.PI / 2);
    for (const [px, pz, front] of [[-1.15, 1.25, 1], [1.15, 1.25, 1], [-1.2, -1.15, 0], [1.2, -1.15, 0]]) {
      const holder = new THREE.Group();
      holder.position.set(px, 0.62, pz);
      const tire = new THREE.Mesh(wheelGeo, darkMat);
      tire.castShadow = true;
      const hub = new THREE.Mesh(hubGeo, accentMat);
      const spinner = new THREE.Group();
      spinner.add(tire, hub);
      holder.add(spinner);
      this.group.add(holder);
      this.wheels.push(spinner);
      if (front) this.frontWheels.push(holder);
    }

    // Driver: round head with helmet + visor in team colors, simple torso.
    const torso = new THREE.Mesh(new THREE.CylinderGeometry(0.45, 0.6, 0.9, 10), accentMat);
    torso.position.set(0, 1.55, -0.3);
    torso.castShadow = true;
    this.group.add(torso);
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.52, 14, 12), new THREE.MeshStandardMaterial({ color: 0xf5c9a2, roughness: 0.8 }));
    head.position.set(0, 2.3, -0.3);
    head.castShadow = true;
    this.group.add(head);
    const helmet = new THREE.Mesh(
      new THREE.SphereGeometry(0.58, 14, 10, 0, Math.PI * 2, 0, Math.PI * 0.55),
      bodyMat,
    );
    helmet.position.copy(head.position).y += 0.08;
    this.group.add(helmet);
    const visor = new THREE.Mesh(new THREE.SphereGeometry(0.45, 10, 8, -0.7, 1.4, 1.1, 0.7), darkMat);
    visor.position.copy(head.position);
    visor.position.z += 0.1;
    this.group.add(visor);

    // Boost flames (hidden unless boosting).
    this.flames = [];
    for (const px of [-0.55, 0.55]) {
      const flame = new THREE.Mesh(
        new THREE.ConeGeometry(0.22, 1.4, 8),
        new THREE.MeshBasicMaterial({ color: 0xffa030, transparent: true, opacity: 0.9 }),
      );
      flame.rotation.x = Math.PI / 2 + 0.35;
      flame.position.set(px, 1.05, -2.6);
      flame.visible = false;
      this.group.add(flame);
      this.flames.push(flame);
    }

    // Shield bubble.
    this.shield = new THREE.Mesh(
      new THREE.SphereGeometry(2.6, 18, 12),
      new THREE.MeshBasicMaterial({ color: 0x66ccff, transparent: true, opacity: 0.22, depthWrite: false }),
    );
    this.shield.position.y = 1.2;
    this.shield.visible = false;
    this.group.add(this.shield);

    // Drop shadow helper blob (cheap, in addition to real shadows).
    const blob = new THREE.Mesh(
      new THREE.CircleGeometry(1.7, 18),
      new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.25, depthWrite: false }),
    );
    blob.rotation.x = -Math.PI / 2;
    blob.position.y = 0.06;
    this.group.add(blob);

    if (showName) this.group.add(nameSprite(label || char.name, '#' + accent.getHexString()));

    scene.add(this.group);
  }

  // state: { x, z, heading, speed, steer, drifting, driftDir, boostT, spinT,
  //          spinPhase, shieldT }
  update(dt, state, t) {
    const g = this.group;
    g.position.set(state.x, 0, state.z);
    const spinExtra = state.spinT > 0 ? state.spinPhase : 0;
    const driftLean = state.drifting ? state.driftDir * 0.35 : 0;
    g.rotation.set(0, state.heading + spinExtra + driftLean, 0);
    g.rotation.z = -driftLean * 0.35 - (state.steer || 0) * 0.05 * Math.min(1, state.speed / 20);

    this.wheelSpin += state.speed * dt / 0.62;
    for (const w of this.wheels) w.rotation.x = this.wheelSpin;
    const steerAng = (state.steer || 0) * 0.45;
    for (const w of this.frontWheels) w.rotation.y = steerAng;
    this.steeringWheel.rotation.z = -steerAng * 2;

    const boosting = state.boostT > 0;
    for (const f of this.flames) {
      f.visible = boosting;
      if (boosting) {
        const s = 0.8 + Math.sin(t * 40 + f.position.x) * 0.3;
        f.scale.set(s, 1 + Math.random() * 0.5, s);
      }
    }
    this.shield.visible = state.shieldT > 0;
    if (this.shield.visible) {
      this.shield.material.opacity = 0.14 + 0.1 * Math.sin(t * 6) + (state.shieldT < 2 ? 0.1 * Math.sin(t * 25) : 0);
    }
  }

  dispose(scene) {
    scene.remove(this.group);
  }
}
