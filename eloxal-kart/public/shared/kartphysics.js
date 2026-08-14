// Arcade kart physics. One pure step function over a plain state object so
// the same code drives the local player (client), the bots (server or
// offline client) and can later move fully server-side.

import { PHYS } from './config.js';

export function makeKartState(x, z, heading, charStats = { accel: 1, top: 1 }) {
  return {
    x, z, heading,
    speed: 0,
    steer: 0,           // smoothed steering -1..1
    drifting: false,
    driftDir: 0,
    driftCharge: 0,
    boostT: 0,
    spinT: 0,
    spinPhase: 0,       // visual rotation while spun out
    shieldT: 0,
    offroad: false,
    trackS: 0,          // arc position hint, maintained by caller/race sim
    accelMul: charStats.accel,
    topMul: charStats.top,
  };
}

// input: { throttle: -1..1, steer: -1..1, drift: bool }
export function stepKart(k, input, dt, track) {
  k.boostT = Math.max(0, k.boostT - dt);
  k.shieldT = Math.max(0, k.shieldT - dt);

  if (k.spinT > 0) {
    // Hit by an item: spin in place, bleed speed, ignore inputs.
    k.spinT = Math.max(0, k.spinT - dt);
    k.spinPhase += dt * 10;
    k.speed *= Math.pow(0.2, dt);
    k.drifting = false;
    k.driftCharge = 0;
  } else {
    k.spinPhase = 0;

    // --- Steering (smoothed toward input) ---
    const steerTarget = Math.max(-1, Math.min(1, input.steer || 0));
    const steerLerp = 1 - Math.pow(0.0008, dt);
    k.steer += (steerTarget - k.steer) * steerLerp;

    // --- Drift state machine ---
    const wantDrift = !!input.drift && Math.abs(k.steer) > 0.25 && k.speed > PHYS.driftMinSpeed;
    if (!k.drifting && wantDrift) {
      k.drifting = true;
      k.driftDir = Math.sign(k.steer) || 1;
      k.driftCharge = 0;
    }
    if (k.drifting) {
      if (!input.drift || k.speed < PHYS.driftMinSpeed * 0.6) {
        // Drift released: pay out the mini-turbo.
        if (k.driftCharge >= PHYS.driftBoost2) k.boostT = Math.max(k.boostT, PHYS.boostTimeBig);
        else if (k.driftCharge >= PHYS.driftBoost1) k.boostT = Math.max(k.boostT, PHYS.boostTimeSmall);
        k.drifting = false;
        k.driftCharge = 0;
      } else {
        k.driftCharge += dt;
      }
    }

    // --- Longitudinal ---
    const boosting = k.boostT > 0;
    const surface = k.offroad && !boosting ? PHYS.offroadFactor : 1;
    const top = PHYS.maxSpeed * k.topMul * surface * (boosting ? PHYS.boostSpeedFactor : 1);
    const throttle = Math.max(-1, Math.min(1, input.throttle || 0));

    if (boosting) {
      k.speed = Math.min(top, k.speed + PHYS.boostAccel * dt);
    } else if (throttle > 0) {
      if (k.speed < top) k.speed = Math.min(top, k.speed + PHYS.accel * k.accelMul * throttle * dt);
      else k.speed = Math.max(top, k.speed - PHYS.drag * 2 * dt); // over top speed: settle down
    } else if (throttle < 0) {
      k.speed = Math.max(-PHYS.maxReverse, k.speed + PHYS.brake * throttle * dt);
    } else {
      const drag = k.offroad ? PHYS.offroadDrag : PHYS.drag;
      k.speed = k.speed > 0 ? Math.max(0, k.speed - drag * dt) : Math.min(0, k.speed + drag * dt);
    }

    // --- Yaw ---
    const speedFrac = Math.min(1, Math.abs(k.speed) / PHYS.maxSpeed);
    const turnScale = 1 - PHYS.turnSpeedFalloff * speedFrac;
    let yawRate = PHYS.turnRate * turnScale * k.steer;
    if (k.drifting) {
      yawRate = PHYS.turnRate * turnScale * (k.driftDir * PHYS.driftTurnBonus * 0.8 + k.steer * 0.55);
    }
    if (k.speed < 0) yawRate = -yawRate;
    k.heading += yawRate * Math.min(1, Math.abs(k.speed) / 6) * dt;
  }

  // --- Integrate position (drift slides slightly sideways) ---
  let dirX = Math.sin(k.heading), dirZ = Math.cos(k.heading);
  if (k.drifting) {
    const slide = (1 - PHYS.driftSideGrip) * k.driftDir;
    const sx = Math.sin(k.heading + Math.PI / 2) * slide;
    const sz = Math.cos(k.heading + Math.PI / 2) * slide;
    const d = Math.hypot(dirX + sx, dirZ + sz) || 1;
    dirX = (dirX + sx) / d; dirZ = (dirZ + sz) / d;
  }
  k.x += dirX * k.speed * dt;
  k.z += dirZ * k.speed * dt;

  // --- Track relation: offroad flag + soft outer wall ---
  k.trackS = track.closestS(k.x, k.z, k.trackS);
  const lat = track.lateralOffset(k.x, k.z, k.trackS);
  k.offroad = Math.abs(lat) > track.halfWidth + 0.6;
  const maxLat = track.halfWidth + PHYS.wallMargin;
  if (Math.abs(lat) > maxLat) {
    const p = track.sample(k.trackS);
    const clamped = Math.sign(lat) * maxLat;
    k.x = p.x + p.nx * clamped;
    k.z = p.z + p.nz * clamped;
    k.speed *= Math.pow(0.05, dt); // scrubbing along the wall costs speed
  }
  return k;
}

// Symmetric sphere push so karts do not overlap. Mutates both states.
export function resolveKartCollision(a, b, radius) {
  const dx = b.x - a.x, dz = b.z - a.z;
  const d = Math.hypot(dx, dz);
  const min = radius * 2;
  if (d > 0.0001 && d < min) {
    const push = (min - d) / 2;
    const nx = dx / d, nz = dz / d;
    a.x -= nx * push; a.z -= nz * push;
    b.x += nx * push; b.z += nz * push;
    // Gentle speed exchange to make bumps feel physical.
    const avg = (a.speed + b.speed) / 2;
    a.speed = a.speed * 0.7 + avg * 0.3;
    b.speed = b.speed * 0.7 + avg * 0.3;
    return true;
  }
  return false;
}
