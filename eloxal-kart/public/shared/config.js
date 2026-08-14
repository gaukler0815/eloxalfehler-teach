// Central tuning constants for Eloxal Kart. Shared by server and client so
// both sides simulate with identical numbers.

export const PHYS = {
  maxSpeed: 36,          // units/s on asphalt
  maxReverse: 10,
  accel: 22,             // forward acceleration units/s^2
  brake: 40,
  drag: 8,               // passive deceleration when coasting
  offroadFactor: 0.45,   // top-speed multiplier off the road
  offroadDrag: 26,
  turnRate: 1.9,         // rad/s at full steer, low speed
  turnSpeedFalloff: 0.55,// how much steering tightens at speed (0..1)
  driftTurnBonus: 1.45,  // extra yaw while drifting
  driftMinSpeed: 16,
  driftSideGrip: 0.82,   // lateral slide factor while drifting
  boostSpeedFactor: 1.42,
  boostAccel: 55,
  spinDuration: 1.5,     // seconds of lost control after a hit
  spinSpeedFactor: 0.25, // speed kept when hit
  kartRadius: 2.1,       // collision radius kart vs kart / hazards
  wallMargin: 6.5,       // how far beyond the road edge the invisible wall sits
  driftBoost1: 0.9,      // charge seconds for small boost
  driftBoost2: 2.0,      // charge seconds for big boost
  boostTimeSmall: 0.85,
  boostTimeBig: 1.5,
};

export const RACE = {
  laps: 3,
  maxRacers: 8,
  countdown: 3.2,        // seconds of 3-2-1
  resultsTime: 14,       // seconds the podium screen stays before next lobby
  finishTimeout: 45,     // after first finisher, race force-ends
  boxRespawn: 4,         // item box respawn seconds
  rouletteTime: 1.1,     // item slot animation time
};

export const NET = {
  stateHz: 15,           // client -> server kart state rate
  snapshotHz: 15,        // server -> clients snapshot rate
  simHz: 30,             // server simulation rate
  interpDelay: 0.12,     // seconds remote karts are rendered in the past
};

// Item catalogue. German display names on the Eloxal theme, English ids.
export const ITEMS = {
  bolt:   { name: 'Lichtbogen',    desc: 'Geradeaus-Blitz, prallt von Banden ab' },
  seeker: { name: 'Zielsucher',    desc: 'Verfolgt den Fahrer vor dir' },
  turbo:  { name: 'Strom-Boost',   desc: 'Volle Badspannung – kurzer Schub' },
  barrel: { name: 'Säurefass',     desc: 'Hindernis für die Verfolger' },
  shield: { name: 'Eloxalschicht', desc: 'Harte Oxidschicht blockt einen Treffer' },
};

export const PROJ = {
  boltSpeed: 58,
  boltLife: 6,
  boltBounces: 3,
  seekerSpeed: 46,
  seekerLife: 10,
  hitRadius: 2.6,
  barrelRadius: 2.4,
  barrelLife: 50,
  shieldTime: 9,
};

// Item probability tables by rank position (front..back). Values are weights.
export function itemWeights(rank, total) {
  const t = total <= 1 ? 0 : rank / (total - 1); // 0 = leader, 1 = last
  return {
    bolt:   3,
    barrel: 2.5 * (1 - t) + 0.5,
    shield: 1.5,
    seeker: 0.5 + 3.5 * t,
    turbo:  0.5 + 4.5 * t * t,
  };
}

// Playable characters – the Eloxal Rebels cast on wheels. Small stat spreads:
// accel/top are multipliers on PHYS values.
export const CHARACTERS = [
  { id: 'al',       name: 'Alu-Al',          color: 0xd6402a, accent: 0xffffff, accel: 1.00, top: 1.00 },
  { id: 'bolle',    name: 'Bolle',           color: 0x2a6bd6, accent: 0xffd23e, accel: 1.06, top: 0.97 },
  { id: 'saeuri',   name: 'Säuri',           color: 0x7ed321, accent: 0x2d4a00, accel: 1.08, top: 0.95 },
  { id: 'titania',  name: 'Titania',         color: 0xe8e8f0, accent: 0x8a8aa0, accel: 0.96, top: 1.04 },
  { id: 'bubbles',  name: 'Bubbles',         color: 0x3ec6e0, accent: 0xffffff, accel: 1.04, top: 0.98 },
  { id: 'zink',     name: 'Zinki',           color: 0xf0c020, accent: 0x704e00, accel: 1.02, top: 0.99 },
  { id: 'ferro',    name: 'Ferro',           color: 0x6a6f78, accent: 0xb8bec8, accel: 0.94, top: 1.05 },
  { id: 'korrosius',name: 'Baron Korrosius', color: 0x9a4f1e, accent: 0x3a1c08, accel: 0.98, top: 1.03 },
];

export function characterById(id) {
  return CHARACTERS.find((c) => c.id === id) || CHARACTERS[0];
}
