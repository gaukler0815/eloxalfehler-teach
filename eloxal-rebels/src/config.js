/*
 * config.js
 * Central place for all balancing and physics numbers, colors, materials and
 * projectiles. Nothing here touches rendering or the DOM, so values can be
 * tuned without reading the rest of the code base (see CLAUDE.md).
 *
 * The project runs by opening index.html directly (file://), where ES module
 * imports are blocked by the browser. Everything therefore attaches to a single
 * global namespace `ER` and the files are loaded in order via <script> tags.
 */
(function (global) {
  'use strict';
  const ER = (global.ER = global.ER || {});

  // World is a fixed 1920x1080 coordinate space. The canvas is scaled to the
  // window; physics always runs in these world coordinates, never in pixels.
  const WORLD = { width: 1920, height: 1080, groundY: 980 };

  // Eloxal colour card (CLAUDE.md). `line` is the shared contour colour.
  const PALETTE = {
    alu: '#C9D2DC',
    blau: '#1E74DC',
    gold: '#F5A81C',
    rot: '#E33A2C',
    titan: '#B4A996',
    line: '#17131F',
    // background / support tones derived from the card
    hallDark: '#211B2B',
    hallMid: '#2E2740',
    bath: '#123',
    arc: '#8FD3FF'
  };

  // Physics / balancing constants. Tuned for the large 1920x1080 world.
  const PHYSICS = {
    gravityY: 1.4,          // Matter world gravity scale multiplier
    launchPower: 0.12,      // slingshot pull distance -> launch speed
    maxPull: 280,           // max slingshot stretch in world units
    grabRadius: 140,        // how close a pointer must be to grab the slingshot
    dmgFactor: 1.0,         // global multiplier on impact damage to blocks
    enemyKillImpact: 5,     // relative speed that finishes a normal enemy
    kalkiKillImpact: 13,    // Kalki must be done in one strong hit
    barrelBlastRadius: 240, // Säurefass explosion radius
    acidRadius: 190,        // Säuri soften radius
    pelletTtl: 55,          // Bürsti pellet lifetime in frames
    settleSpeed: 0.6,       // below this speed a body counts as at rest
    settleFrames: 40,       // frames of quiet before the shot is over
    offWorldY: 1240,        // below this a body (or enemy) is gone
    previewGravity: 0.38    // gravity used only for the aiming trajectory dots
  };

  /*
   * Materials. Five build materials plus the acid barrel, exactly as in the
   * Game Design Bible.
   *  - density      relative mass
   *  - restitution  bounciness (Kunststoff bounces, everything else little)
   *  - friction     surface friction
   *  - hp           impulse a body absorbs before it breaks
   *  - conductive   carries the anodizing arc (Alu + Steel), or insulates
   *  - color        fill colour from the palette
   */
  const MATERIALS = {
    // Schutzfolie / Kartonage: very light, damps hits, insulates. Bürsti clears it.
    kartonage: {
      label: 'Kartonage', density: 0.0006, restitution: 0.02, friction: 0.9,
      hp: 14, conductive: false, color: '#D8C39A', light: true
    },
    // Kunststoff-Distanzstück: bounces shots back, insulates.
    kunststoff: {
      label: 'Kunststoff', density: 0.0012, restitution: 0.72, friction: 0.4,
      hp: 40, conductive: false, color: '#1E74DC'
    },
    // Rohaluminium: medium mass, bends rather than shatters, conducts.
    aluminium: {
      label: 'Rohaluminium', density: 0.0020, restitution: 0.05, friction: 0.6,
      hp: 70, conductive: true, color: '#C9D2DC'
    },
    // Stahlgestell: heavy, hard, conducts excellently.
    stahl: {
      label: 'Stahlgestell', density: 0.0045, restitution: 0.03, friction: 0.7,
      hp: 240, conductive: true, color: '#8A94A0'
    },
    // Fehlcharge-Eloxal: brittle, shatters on first real contact. The glass.
    fehlcharge: {
      label: 'Fehlcharge-Eloxal', density: 0.0015, restitution: 0.0, friction: 0.5,
      hp: 22, conductive: false, color: '#F5A81C', brittle: true
    },
    // Säurefass: explodes in a radius and etches neighbours.
    saeurefass: {
      label: 'Säurefass', density: 0.0018, restitution: 0.1, friction: 0.5,
      hp: 18, conductive: false, color: '#7ED957', explosive: true
    },
    // Stromschiene: the power rail. Conductive and where the arc starts.
    rail: {
      label: 'Stromschiene', density: 0.006, restitution: 0.02, friction: 0.7,
      hp: 100000, conductive: true, color: '#F5A81C', rail: true, static: true
    }
  };

  /*
   * Projectiles. Eight rebels, each with exactly one ability triggered by a
   * tap/click in flight (Lasar once per level). `mass` scales the body.
   * `ability` keys are handled in abilities.js.
   */
  const PROJECTILES = {
    ali:     { label: 'Ali',     hint: 'Blechzuschnitt · kein Trick',        color: PALETTE.alu,  radius: 26, mass: 1.0,  ability: null },
    bolle:   { label: 'Bolle',   hint: 'Drehteil · Sturzflug/Durchschlag',   color: PALETTE.titan, radius: 30, mass: 1.8,  ability: 'dive' },
    rippi:   { label: 'Rippi',   hint: 'Kühlkörper · Streuung in 3',         color: PALETTE.alu,  radius: 24, mass: 0.7,  ability: 'split' },
    titania: { label: 'Titania', hint: 'Titanhaken · reißt Kanten herunter',  color: PALETTE.titan, radius: 26, mass: 1.1,  ability: 'hook' },
    bubbles: { label: 'Bubbles', hint: 'H₂-Blase · Auftrieb, oben zünden',    color: PALETTE.blau, radius: 24, mass: 0.4,  ability: 'lift' },
    saeuri:  { label: 'Säuri',   hint: 'Säuretropfen · weicht Fläche an',     color: '#7ED957', radius: 25, mass: 0.8,  ability: 'acid' },
    buersti: { label: 'Bürsti',  hint: 'Strahlkopf · Schrot nach vorn',      color: PALETTE.gold, radius: 26, mass: 0.9,  ability: 'blast' },
    lasar:   { label: 'Lasar',   hint: 'Laserkopf · Schnittlinie (1×)',       color: PALETTE.rot,  radius: 24, mass: 0.9,  ability: 'cut' }
  };

  /*
   * Enemies — every one is a real surface defect from the shop floor.
   *  hp is informational; most die from one solid hit or from an arc.
   */
  const ENEMIES = {
    stauber:   { label: 'Stauber',   radius: 24, color: '#6E6A7A', face: 'dots' },
    lenny:     { label: 'Lochfraß-Lenny', radius: 30, color: '#3B7A57', face: 'drill' },
    fetti:     { label: 'Fetti',     radius: 28, color: '#C7B24A', face: 'oil' },
    kalki:     { label: 'Kalki',     radius: 30, color: '#E7E2D6', face: 'crystal' },
    korrosius: { label: 'Baron Korrosius', radius: 54, color: '#F0EAF2', face: 'boss' }
  };

  // Scoring in micrometres of anodized layer. Shots left over after clearing
  // the level decide the grade. Max 20 µm per level (no hard anodizing).
  const SCORING = {
    perfect: 20, // 2+ shots left over
    good: 12,    // 1 shot left over
    pass: 6,     // cleared, no shots to spare
    fail: 0,
    maxPerLevel: 20
  };

  ER.config = { WORLD, PALETTE, PHYSICS, MATERIALS, PROJECTILES, ENEMIES, SCORING };
})(typeof window !== 'undefined' ? window : globalThis);
