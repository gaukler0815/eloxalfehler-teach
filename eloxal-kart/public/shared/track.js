// Track geometry for the "Canyon-Kurs". A closed Catmull-Rom spline over
// hand-placed control points, resampled to an arc-length table. Pure math,
// no rendering – used by the server (progress, bots, projectiles) and by the
// client (road mesh, minimap, spawn points).

// Control points of the circuit on the XZ plane (y is flat). Roughly 900x620
// units: start straight, sweeping right, S-curves through the canyon, a wide
// hairpin and a fast back straight.
const CONTROL_POINTS = [
  [0, -220], [120, -235], [240, -215], [330, -150],
  [365, -40], [330, 60], [240, 120], [200, 210],
  [255, 290], [180, 355], [60, 330], [-30, 260],
  [-140, 290], [-250, 340], [-360, 300], [-395, 190],
  [-350, 80], [-240, 40], [-190, -60], [-260, -140],
  [-350, -200], [-310, -290], [-190, -300], [-90, -250],
];

export const ROAD_HALF_WIDTH = 9;

function catmullRom(p0, p1, p2, p3, t) {
  const t2 = t * t, t3 = t2 * t;
  const f = (a, b, c, d) =>
    0.5 * ((2 * b) + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (-a + 3 * b - 3 * c + d) * t3);
  return [f(p0[0], p1[0], p2[0], p3[0]), f(p0[1], p1[1], p2[1], p3[1])];
}

export function buildTrack(samplesPerSegment = 26) {
  const n = CONTROL_POINTS.length;
  const raw = [];
  for (let i = 0; i < n; i++) {
    const p0 = CONTROL_POINTS[(i - 1 + n) % n];
    const p1 = CONTROL_POINTS[i];
    const p2 = CONTROL_POINTS[(i + 1) % n];
    const p3 = CONTROL_POINTS[(i + 2) % n];
    for (let j = 0; j < samplesPerSegment; j++) {
      raw.push(catmullRom(p0, p1, p2, p3, j / samplesPerSegment));
    }
  }

  // Arc-length table: points[], cumulative s[], tangents, left normals.
  const pts = [];
  let length = 0;
  for (let i = 0; i < raw.length; i++) {
    const [x, z] = raw[i];
    if (i > 0) {
      const dx = x - pts[i - 1].x, dz = z - pts[i - 1].z;
      length += Math.hypot(dx, dz);
    }
    pts.push({ x, z, s: length });
  }
  const closeSeg = Math.hypot(raw[0][0] - pts[pts.length - 1].x, raw[0][1] - pts[pts.length - 1].z);
  const total = length + closeSeg;

  for (let i = 0; i < pts.length; i++) {
    const a = pts[i], b = pts[(i + 1) % pts.length];
    const dx = b.x - a.x, dz = b.z - a.z;
    const d = Math.hypot(dx, dz) || 1;
    a.tx = dx / d; a.tz = dz / d;   // tangent
    a.nx = -a.tz; a.nz = a.tx;      // left normal
  }

  const track = {
    points: pts,
    length: total,
    halfWidth: ROAD_HALF_WIDTH,

    // Point on the centerline at arc position s (wrapped).
    sample(s) {
      s = ((s % total) + total) % total;
      // Binary search the arc-length table.
      let lo = 0, hi = pts.length - 1;
      while (lo < hi) {
        const mid = (lo + hi + 1) >> 1;
        if (pts[mid].s <= s) lo = mid; else hi = mid - 1;
      }
      const a = pts[lo], b = pts[(lo + 1) % pts.length];
      const segLen = (lo === pts.length - 1 ? total - a.s : b.s - a.s) || 1;
      const t = (s - a.s) / segLen;
      return {
        x: a.x + (b.x - a.x) * t,
        z: a.z + (b.z - a.z) * t,
        tx: a.tx, tz: a.tz, nx: a.nx, nz: a.nz,
      };
    },

    // Closest arc position to (x, z). With a hint the search is local (fast,
    // and stable against picking the wrong side of the circuit).
    closestS(x, z, hintS = null) {
      const N = pts.length;
      let bestI = 0, bestD = Infinity;
      if (hintS === null) {
        for (let i = 0; i < N; i++) {
          const dx = x - pts[i].x, dz = z - pts[i].z;
          const d = dx * dx + dz * dz;
          if (d < bestD) { bestD = d; bestI = i; }
        }
      } else {
        const s = ((hintS % total) + total) % total;
        let lo = 0, hi = N - 1;
        while (lo < hi) {
          const mid = (lo + hi + 1) >> 1;
          if (pts[mid].s <= s) lo = mid; else hi = mid - 1;
        }
        const win = 40; // ~40 samples ≈ enough for one physics tick of travel
        for (let k = -win; k <= win; k++) {
          const i = ((lo + k) % N + N) % N;
          const dx = x - pts[i].x, dz = z - pts[i].z;
          const d = dx * dx + dz * dz;
          if (d < bestD) { bestD = d; bestI = i; }
        }
      }
      // Project onto the segment for sub-sample accuracy.
      const a = pts[bestI], b = pts[(bestI + 1) % N];
      const abx = b.x - a.x, abz = b.z - a.z;
      const ab2 = abx * abx + abz * abz || 1;
      let t = ((x - a.x) * abx + (z - a.z) * abz) / ab2;
      t = Math.max(0, Math.min(1, t));
      const segLen = (bestI === N - 1 ? total - a.s : b.s - a.s);
      return (a.s + t * segLen) % total;
    },

    // Signed lateral offset from the centerline (positive = left of travel).
    lateralOffset(x, z, s) {
      const p = track.sample(s);
      return (x - p.x) * p.nx + (z - p.z) * p.nz;
    },
  };

  // Starting grid: 8 slots in 4 rows of 2, behind the finish line (s = 0).
  track.startGrid = [];
  for (let i = 0; i < 8; i++) {
    const row = Math.floor(i / 2), col = i % 2;
    const p = track.sample(total - 14 - row * 8);
    const side = (col === 0 ? -1 : 1) * 3.6;
    track.startGrid.push({
      x: p.x + p.nx * side,
      z: p.z + p.nz * side,
      heading: Math.atan2(p.tx, p.tz),
      s: total - 14 - row * 8,
    });
  }

  // Item box rows: three boxes across the road at fixed arc positions.
  track.itemBoxes = [];
  const rows = [0.14, 0.36, 0.58, 0.82];
  let boxId = 0;
  for (const f of rows) {
    const p = track.sample(f * total);
    for (const off of [-5, 0, 5]) {
      track.itemBoxes.push({
        id: boxId++,
        x: p.x + p.nx * off,
        z: p.z + p.nz * off,
        s: f * total,
      });
    }
  }

  return track;
}
