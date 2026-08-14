// Jacobi Eloxal corporate branding, shared by menu, world textures and kart
// decals. Vector rebuild of the round JX badge + wordmark, identical to the
// one used in Eloxal Rebels (src/render.js drawLogoSign / index.html SVG).

export const BRAND = {
  blue: '#1E74DC',
  blueLight: '#8FD3FF',
  gold: '#F5A81C',
  silver: '#C9D2DC',
  line: '#17131F',
  hallDark: '#241D33',
  hallMid: '#332A47',
};

// Draws the JX badge (and optionally the wordmark) onto a 2D canvas context.
// The badge is a circle of radius 34 around (0,0) at scale 1; the wordmark
// extends to the right, ~230px wide in total.
export function drawLogo(g, x, y, s = 1, { wordmark = true, light = false } = {}) {
  g.save();
  g.translate(x, y);
  g.scale(s, s);
  const hg = g.createLinearGradient(-34, -34, 34, 34);
  hg.addColorStop(0, BRAND.blue);
  hg.addColorStop(1, BRAND.blueLight);
  g.fillStyle = hg;
  g.beginPath();
  g.arc(0, 0, 34, 0, Math.PI * 2);
  g.fill();
  g.strokeStyle = BRAND.line;
  g.lineWidth = 5;
  g.stroke();
  g.fillStyle = '#fff';
  g.font = '800 30px system-ui, sans-serif';
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  g.fillText('JX', 0, 2);
  if (wordmark) {
    g.textAlign = 'left';
    g.textBaseline = 'alphabetic';
    g.fillStyle = light ? '#ffffff' : BRAND.silver;
    g.font = '800 44px system-ui, sans-serif';
    g.fillText('JACOBI', 52, -2);
    g.fillStyle = BRAND.gold;
    g.font = '700 30px system-ui, sans-serif';
    g.fillText('E L O X A L', 53, 32);
  }
  g.restore();
}

// The same logo as an inline SVG string, for HTML panels (menu/lobby/results).
export const LOGO_SVG = `
<svg viewBox="0 0 300 74" width="240" height="60" aria-label="Jacobi Eloxal">
  <defs>
    <linearGradient id="jxg-kart" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="${BRAND.blue}"/><stop offset="1" stop-color="${BRAND.blueLight}"/>
    </linearGradient>
  </defs>
  <circle cx="37" cy="37" r="32" fill="url(#jxg-kart)" stroke="${BRAND.line}" stroke-width="4"/>
  <text x="37" y="49" text-anchor="middle" font-family="system-ui,sans-serif" font-size="32" font-weight="800" letter-spacing="1" fill="#fff">JX</text>
  <text x="82" y="36" font-family="system-ui,sans-serif" font-size="30" font-weight="800" fill="${BRAND.silver}">JACOBI</text>
  <text x="83" y="62" font-family="system-ui,sans-serif" font-size="20" font-weight="700" letter-spacing="6" fill="${BRAND.gold}">ELOXAL</text>
</svg>`;
