/** Winzige DOM-Helfer - die App kommt ohne Framework aus. */

export function el(tag, attrs = {}, kinder = []) {
  const node = document.createElement(tag);
  Object.entries(attrs).forEach(([k, v]) => {
    if (v === null || v === undefined || v === false) return;
    if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v;
    else if (k === 'style' && typeof v === 'object') {
      // CSS-Variablen (--name) brauchen setProperty, normale Angaben nicht.
      Object.entries(v).forEach(([eigenschaft, wert]) => {
        if (eigenschaft.startsWith('--')) node.style.setProperty(eigenschaft, wert);
        else node.style[eigenschaft] = wert;
      });
    }
    else if (k.startsWith('on') && typeof v === 'function') {
      node.addEventListener(k.slice(2).toLowerCase(), v);
    } else node.setAttribute(k, v);
  });
  (Array.isArray(kinder) ? kinder : [kinder]).forEach((kind) => {
    if (kind === null || kind === undefined || kind === false) return;
    node.appendChild(typeof kind === 'string' ? document.createTextNode(kind) : kind);
  });
  return node;
}

export function knopf(text, onClick, klasse = '') {
  return el('button', { class: `knopf ${klasse}`.trim(), type: 'button', onClick }, text);
}

/** Ersetzt den kompletten Bildschirminhalt durch eine Karte. */
export function zeigen(...inhalt) {
  const app = document.getElementById('app');
  app.replaceChildren(...inhalt);
  window.scrollTo(0, 0);
}

export function karte(kinder, klasse = '') {
  return el('div', { class: `karte ${klasse}`.trim() }, kinder);
}

/** Fisher-Yates - mischt eine Kopie der Liste. */
export function mischen(liste) {
  const a = [...liste];
  for (let i = a.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export const sterneText = (n) => '⭐'.repeat(n) + '☆'.repeat(Math.max(0, 3 - n));

/** Setzt die Farbe der aktuellen Welt fuer die ganze Oberflaeche. */
export function weltFarbe(farbe) {
  document.documentElement.style.setProperty('--welt', farbe || '#1b7f4b');
}
