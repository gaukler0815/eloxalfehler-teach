/**
 * Kleine DOM-Helfer und das Karten-/Dialogsystem der HTML-Overlays.
 * Bewusst ohne Framework - das Spiel soll leicht und offline-fest bleiben.
 */

const wurzel = () => document.getElementById('overlay');

/** el('div', { class: 'x', onClick: fn }, [kind1, 'Text']) */
export function el(tag, attrs = {}, kinder = []) {
  const node = document.createElement(tag);
  Object.entries(attrs).forEach(([k, v]) => {
    if (v === null || v === undefined || v === false) return;
    if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v;
    else if (k === 'html') node.innerHTML = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(node.style, v);
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

/** Entfernt alle offenen Overlays. */
export function overlayLeeren() {
  const w = wurzel();
  while (w.firstChild) w.removeChild(w.firstChild);
}

export function overlayAktiv() {
  return wurzel().childElementCount > 0;
}

/**
 * Zeigt eine Karte im Overlay.
 * @param {object} o
 * @param {string} [o.titel]
 * @param {string} [o.untertitel]
 * @param {HTMLElement[]} [o.inhalt]
 * @param {{text:string,klasse?:string,onClick:Function}[]} [o.knoepfe]
 * @param {boolean} [o.ersetzen] vorherige Karte entfernen (Standard: true)
 * @returns {{karte:HTMLElement, schliessen:Function}}
 */
export function karteZeigen({
  titel,
  untertitel,
  inhalt = [],
  knoepfe = [],
  ersetzen = true,
  breit = false,
} = {}) {
  if (ersetzen) overlayLeeren();

  const karte = el('div', { class: 'karte', style: breit ? { width: 'min(1100px, 100%)' } : null }, [
    titel ? el('h1', { text: titel }) : null,
    untertitel ? el('p', { class: 'lead', text: untertitel }) : null,
    ...inhalt,
    knoepfe.length
      ? el(
          'div',
          { class: 'knopf-reihe' },
          knoepfe.map((k) => knopf(k.text, k.onClick, k.klasse || ''))
        )
      : null,
  ]);

  const hintergrund = el('div', { class: 'backdrop' }, [karte]);
  wurzel().appendChild(hintergrund);
  karte.scrollTop = 0;

  return {
    karte,
    hintergrund,
    schliessen: () => hintergrund.remove(),
  };
}

/** Einfacher Ja/Nein-Dialog. */
export function frageZeigen(titel, text, onJa, onNein, jaText = 'Ja', neinText = 'Nein') {
  return karteZeigen({
    titel,
    untertitel: text,
    knoepfe: [
      { text: jaText, onClick: () => onJa?.(), klasse: '' },
      { text: neinText, onClick: () => onNein?.(), klasse: 'grau' },
    ],
  });
}

/** Kurze Jubelmeldung, verschwindet von selbst. */
export function toastZeigen(text, dauer = 1800) {
  const t = el(
    'div',
    {
      class: 'hud-chip',
      style: {
        position: 'absolute',
        left: '50%',
        top: '18%',
        transform: 'translateX(-50%)',
        fontSize: '22px',
        padding: '12px 22px',
        background: 'rgba(27,127,75,0.92)',
        zIndex: 60,
        pointerEvents: 'none',
      },
    },
    text
  );
  wurzel().appendChild(t);
  setTimeout(() => t.remove(), dauer);
  return t;
}

export const sterneText = (n) => '★'.repeat(n) + '☆'.repeat(Math.max(0, 3 - n));
