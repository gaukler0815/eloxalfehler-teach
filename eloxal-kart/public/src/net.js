// WebSocket client. Connects to the same origin that served the page (the
// Node server does both). On a static host (e.g. GitHub Pages) there is no
// game server – connect() fails fast and the menu falls back to solo mode.

export class Net {
  constructor() {
    this.ws = null;
    this.id = null;
    this.handlers = {};
    this.connected = false;
  }

  on(type, fn) { this.handlers[type] = fn; }

  connect(url = null) {
    if (!url) {
      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      url = `${proto}//${location.host}`;
    }
    return new Promise((resolve, reject) => {
      let settled = false;
      let ws;
      try {
        ws = new WebSocket(url);
      } catch (err) {
        reject(err);
        return;
      }
      const timeout = setTimeout(() => {
        if (!settled) { settled = true; ws.close(); reject(new Error('timeout')); }
      }, 4000);

      ws.onopen = () => { /* wait for welcome to resolve */ };
      ws.onmessage = (e) => {
        let msg;
        try { msg = JSON.parse(e.data); } catch { return; }
        if (msg.type === 'welcome' && !settled) {
          settled = true;
          clearTimeout(timeout);
          this.ws = ws;
          this.id = msg.id;
          this.connected = true;
          resolve(msg);
        }
        this.handlers[msg.type]?.(msg);
      };
      ws.onerror = () => {
        if (!settled) { settled = true; clearTimeout(timeout); reject(new Error('Verbindung fehlgeschlagen')); }
      };
      ws.onclose = () => {
        this.connected = false;
        this.handlers.close?.();
      };
    });
  }

  send(msg) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(msg));
  }

  close() {
    this.ws?.close();
    this.ws = null;
    this.connected = false;
  }
}
