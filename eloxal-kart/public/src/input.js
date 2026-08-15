// Keyboard + touch input. Keys: WASD / arrows to drive, Space or Shift to
// drift (hop), Enter / Ctrl / E to fire the item.

export class Input {
  constructor() {
    this.keys = new Set();
    this.firePressed = false; // edge-triggered
    this.touch = { left: false, right: false, gas: false, drift: false };

    window.addEventListener('keydown', (e) => {
      if (e.repeat) return;
      this.keys.add(e.code);
      if (['Enter', 'ControlLeft', 'ControlRight', 'KeyE'].includes(e.code)) this.firePressed = true;
      if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space'].includes(e.code)) e.preventDefault();
    });
    window.addEventListener('keyup', (e) => this.keys.delete(e.code));
    window.addEventListener('blur', () => this.keys.clear());

    // Touch buttons (shown automatically on touch devices).
    if ('ontouchstart' in window) {
      document.body.classList.add('touch');
      const bind = (id, prop) => {
        const el = document.getElementById(id);
        if (!el) return;
        const on = (e) => { e.preventDefault(); this.touch[prop] = true; if (prop === 'item') this.firePressed = true; };
        const off = (e) => { e.preventDefault(); this.touch[prop] = false; };
        el.addEventListener('touchstart', on, { passive: false });
        el.addEventListener('touchend', off, { passive: false });
        el.addEventListener('touchcancel', off, { passive: false });
      };
      bind('t-left', 'left');
      bind('t-right', 'right');
      bind('t-gas', 'gas');
      bind('t-drift', 'drift');
      bind('t-item', 'item');
    }
  }

  // Returns { throttle, steer, drift, fire } – fire only true for one read.
  // Steering convention: positive steer = counterclockwise yaw = LEFT on
  // screen (heading maps to dir via (sin h, cos h)), so the left keys add +1.
  read() {
    const k = this.keys;
    let throttle = 0, steer = 0;
    if (k.has('KeyW') || k.has('ArrowUp') || this.touch.gas) throttle += 1;
    if (k.has('KeyS') || k.has('ArrowDown')) throttle -= 1;
    if (k.has('KeyA') || k.has('ArrowLeft') || this.touch.left) steer += 1;
    if (k.has('KeyD') || k.has('ArrowRight') || this.touch.right) steer -= 1;
    const drift = k.has('Space') || k.has('ShiftLeft') || k.has('ShiftRight') || this.touch.drift;
    const fire = this.firePressed;
    this.firePressed = false;
    return { throttle, steer, drift, fire };
  }
}
