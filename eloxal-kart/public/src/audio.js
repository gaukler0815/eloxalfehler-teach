// Procedural WebAudio sound: engine hum that follows speed, drift scrape,
// boost whoosh, item pickup, explosion, countdown beeps and a finish
// fanfare. No audio files needed.

export class SoundKit {
  constructor() {
    this.ctx = null;
    this.engine = null;
    this.muted = false;
  }

  // Must be called from a user gesture (browser autoplay policy).
  init() {
    if (this.ctx) return;
    try {
      this.ctx = new (window.AudioContext || window.webkitAudioContext)();
      this.master = this.ctx.createGain();
      this.master.gain.value = 0.5;
      this.master.connect(this.ctx.destination);
    } catch { /* no audio available */ }
  }

  startEngine() {
    if (!this.ctx || this.engine) return;
    const osc = this.ctx.createOscillator();
    osc.type = 'sawtooth';
    osc.frequency.value = 40;
    const osc2 = this.ctx.createOscillator();
    osc2.type = 'square';
    osc2.frequency.value = 41;
    const gain = this.ctx.createGain();
    gain.gain.value = 0.0;
    const filter = this.ctx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.value = 400;
    osc.connect(filter); osc2.connect(filter);
    filter.connect(gain); gain.connect(this.master);
    osc.start(); osc2.start();
    this.engine = { osc, osc2, gain, filter };
  }

  setEngine(speedFrac, boosting) {
    if (!this.engine) return;
    const f = 42 + speedFrac * 120 + (boosting ? 30 : 0);
    this.engine.osc.frequency.setTargetAtTime(f, this.ctx.currentTime, 0.08);
    this.engine.osc2.frequency.setTargetAtTime(f * 1.007, this.ctx.currentTime, 0.08);
    this.engine.gain.gain.setTargetAtTime(0.05 + speedFrac * 0.06, this.ctx.currentTime, 0.1);
    this.engine.filter.frequency.setTargetAtTime(300 + speedFrac * 900, this.ctx.currentTime, 0.1);
  }

  stopEngine() {
    if (!this.engine) return;
    try { this.engine.osc.stop(); this.engine.osc2.stop(); } catch { /* noop */ }
    this.engine = null;
  }

  blip(freq, dur = 0.12, type = 'square', vol = 0.25, slide = 0) {
    if (!this.ctx) return;
    const t = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, t);
    if (slide) osc.frequency.exponentialRampToValueAtTime(Math.max(30, freq + slide), t + dur);
    const g = this.ctx.createGain();
    g.gain.setValueAtTime(vol, t);
    g.gain.exponentialRampToValueAtTime(0.001, t + dur);
    osc.connect(g); g.connect(this.master);
    osc.start(t); osc.stop(t + dur + 0.02);
  }

  noise(dur = 0.3, vol = 0.3, freq = 800) {
    if (!this.ctx) return;
    const t = this.ctx.currentTime;
    const len = Math.floor(this.ctx.sampleRate * dur);
    const buf = this.ctx.createBuffer(1, len, this.ctx.sampleRate);
    const d = buf.getChannelData(0);
    for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * (1 - i / len);
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    const f = this.ctx.createBiquadFilter();
    f.type = 'bandpass'; f.frequency.value = freq; f.Q.value = 0.8;
    const g = this.ctx.createGain();
    g.gain.value = vol;
    src.connect(f); f.connect(g); g.connect(this.master);
    src.start(t);
  }

  countdownBeep(final) { this.blip(final ? 880 : 440, final ? 0.5 : 0.18, 'square', 0.3); }
  pickup() { this.blip(660, 0.09, 'square', 0.2, 500); this.blip(990, 0.12, 'square', 0.15); }
  itemReady() { this.blip(784, 0.1, 'triangle', 0.25, 300); }
  fire() { this.noise(0.15, 0.25, 1600); this.blip(300, 0.15, 'sawtooth', 0.2, -150); }
  explosion() { this.noise(0.5, 0.5, 300); this.blip(90, 0.4, 'sawtooth', 0.35, -50); }
  shieldPop() { this.blip(1200, 0.2, 'sine', 0.3, -700); }
  boost() { this.noise(0.4, 0.3, 2400); this.blip(220, 0.35, 'sawtooth', 0.2, 400); }
  lap() { this.blip(523, 0.1, 'square', 0.2); setTimeout(() => this.blip(659, 0.15, 'square', 0.2), 110); }
  spin() { this.blip(400, 0.4, 'sawtooth', 0.3, -300); }

  fanfare(win) {
    if (!this.ctx) return;
    const notes = win ? [523, 659, 784, 1047, 784, 1047] : [392, 330, 262];
    notes.forEach((n, i) => setTimeout(() => this.blip(n, 0.22, 'triangle', 0.3), i * 140));
  }
}
