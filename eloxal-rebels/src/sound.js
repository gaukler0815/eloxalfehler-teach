/*
 * sound.js — all effects are synthesized with WebAudio, no audio files, so the
 * game stays a folder of text files and works offline. The context can only
 * start after a user gesture (browser autoplay policy): main.js calls unlock()
 * on the first pointerdown. Every effect is fire-and-forget and cheap.
 */
(function (global) {
  'use strict';
  const ER = (global.ER = global.ER || {});

  let actx = null;
  let master = null;
  let muted = false;

  function ensure() {
    if (actx) return actx;
    const AC = global.AudioContext || global.webkitAudioContext;
    if (!AC) return null;
    actx = new AC();
    master = actx.createGain();
    master.gain.value = 0.5;
    master.connect(actx.destination);
    return actx;
  }

  function unlock() {
    const c = ensure();
    if (c && c.state === 'suspended') c.resume();
  }

  function setMuted(m) {
    muted = m;
    if (master) master.gain.value = m ? 0 : 0.5;
  }

  function ready() { return actx && actx.state === 'running' && !muted; }

  // One decaying oscillator note.
  function tone(freq, opts) {
    if (!ready()) return;
    const o = opts || {};
    const t0 = actx.currentTime + (o.delay || 0);
    const dur = o.dur || 0.18;
    const osc = actx.createOscillator();
    const g = actx.createGain();
    osc.type = o.type || 'sine';
    osc.frequency.setValueAtTime(freq, t0);
    if (o.slideTo) osc.frequency.exponentialRampToValueAtTime(Math.max(30, o.slideTo), t0 + dur);
    g.gain.setValueAtTime(o.vol || 0.25, t0);
    g.gain.exponentialRampToValueAtTime(0.001, t0 + dur);
    osc.connect(g); g.connect(master);
    osc.start(t0); osc.stop(t0 + dur + 0.03);
  }

  // Filtered noise burst — thuds, crashes, blasts.
  function noiseBurst(opts) {
    if (!ready()) return;
    const o = opts || {};
    const dur = o.dur || 0.2;
    const t0 = actx.currentTime + (o.delay || 0);
    const len = Math.max(1, Math.floor(actx.sampleRate * dur));
    const buf = actx.createBuffer(1, len, actx.sampleRate);
    const data = buf.getChannelData(0);
    for (let i = 0; i < len; i++) data[i] = (Math.random() * 2 - 1) * (1 - i / len);
    const src = actx.createBufferSource();
    src.buffer = buf;
    const f = actx.createBiquadFilter();
    f.type = o.filter || 'lowpass';
    f.frequency.setValueAtTime(o.freq || 700, t0);
    if (o.freqTo) f.frequency.exponentialRampToValueAtTime(o.freqTo, t0 + dur);
    f.Q.value = o.q || 0.8;
    const g = actx.createGain();
    g.gain.setValueAtTime(o.vol || 0.3, t0);
    g.gain.exponentialRampToValueAtTime(0.001, t0 + dur);
    src.connect(f); f.connect(g); g.connect(master);
    src.start(t0);
  }

  // --- the actual effects --------------------------------------------------
  let lastStretch = 0;
  const sfx = {
    // Rubber creak while pulling; pitch follows pull strength (0..1).
    stretch(pull) {
      const now = Date.now();
      if (now - lastStretch < 90) return; // don't machine-gun the creak
      lastStretch = now;
      tone(140 + pull * 260, { type: 'triangle', dur: 0.06, vol: 0.06 });
    },
    launch() {
      noiseBurst({ dur: 0.22, freq: 1400, freqTo: 300, vol: 0.22, filter: 'bandpass', q: 1.4 });
      tone(340, { type: 'triangle', dur: 0.18, vol: 0.14, slideTo: 90 });
    },
    impact(strength) {
      const s = Math.min(1, strength / 18);
      noiseBurst({ dur: 0.1 + s * 0.12, freq: 500 - s * 200, vol: 0.1 + s * 0.28 });
      if (s > 0.4) tone(90, { type: 'sine', dur: 0.12, vol: 0.18, slideTo: 45 });
    },
    thudSoft() {
      noiseBurst({ dur: 0.1, freq: 300, vol: 0.14 });
    },
    shatter() {
      noiseBurst({ dur: 0.28, freq: 3200, freqTo: 900, vol: 0.26, filter: 'highpass' });
      tone(1250, { type: 'square', dur: 0.1, vol: 0.06, slideTo: 700 });
    },
    crack() {
      noiseBurst({ dur: 0.16, freq: 1200, freqTo: 400, vol: 0.24, filter: 'bandpass', q: 1.2 });
    },
    pop() {
      tone(520, { type: 'square', dur: 0.09, vol: 0.16, slideTo: 130 });
      noiseBurst({ dur: 0.07, freq: 1800, vol: 0.1, filter: 'bandpass', q: 2 });
    },
    // The arc — deliberately the loudest moment in the game (design bible).
    zap() {
      noiseBurst({ dur: 0.34, freq: 5200, freqTo: 1400, vol: 0.4, filter: 'highpass', q: 0.7 });
      tone(70, { type: 'sawtooth', dur: 0.3, vol: 0.3, slideTo: 42 });
      tone(2400, { type: 'square', dur: 0.12, vol: 0.1, slideTo: 3600 });
    },
    boom() {
      noiseBurst({ dur: 0.5, freq: 900, freqTo: 90, vol: 0.42 });
      tone(60, { type: 'sine', dur: 0.42, vol: 0.34, slideTo: 34 });
    },
    win() {
      [392, 494, 587, 784].forEach((f, i) => tone(f, { type: 'triangle', dur: 0.22, vol: 0.2, delay: i * 0.13 }));
    },
    lose() {
      [330, 262, 196].forEach((f, i) => tone(f, { type: 'triangle', dur: 0.3, vol: 0.18, delay: i * 0.18 }));
    },
    click() {
      tone(700, { type: 'square', dur: 0.04, vol: 0.06 });
    }
  };

  ER.sound = Object.assign({ unlock, setMuted, isMuted: () => muted }, sfx);
})(typeof window !== 'undefined' ? window : globalThis);
