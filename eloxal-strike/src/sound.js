/* Eloxal Strike — synthesized WebAudio sound, no audio files needed.
 * Mirrors the approach used in eloxal-rebels: tiny procedural blips built
 * from oscillators + noise buffers, all routed through one master gain.
 */
(function () {
  'use strict';

  var ctx = null;
  var master = null;
  var muted = false;
  var droneNodes = null;

  function ensure() {
    if (ctx) {
      if (ctx.state === 'suspended') { ctx.resume(); }
      return true;
    }
    var AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) { return false; }
    ctx = new AC();
    master = ctx.createGain();
    master.gain.value = 0.5;
    master.connect(ctx.destination);
    return true;
  }

  function noiseBuffer(seconds) {
    var len = Math.max(1, Math.floor(ctx.sampleRate * seconds));
    var buf = ctx.createBuffer(1, len, ctx.sampleRate);
    var d = buf.getChannelData(0);
    for (var i = 0; i < len; i++) { d[i] = Math.random() * 2 - 1; }
    return buf;
  }

  function env(node, t0, attack, peak, decay) {
    node.gain.setValueAtTime(0.0001, t0);
    node.gain.linearRampToValueAtTime(peak, t0 + attack);
    node.gain.exponentialRampToValueAtTime(0.0001, t0 + attack + decay);
  }

  function blip(freqFrom, freqTo, dur, type, vol) {
    if (muted || !ensure()) { return; }
    var t = ctx.currentTime;
    var o = ctx.createOscillator();
    var g = ctx.createGain();
    o.type = type || 'square';
    o.frequency.setValueAtTime(freqFrom, t);
    o.frequency.exponentialRampToValueAtTime(Math.max(1, freqTo), t + dur);
    env(g, t, 0.005, vol || 0.2, dur);
    o.connect(g); g.connect(master);
    o.start(t); o.stop(t + dur + 0.05);
  }

  function noiseBurst(dur, vol, filterFreq) {
    if (muted || !ensure()) { return; }
    var t = ctx.currentTime;
    var src = ctx.createBufferSource();
    src.buffer = noiseBuffer(dur);
    var f = ctx.createBiquadFilter();
    f.type = 'lowpass';
    f.frequency.value = filterFreq || 1800;
    var g = ctx.createGain();
    env(g, t, 0.003, vol, dur);
    src.connect(f); f.connect(g); g.connect(master);
    src.start(t);
  }

  var sound = {
    unlock: function () { ensure(); },

    setMuted: function (m) {
      muted = m;
      if (m) { sound.stopDrone(); } else { sound.startDrone(); }
    },
    isMuted: function () { return muted; },

    shoot: function (weaponId) {
      if (weaponId === 'streuer') {
        noiseBurst(0.28, 0.5, 900);
        blip(160, 40, 0.22, 'sawtooth', 0.25);
      } else if (weaponId === 'lichtbogen') {
        noiseBurst(0.07, 0.22, 3200);
        blip(900, 220, 0.06, 'square', 0.12);
      } else {
        noiseBurst(0.1, 0.3, 2200);
        blip(520, 90, 0.12, 'square', 0.2);
      }
    },

    empty: function () { blip(300, 240, 0.08, 'square', 0.12); },

    reload: function () {
      blip(700, 500, 0.06, 'square', 0.12);
      setTimeout(function () { blip(500, 750, 0.06, 'square', 0.12); }, 140);
    },

    hit: function () { blip(1100, 700, 0.05, 'triangle', 0.16); },

    headshot: function () {
      blip(1400, 900, 0.06, 'triangle', 0.2);
      blip(2100, 1400, 0.08, 'sine', 0.12);
    },

    kill: function () {
      noiseBurst(0.25, 0.3, 700);
      blip(220, 40, 0.3, 'sawtooth', 0.2);
    },

    hurt: function () {
      blip(180, 60, 0.25, 'sawtooth', 0.3);
      noiseBurst(0.15, 0.2, 500);
    },

    spit: function () { blip(600, 200, 0.18, 'sine', 0.14); },
    splash: function () { noiseBurst(0.2, 0.25, 1200); },

    pickup: function () {
      blip(660, 990, 0.09, 'sine', 0.2);
      setTimeout(function () { blip(880, 1320, 0.12, 'sine', 0.2); }, 90);
    },

    waveStart: function () {
      blip(110, 220, 0.4, 'sawtooth', 0.22);
      setTimeout(function () { blip(150, 300, 0.4, 'sawtooth', 0.22); }, 250);
    },

    waveClear: function () {
      [523, 659, 784, 1047].forEach(function (f, i) {
        setTimeout(function () { blip(f, f, 0.16, 'triangle', 0.2); }, i * 110);
      });
    },

    gameOver: function () {
      [392, 311, 233, 155].forEach(function (f, i) {
        setTimeout(function () { blip(f, f * 0.97, 0.35, 'sawtooth', 0.2); }, i * 260);
      });
    },

    /* Low ominous hall drone under everything. */
    startDrone: function () {
      if (muted || !ensure() || droneNodes) { return; }
      var g = ctx.createGain();
      g.gain.value = 0.05;
      var o1 = ctx.createOscillator();
      o1.type = 'sawtooth'; o1.frequency.value = 55;
      var o2 = ctx.createOscillator();
      o2.type = 'sine'; o2.frequency.value = 55.7;
      var f = ctx.createBiquadFilter();
      f.type = 'lowpass'; f.frequency.value = 160;
      var lfo = ctx.createOscillator();
      lfo.frequency.value = 0.11;
      var lfoGain = ctx.createGain();
      lfoGain.gain.value = 60;
      lfo.connect(lfoGain); lfoGain.connect(f.frequency);
      o1.connect(f); o2.connect(f); f.connect(g); g.connect(master);
      o1.start(); o2.start(); lfo.start();
      droneNodes = { g: g, oscs: [o1, o2, lfo] };
    },

    stopDrone: function () {
      if (!droneNodes) { return; }
      try {
        droneNodes.oscs.forEach(function (o) { o.stop(); });
      } catch (e) { /* already stopped */ }
      droneNodes.g.disconnect();
      droneNodes = null;
    }
  };

  window.ES = window.ES || {};
  window.ES.sound = sound;
})();
