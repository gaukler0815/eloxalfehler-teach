/* Eloxal Strike — thin WebSocket client for the multiplayer server.
 * Pure transport: connects, JSON in/out, reconnect-free (the UI handles
 * failures). Game logic lives in mp.js.
 */
(function () {
  'use strict';

  var ws = null;
  var handlers = {};

  var net = {
    /* Register callbacks: on('rooms', fn), on('room', fn), ... Unknown
     * message types are delivered to the 'peer' handler. */
    on: function (type, fn) { handlers[type] = fn; },

    connected: function () { return !!(ws && ws.readyState === 1); },

    defaultUrl: function () {
      if (location.protocol === 'http:' || location.protocol === 'https:') {
        var proto = location.protocol === 'https:' ? 'wss://' : 'ws://';
        return proto + location.host;
      }
      return 'ws://localhost:8081';
    },

    connect: function (url, name) {
      net.disconnect();
      try {
        ws = new WebSocket(url);
      } catch (e) {
        if (handlers.error) { handlers.error('Adresse ungültig: ' + url); }
        return;
      }
      ws.onopen = function () {
        net.send({ t: 'hello', name: name });
        if (handlers.open) { handlers.open(); }
      };
      ws.onclose = function () {
        ws = null;
        if (handlers.close) { handlers.close(); }
      };
      ws.onerror = function () {
        if (handlers.error) { handlers.error('Keine Verbindung zum Server.'); }
      };
      ws.onmessage = function (ev) {
        var msg;
        try { msg = JSON.parse(ev.data); } catch (e) { return; }
        var fn = handlers[msg.t] || handlers.peer;
        if (fn) { fn(msg); }
      };
    },

    disconnect: function () {
      if (ws) {
        ws.onclose = null;
        try { ws.close(); } catch (e) { /* already gone */ }
        ws = null;
      }
    },

    send: function (obj) {
      if (net.connected()) { ws.send(JSON.stringify(obj)); }
    }
  };

  window.ES = window.ES || {};
  window.ES.net = net;
})();
